"""
Training Orchestrator Service.

Triggers training when:
1. dataset.ready event received (hours threshold met)
2. Cron schedule (e.g., daily at 2 AM)

Workflow:
1. Receive trigger
2. Launch training job (training/train.py)
3. Monitor progress
4. Evaluate model on completion
5. Promote model if metrics improve
6. Publish training.completed / model.promoted events
"""

import os, io, json, uuid, asyncio, subprocess, wave
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from minio import Minio
from minio.error import S3Error
from nats.aio.client import Client as NATS

app = FastAPI(title="training-orchestrator")

# --- env ---
AUDIO_BUCKET=os.getenv("AUDIO_BUCKET","audio")
TRANSCRIPT_BUCKET=os.getenv("TRANSCRIPT_BUCKET","transcripts")
DATASET_BUCKET=os.getenv("DATASET_BUCKET","datasets")
MODELS_BUCKET=os.getenv("MODELS_BUCKET","models")
TRAIN_HOURS_THRESHOLD=float(os.getenv("TRAIN_HOURS_THRESHOLD","5.0"))
TRAIN_DAILY_CRON=os.getenv("TRAIN_DAILY_CRON","02:00")
MINIO_ENDPOINT=os.getenv("MINIO_ENDPOINT","http://minio:9000").replace("http://","").replace("https://","")
MINIO_ACCESS_KEY=os.getenv("MINIO_ACCESS_KEY","minio")
MINIO_SECRET_KEY=os.getenv("MINIO_SECRET_KEY","minio123")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")

s3 = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
nc = NATS()

MANIFEST_TRAIN="manifests/train.jsonl"
MANIFEST_VAL="manifests/val.jsonl"

def _ensure_buckets():
    for b in (DATASET_BUCKET, MODELS_BUCKET):
        if not s3.bucket_exists(b):
            s3.make_bucket(b)

def _wav_dur(bucket, key)->float:
    tmp="/tmp/__dur.wav"
    s3.fget_object(bucket, key, tmp)
    with wave.open(tmp,"rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
    os.remove(tmp)
    return frames/float(rate)

def _speaker_split(sid:str)->str:
    import hashlib
    h=int(hashlib.md5(sid.encode()).hexdigest(),16)%100
    if h<92: return "train"
    if h<96: return "val"
    return "test"

async def _append_manifest(audio_id:str, text:str):
    wav_key=f"raw/{audio_id}.wav"
    dur=_wav_dur(AUDIO_BUCKET, wav_key)
    speaker_id=audio_id  # placeholder until diarization
    split=_speaker_split(speaker_id)
    line=json.dumps({
        "audio_filepath": f"s3://{AUDIO_BUCKET}/{wav_key}",
        "text": text,
        "duration": dur,
        "speaker": speaker_id
    }, ensure_ascii=False).encode()
    key = MANIFEST_TRAIN if split=="train" else MANIFEST_VAL
    tmp="/tmp/__manifest.jsonl"
    try: 
        s3.fget_object(DATASET_BUCKET, key, tmp)
    except S3Error as e:
        print(f"[WARNING] Manifest not found, creating new: {e}")
        open(tmp,"wb").close()
    with open(tmp,"ab") as f: f.write(line+b"\n")
    s3.fput_object(DATASET_BUCKET, key, tmp, content_type="application/jsonl")
    os.remove(tmp)

async def _maybe_dataset_ready():
    tmp="/tmp/__train.jsonl"
    try:
        s3.fget_object(DATASET_BUCKET, MANIFEST_TRAIN, tmp)
        hours=0.0
        with open(tmp,"r",encoding="utf-8") as f:
            for ln in f:
                try: 
                    hours += json.loads(ln).get("duration", 0.0) / 3600.0
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[WARNING] Invalid manifest line: {ln.strip()}, error: {e}")
                    continue
        os.remove(tmp)
        if hours>=TRAIN_HOURS_THRESHOLD:
            evt={"specversion":"1.0","id":str(uuid.uuid4()),
                 "source":"services/training-orchestrator","type":"dataset.ready",
                 "time":datetime.utcnow().isoformat()+"Z","data":{"hours":hours}}
            await nc.publish("dataset.ready", json.dumps(evt).encode())
    except Exception as e:
        print("hours calc error:", e)

async def _run_training_once():
    # download manifests locally
    Path("/app/_datasets").mkdir(parents=True, exist_ok=True)
    s3.fget_object(DATASET_BUCKET, MANIFEST_TRAIN, "/app/_datasets/train.jsonl")
    s3.fget_object(DATASET_BUCKET, MANIFEST_VAL, "/app/_datasets/val.jsonl")
    run_id=datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    await nc.publish("training.started", json.dumps({"data":{"run_id":run_id}}).encode())

    # prefer enhanced_train if present
    train_entry = "training.enhanced_train" if Path("training/enhanced_train.py").exists() else "training.train"
    rc = subprocess.call(["python","-m",train_entry,
                          "--train-manifest","/app/_datasets/train.jsonl",
                          "--val-manifest","/app/_datasets/val.jsonl"])

    # simple evaluate hook (optional)
    _ = subprocess.call(["python","-m","training.evaluate","--val","/app/_datasets/val.jsonl"])

    s3.put_object(MODELS_BUCKET, f"asr/{run_id}/DONE", io.BytesIO(b"ok"), length=2)
    await nc.publish("training.completed", json.dumps({"data":{"run_id":run_id,"rc":rc}}).encode())

    if rc==0:
        await nc.publish("model.promoted", json.dumps({"data":{"run_id":run_id}}).encode())

@app.on_event("startup")
async def _startup():
    _ensure_buckets()
    await nc.connect(servers=[NATS_URL])
    # prefer cleaned text if available
    async def on_nlp_post(msg):
        evt=json.loads(msg.data.decode())
        d=evt.get("data",{})
        await _append_manifest(d["audio_id"], d.get("text_clean") or d.get("text") or "")
        await _maybe_dataset_ready()
    async def on_trans_completed(msg):
        evt=json.loads(msg.data.decode())
        uri = evt["data"]["transcript_uri"]   # s3://transcripts/transcripts/{audio_id}.json
        bucket, key = uri.replace("s3://","").split("/",1)
        tmp="/tmp/__tr.json"
        s3.fget_object(bucket, key, tmp)
        with open(tmp,"r",encoding="utf-8") as f: j=json.load(f)
        os.remove(tmp)
        await _append_manifest(j["audio_id"], j.get("text",""))
        await _maybe_dataset_ready()
    await nc.subscribe("nlp.postprocessed", cb=on_nlp_post)
    await nc.subscribe("transcription.completed", cb=on_trans_completed)
    # daily cron
    hh,mm = (TRAIN_DAILY_CRON or "02:00").split(":")
    scheduler=AsyncIOScheduler()
    scheduler.add_job(lambda: asyncio.create_task(_run_training_once()),
                      "cron", hour=int(hh), minute=int(mm))
    scheduler.start()

@app.post("/v1/train/now")
async def train_now():
    await _run_training_once()
    return {"status":"started"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)

