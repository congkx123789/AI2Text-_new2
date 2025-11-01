"""Streaming ASR server.

This WebSocket endpoint accepts PCM16 audio frames and emits
partial/final transcripts. Swap `DummyTranscriber` with your
Conformer/FastConformer (or hybrid CTC/RNNT) engine when ready.
"""

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import nats

try:
    from libs.common.observability import wire_otel  # provides OTEL hooks
except ImportError:  # pragma: no cover - optional dependency
    def wire_otel(*_args, **_kwargs):
        return None

NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
ASR_PARTIAL_INTERVAL_MS = int(os.getenv("ASR_PARTIAL_INTERVAL_MS", "400"))

app = FastAPI(title="asr-streaming", version="0.1.0")
wire_otel(app, "asr-streaming")


class DummyTranscriber:
    """Placeholder transcriber that yields fake results."""

    def __init__(self, sample_rate: int = 16000):  # keeps signature self-documenting
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.closed = asyncio.Event()

    async def accept_audio(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)

    async def finalize(self) -> None:
        self.closed.set()

    async def results(self):
        """Yield partial transcripts at a fixed cadence then emit final."""
        spent = 0
        # emit partials until finalize() is called
        while not self.closed.is_set():
            await asyncio.sleep(ASR_PARTIAL_INTERVAL_MS / 1000.0)
            spent += ASR_PARTIAL_INTERVAL_MS
            yield {
                "type": "partial",
                "start_ms": max(0, spent - ASR_PARTIAL_INTERVAL_MS),
                "end_ms": spent,
                "text": "…",  # placeholder text
            }

        # final result once closed
        yield {
            "type": "final",
            "text": "xin chào thế giới",
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": spent,
                    "text": "xin chào thế giới",
                }
            ],
        }


@app.on_event("startup")
async def on_startup() -> None:
    app.nc = await nats.connect(NATS_URL)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if hasattr(app, "nc"):
        await app.nc.drain()


@app.websocket("/v1/asr/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    audio_id = f"aud_{uuid.uuid4().hex[:8]}"
    transcriber = DummyTranscriber()
    emitter_task = asyncio.create_task(_emit_results(ws, transcriber, audio_id, app.nc))

    try:
        while True:
            message = await ws.receive()
            if "text" in message:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")
                if msg_type == "start":
                    await ws.send_text(json.dumps({"type": "ack", "audio_id": audio_id}))
                elif msg_type == "frame":
                    pcm = base64.b64decode(payload["base64"])
                    await transcriber.accept_audio(pcm)
                elif msg_type == "end":
                    await transcriber.finalize()
                else:  # keep protocol failure obvious
                    await ws.send_text(json.dumps({"type": "error", "message": "unknown message"}))
            elif "bytes" in message:
                await transcriber.accept_audio(message["bytes"])
    except WebSocketDisconnect:
        # client dropped the connection; stop producing results
        await transcriber.finalize()
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
        if not emitter_task.done():
            emitter_task.cancel()


async def _emit_results(ws: WebSocket, transcriber: DummyTranscriber, audio_id: str, nc) -> None:
    async for result in transcriber.results():
        if ws.client_state != WebSocketState.CONNECTED:
            break
        result["audio_id"] = audio_id
        await ws.send_text(json.dumps(result))

        if result["type"] == "final":
            event = {
                "specversion": "1.0",
                "id": str(uuid.uuid4()),
                "source": "services/asr",
                "type": "TranscriptionCompleted",
                "time": datetime.utcnow().isoformat() + "Z",
                "datacontenttype": "application/json",
                "data": {
                    "audio_id": audio_id,
                    "transcript_uri": f"s3://transcripts/{audio_id}.json",
                },
            }
            await nc.publish("transcription.completed", json.dumps(event).encode())
            break


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

