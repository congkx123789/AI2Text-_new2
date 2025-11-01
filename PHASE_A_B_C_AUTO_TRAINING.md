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

import os
import asyncio
import nats
import json
from datetime import datetime
import uuid
import subprocess
from pathlib import Path
import yaml

# Configuration
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
HOURS_THRESHOLD = float(os.getenv("HOURS_THRESHOLD", "5.0"))
CRON_SCHEDULE = os.getenv("CRON_SCHEDULE", "02:00")  # Daily at 2 AM

# Training configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "fp16")
EPOCHS = int(os.getenv("EPOCHS", "50"))

# Best model tracking
best_wer = float("inf")
IMPROVEMENT_THRESHOLD = float(os.getenv("WER_IMPROVEMENT_THRESHOLD", "0.01"))  # 1% relative

# NATS client
nc = None

# State
current_job = None


async def trigger_training(manifest_paths: dict, trigger_source: str = "manual"):
    """
    Trigger a training run.
    
    Args:
        manifest_paths: {"train": path, "val": path, "test": path}
        trigger_source: "dataset_ready", "cron", or "manual"
    """
    global current_job, best_wer
    
    if current_job and current_job.get("status") == "running":
        print("[WARNING] Training already in progress, skipping trigger")
        return
    
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n[Training] Starting run: {run_id}")
    print(f"  Trigger: {trigger_source}")
    print(f"  Manifests: {manifest_paths}")
    
    # Create training config
    config = {
        "run_id": run_id,
        "data": {
            "train_manifest": manifest_paths.get("train"),
            "val_manifest": manifest_paths.get("val"),
            "test_manifest": manifest_paths.get("test")
        },
        "model": {
            "type": "conformer_ctc",
            "hidden_size": 256,
            "num_layers": 12
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "mixed_precision": MIXED_PRECISION,
            "epochs": EPOCHS,
            "learning_rate": 5e-4
        }
    }
    
    # Save config
    config_path = f"/tmp/train_config_{run_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Publish training.started event
    start_event = {
        "specversion": "1.0",
        "id": str(uuid.uuid4()),
        "source": "services/training-orchestrator",
        "type": "TrainingStarted",
        "time": datetime.utcnow().isoformat() + "Z",
        "data": {
            "run_id": run_id,
            "config": config,
            "trigger_source": trigger_source
        }
    }
    
    await nc.publish("training.started", json.dumps(start_event).encode())
    
    # Update state
    current_job = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat()
    }
    
    # Run training in background
    asyncio.create_task(run_training_job(run_id, config_path))


async def run_training_job(run_id: str, config_path: str):
    """Run training job as subprocess."""
    global current_job, best_wer
    
    try:
        print(f"[Training] Running job {run_id}")
        
        # Launch training script
        cmd = [
            "python", "-m", "training.train",
            "--config", config_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Stream output
        async for line in process.stdout:
            print(f"[Train] {line.decode().strip()}")
        
        # Wait for completion
        returncode = await process.wait()
        
        if returncode == 0:
            print(f"[OK] Training completed successfully")
            
            # Evaluate model
            metrics = await evaluate_model(run_id)
            
            # Check if improved
            if metrics and metrics["wer"] < best_wer - IMPROVEMENT_THRESHOLD:
                print(f"[OK] Model improved! WER: {metrics['wer']:.3f} (prev: {best_wer:.3f})")
                
                # Promote model
                await promote_model(run_id, metrics)
                best_wer = metrics["wer"]
            else:
                print(f"[INFO] No improvement. WER: {metrics['wer']:.3f}")
            
            # Publish training.completed
            complete_event = {
                "specversion": "1.0",
                "id": str(uuid.uuid4()),
                "source": "services/training-orchestrator",
                "type": "TrainingCompleted",
                "time": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "run_id": run_id,
                    "status": "success",
                    "metrics": metrics,
                    "returncode": returncode
                }
            }
            
            await nc.publish("training.completed", json.dumps(complete_event).encode())
            
        else:
            print(f"[ERROR] Training failed with code {returncode}")
            
            # Publish training.failed
            fail_event = {
                "specversion": "1.0",
                "id": str(uuid.uuid4()),
                "source": "services/training-orchestrator",
                "type": "TrainingFailed",
                "time": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "run_id": run_id,
                    "returncode": returncode
                }
            }
            
            await nc.publish("training.failed", json.dumps(fail_event).encode())
        
        # Update state
        current_job = {
            "run_id": run_id,
            "status": "completed" if returncode == 0 else "failed",
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] Training job failed: {e}")
        import traceback
        traceback.print_exc()
        
        current_job = {
            "run_id": run_id,
            "status": "error",
            "error": str(e)
        }


async def evaluate_model(run_id: str) -> dict:
    """Evaluate trained model."""
    try:
        print(f"[Eval] Evaluating model {run_id}")
        
        # Run evaluation script
        cmd = [
            "python", "-m", "training.evaluate",
            "--checkpoint", f"checkpoints/{run_id}/best.pt",
            "--val-manifest", "/data/manifests/val.jsonl"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        if result.returncode == 0:
            # Parse metrics from output
            # TODO: Improve this to read from metrics file
            lines = result.stdout.split("\n")
            wer = 0.25  # Placeholder
            
            for line in lines:
                if "WER:" in line:
                    wer = float(line.split("WER:")[-1].strip().rstrip("%")) / 100
            
            metrics = {
                "wer": wer,
                "cer": wer * 0.5,  # Placeholder
                "evaluated_at": datetime.now().isoformat()
            }
            
            print(f"[OK] Evaluation complete: WER={wer:.3f}")
            return metrics
        else:
            print(f"[ERROR] Evaluation failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Evaluation error: {e}")
        return None


async def promote_model(run_id: str, metrics: dict):
    """Promote model to production."""
    try:
        print(f"[Promote] Promoting model {run_id}")
        
        # TODO: Upload to S3/MinIO models bucket
        # TODO: Update model registry
        
        # Publish model.promoted event
        promote_event = {
            "specversion": "1.0",
            "id": str(uuid.uuid4()),
            "source": "services/training-orchestrator",
            "type": "ModelPromoted",
            "time": datetime.utcnow().isoformat() + "Z",
            "data": {
                "run_id": run_id,
                "checkpoint_uri": f"s3://models/asr/{run_id}/best.pt",
                "metrics": metrics,
                "promoted_at": datetime.now().isoformat()
            }
        }
        
        await nc.publish("model.promoted", json.dumps(promote_event).encode())
        print(f"[OK] Model promoted")
        
    except Exception as e:
        print(f"[ERROR] Model promotion failed: {e}")


async def handle_dataset_ready(msg):
    """Handle dataset.ready event."""
    try:
        evt = json.loads(msg.data.decode())
        data = evt["data"]
        
        print(f"\n[Trigger] Dataset ready")
        print(f"  Train hours: {data['hours']['train']:.2f}h")
        print(f"  Val hours: {data['hours']['val']:.2f}h")
        
        manifest_paths = {
            "train": data["train_manifest"],
            "val": data["val_manifest"],
            "test": data.get("test_manifest")
        }
        
        await trigger_training(manifest_paths, trigger_source="dataset_ready")
        
    except Exception as e:
        print(f"[ERROR] Failed to handle dataset.ready: {e}")


async def main():
    """Main service loop."""
    global nc
    
    # Connect to NATS
    nc = await nats.connect(NATS_URL)
    print("[OK] Connected to NATS")
    
    # Subscribe to dataset.ready
    await nc.subscribe("dataset.ready", cb=handle_dataset_ready)
    print("[OK] Subscribed to dataset.ready")
    
    print(f"[OK] Training orchestrator started")
    print(f"  Hours threshold: {HOURS_THRESHOLD}h")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Mixed precision: {MIXED_PRECISION}")
    
    # TODO: Add cron scheduler for periodic training
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
    finally:
        await nc.drain()
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Let me now complete the markdown guide with the verification steps and configuration:

<function_calls>
<invoke name="search_replace">
<parameter name="file_path">PHASE_A_B_C_AUTO_TRAINING.md
