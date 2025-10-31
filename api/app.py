"""
REST API for ASR system.

FastAPI application providing endpoints for:
- Transcribing audio files
- Training models
- Evaluating models
- Managing models and experiments
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from pathlib import Path
import tempfile
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from models.lstm_asr import LSTMASRModel
from models.enhanced_asr import EnhancedASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from database.db_utils import ASRDatabase
from decoding.beam_search import BeamSearchDecoder
from decoding.lm_decoder import LMBeamSearchDecoder
from decoding.confidence import ConfidenceScorer
from utils.metrics import calculate_wer, calculate_cer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese ASR API",
    description="REST API for Vietnamese Speech-to-Text system",
    version="1.0.0"
)

# Global state
models_cache = {}
tokenizer_cache = None
processor_cache = None


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    model_name: Optional[str] = "default"
    use_beam_search: bool = True
    beam_width: int = 5
    use_lm: bool = False
    lm_path: Optional[str] = None
    min_confidence: Optional[float] = 0.5


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    confidence: float
    model_name: str
    processing_time: float


class TrainingRequest(BaseModel):
    """Request model for training."""
    config_path: str
    model_type: str = "transformer"  # transformer, lstm, enhanced
    num_epochs: int = 10
    batch_size: int = 16


@app.on_event("startup")
async def startup_event():
    """Initialize models and components on startup."""
    global tokenizer_cache, processor_cache
    
    logger.info("Initializing ASR API...")
    
    # Initialize tokenizer
    tokenizer_cache = Tokenizer()
    logger.info(f"Tokenizer initialized with vocab size: {len(tokenizer_cache)}")
    
    # Initialize audio processor
    processor_cache = AudioProcessor(sample_rate=16000, n_mels=80)
    logger.info("Audio processor initialized")
    
    logger.info("API ready!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vietnamese ASR API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models_cache),
        "tokenizer_ready": tokenizer_cache is not None,
        "processor_ready": processor_cache is not None
    }


def load_model(model_path: str, model_type: str = "transformer"):
    """Load ASR model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    input_dim = config.get('n_mels', 80)
    vocab_size = len(tokenizer_cache)
    d_model = config.get('d_model', 256)
    
    if model_type == "lstm":
        model = LSTMASRModel(input_dim=input_dim, vocab_size=vocab_size, hidden_size=d_model)
    elif model_type == "enhanced":
        model = EnhancedASRModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            use_contextual_embeddings=config.get('use_contextual_embeddings', True)
        )
    else:  # transformer
        model = ASRModel(input_dim=input_dim, vocab_size=vocab_size, d_model=d_model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    model_name: str = "default",
    use_beam_search: bool = True,
    beam_width: int = 5,
    use_lm: bool = False,
    lm_path: Optional[str] = None,
    min_confidence: Optional[float] = None
):
    """
    Transcribe audio file to text.
    
    Args:
        audio: Audio file (WAV, MP3, FLAC)
        model_name: Model name to use
        use_beam_search: Use beam search decoding
        beam_width: Beam search width
        use_lm: Use language model (KenLM)
        lm_path: Path to language model file
        min_confidence: Minimum confidence threshold
        
    Returns:
        Transcription result with text and confidence
    """
    import time
    import soundfile as sf
    import librosa
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await audio.read()
            tmp_path.write_bytes(content)
        
        # Load audio
        try:
            audio_data, sample_rate = sf.read(str(tmp_path))
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Mono
        except:
            # Try librosa if soundfile fails
            audio_data, sample_rate = librosa.load(str(tmp_path), sr=16000)
        
        # Preprocess audio
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Extract features
        features = processor_cache.extract_features(audio_data)
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()
        lengths = torch.tensor([features.shape[0]])
        
        # Load model (lazy loading)
        if model_name not in models_cache:
            # Try to find checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoint_path = checkpoint_dir / f"{model_name}.pt"
            if not checkpoint_path.exists():
                checkpoint_path = checkpoint_dir / "best_model.pt"
            
            if checkpoint_path.exists():
                model = load_model(str(checkpoint_path), model_type="transformer")
                models_cache[model_name] = model
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        model = models_cache[model_name]
        
        # Run inference
        with torch.no_grad():
            logits, output_lengths = model(features_tensor, lengths)
        
        # Decode
        if use_lm and lm_path and Path(lm_path).exists():
            # LM decoding
            vocab = [tokenizer_cache.id_to_token.get(i, "") for i in range(len(tokenizer_cache))]
            lm_decoder = LMBeamSearchDecoder(vocab=vocab, lm_path=lm_path)
            results = lm_decoder.decode(logits, output_lengths)
            if results:
                text = results[0]["text"]
                confidence = results[0].get("score", 0.0)
            else:
                text = ""
                confidence = 0.0
        elif use_beam_search:
            # Beam search decoding
            decoder = BeamSearchDecoder(
                vocab_size=len(tokenizer_cache),
                blank_token_id=tokenizer_cache.blank_token_id,
                beam_width=beam_width
            )
            results = decoder.decode_batch(logits, output_lengths, tokenizer_cache)
            if results:
                text = results[0].get("text_decoded", "")
                confidence = results[0].get("confidence", 0.0)
            else:
                text = ""
                confidence = 0.0
        else:
            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)
            pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
            text = tokenizer_cache.decode(pred_tokens)
            
            # Compute confidence
            scorer = ConfidenceScorer()
            confidence = scorer.compute(logits, None, output_lengths)[0].item()
        
        # Filter by confidence if requested
        if min_confidence is not None and confidence < min_confidence:
            text = ""  # Reject low-confidence predictions
        
        processing_time = time.time() - start_time
        
        # Cleanup
        tmp_path.unlink()
        
        return TranscriptionResponse(
            text=text,
            confidence=float(confidence),
            model_name=model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    checkpoint_dir = Path("checkpoints")
    models = []
    
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.glob("*.pt"):
            models.append({
                "name": checkpoint_file.stem,
                "path": str(checkpoint_file),
                "size_mb": checkpoint_file.stat().st_size / (1024 * 1024)
            })
    
    return {
        "models": models,
        "loaded": list(models_cache.keys())
    }


@app.post("/models/load")
async def load_model_endpoint(model_path: str, model_name: str = "default"):
    """Load a model into cache."""
    try:
        model = load_model(model_path, model_type="transformer")
        models_cache[model_name] = model
        return {"status": "loaded", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from cache."""
    if model_name in models_cache:
        del models_cache[model_name]
        return {"status": "unloaded", "model_name": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not in cache")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

