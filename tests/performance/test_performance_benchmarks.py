"""
Performance benchmark tests.

Tests for system performance under various conditions.
"""

import pytest
import torch
import numpy as np
import time
from pathlib import Path
import tempfile
import soundfile as sf

# Mark all tests in this file as performance tests
pytestmark = pytest.mark.performance


class TestPreprocessingPerformance:
    """Performance tests for preprocessing."""
    
    def test_audio_loading_performance(self, benchmark):
        """Benchmark audio loading speed."""
        from preprocessing.audio_processing import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000)
        duration = 10.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name
        
        def load():
            return processor.load_audio(tmp_path)
        
        benchmark(load)
        
        Path(tmp_path).unlink()
    
    def test_mel_spectrogram_extraction_performance(self, benchmark):
        """Benchmark mel spectrogram extraction."""
        from preprocessing.audio_processing import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000)
        audio = np.random.randn(160000).astype(np.float32)  # 10 seconds
        
        benchmark(processor.extract_mel_spectrogram, audio)
    
    def test_batch_audio_processing(self, benchmark):
        """Benchmark batch audio processing."""
        from preprocessing.audio_processing import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000)
        batch_size = 32
        audio_batch = [np.random.randn(16000).astype(np.float32) for _ in range(batch_size)]
        
        def process_batch():
            return [processor.extract_mel_spectrogram(audio) for audio in audio_batch]
        
        benchmark(process_batch)


class TestModelPerformance:
    """Performance tests for models."""
    
    def test_model_inference_speed(self, benchmark):
        """Benchmark model inference speed."""
        from models.asr_base import ASRModel
        
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        model.eval()
        
        batch_size = 4
        seq_len = 200
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len] * batch_size)
        
        def infer():
            with torch.no_grad():
                return model(input_features, lengths)
        
        benchmark(infer)
    
    def test_model_training_step_speed(self, benchmark):
        """Benchmark model training step speed."""
        from models.asr_base import ASRModel
        
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        model.train()
        
        batch_size = 4
        seq_len = 200
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len] * batch_size)
        
        def training_step():
            logits, _ = model(input_features, lengths)
            loss = logits.mean()
            loss.backward()
            model.zero_grad()
            return loss.item()
        
        benchmark(training_step)


class TestDecodingPerformance:
    """Performance tests for decoding."""
    
    def test_beam_search_performance(self, benchmark):
        """Benchmark beam search decoding."""
        from decoding.beam_search import BeamSearchDecoder
        
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=10
        )
        
        batch_size = 1
        seq_len = 300
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        def decode():
            return decoder.decode(logits, lengths)
        
        benchmark(decode)
    
    def test_lm_decoder_performance(self, benchmark):
        """Benchmark LM decoder performance."""
        from decoding.lm_decoder import LMBeamSearchDecoder
        
        vocab = ["<blank>", "<pad>"] + [f"token_{i}" for i in range(100)]
        decoder = LMBeamSearchDecoder(vocab=vocab, blank_token_id=0, beam_width=10)
        
        batch_size = 1
        seq_len = 300
        logits = torch.randn(batch_size, seq_len, len(vocab))
        lengths = torch.tensor([seq_len])
        
        def decode():
            return decoder.decode(logits, lengths)
        
        benchmark(decode)


class TestEndToEndPerformance:
    """End-to-end performance tests."""
    
    def test_full_transcription_pipeline(self, benchmark):
        """Benchmark full transcription pipeline."""
        from preprocessing.audio_processing import AudioProcessor
        from models.asr_base import ASRModel
        from decoding.beam_search import BeamSearchDecoder
        
        # Setup
        processor = AudioProcessor(sample_rate=16000)
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        model.eval()
        decoder = BeamSearchDecoder(vocab_size=100, blank_token_id=0, beam_width=5)
        
        # Create audio
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        
        def transcribe():
            # Preprocess
            mel_spec = processor.extract_mel_spectrogram(audio)
            features = torch.from_numpy(mel_spec).unsqueeze(0).float()
            lengths = torch.tensor([mel_spec.shape[1]])
            
            # Model inference
            with torch.no_grad():
                logits, output_lengths = model(features, lengths)
            
            # Decode
            results = decoder.decode(logits, output_lengths)
            return results
        
        benchmark(transcribe)

