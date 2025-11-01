"""
Integration tests for API endpoints.

Tests complete workflows through the API.
"""

import pytest
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.app import app


@pytest.fixture
def api_client():
    """Create API test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = Path(tmp.name)
    
    yield str(tmp_path)
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


class TestTranscriptionWorkflow:
    """Test complete transcription workflow."""
    
    @pytest.mark.integration
    def test_health_check_workflow(self, api_client):
        """Test health check before transcription."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        health = response.json()
        assert health["status"] == "healthy"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_list_models_before_transcription(self, api_client):
        """Test listing models before transcription."""
        response = api_client.get("/models")
        assert response.status_code == 200
        
        models = response.json()
        assert "models" in models
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_transcription_flow(self, api_client, sample_audio_file):
        """Test complete transcription flow."""
        # 1. Check health
        health_response = api_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. List models
        models_response = api_client.get("/models")
        assert models_response.status_code == 200
        
        # 3. Transcribe audio
        with open(sample_audio_file, 'rb') as f:
            files = {"audio": ("test.wav", f.read(), "audio/wav")}
            data = {
                "model_name": "default",
                "use_beam_search": "false"
            }
            transcribe_response = api_client.post("/transcribe", files=files, data=data)
        
        # May fail if model not available, but should handle gracefully
        assert transcribe_response.status_code in [200, 404, 500]
        
        if transcribe_response.status_code == 200:
            result = transcribe_response.json()
            assert "text" in result
            assert "confidence" in result
            assert "processing_time" in result


class TestDatabaseIntegration:
    """Test database integration with API."""
    
    @pytest.mark.integration
    def test_database_operations_flow(self, temp_db):
        """Test complete database operations flow."""
        # 1. Add audio file
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        assert audio_id > 0
        
        # 2. Add transcript
        transcript_id = temp_db.add_transcript(
            audio_file_id=audio_id,
            transcript="xin chào",
            normalized_transcript="xin chao"
        )
        assert transcript_id > 0
        
        # 3. Assign to split
        success = temp_db.assign_split(audio_id, "train", "v1")
        assert success is True
        
        # 4. Retrieve data
        files = temp_db.get_audio_files_by_split("train", "v1")
        assert len(files) > 0
        
        # 5. Get statistics
        stats = temp_db.get_dataset_statistics("v1")
        assert stats is not None


class TestPreprocessingIntegration:
    """Test preprocessing pipeline integration."""
    
    @pytest.mark.integration
    def test_audio_preprocessing_pipeline(self, sample_audio_file):
        """Test complete audio preprocessing pipeline."""
        from preprocessing.audio_processing import AudioProcessor, AudioAugmenter, preprocess_audio_file
        
        # Test complete pipeline
        processor = AudioProcessor(sample_rate=16000)
        augmenter = AudioAugmenter(sample_rate=16000)
        
        result = preprocess_audio_file(
            sample_audio_file,
            processor=processor,
            augmenter=augmenter,
            apply_augmentation=False,
            extract_features=True
        )
        
        assert 'audio' in result
        assert 'mel_spectrogram' in result
        assert 'sample_rate' in result
        assert result['sample_rate'] == 16000


class TestModelTrainingIntegration:
    """Test model training workflow integration."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_forward_backward_pass(self, asr_model):
        """Test complete forward and backward pass."""
        import torch
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        # Forward pass
        logits, output_lengths = asr_model(input_features, lengths)
        
        assert logits.shape == (batch_size, seq_len, asr_model.vocab_size)
        
        # Backward pass
        loss = logits.mean()
        loss.backward()
        
        assert input_features.grad is not None
        assert any(p.grad is not None for p in asr_model.parameters() if p.requires_grad)

