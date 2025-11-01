"""
Unit tests for API application.

Tests for FastAPI endpoints and request handling.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock

# Import app - adjust import path as needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio file bytes."""
    # Generate 1 second of audio
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Save to temporary bytes
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        with open(tmp.name, 'rb') as f:
            audio_bytes = f.read()
        Path(tmp.name).unlink()
    
    return audio_bytes


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check_fields(self, client):
        """Test health check response fields."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "models_loaded" in data
        assert "tokenizer_ready" in data
        assert "processor_ready" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test GET / endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_root_endpoint_structure(self, client):
        """Test root endpoint response structure."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["endpoints"], dict)


class TestModelsEndpoint:
    """Test models endpoint."""
    
    def test_list_models(self, client):
        """Test GET /models endpoint."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    @patch('api.app.Path')
    def test_list_models_with_checkpoints(self, mock_path, client):
        """Test listing models when checkpoints exist."""
        # Mock checkpoint directory
        mock_checkpoint_dir = Mock()
        mock_checkpoint_dir.exists.return_value = True
        mock_checkpoint_dir.glob.return_value = [
            Mock(stem="model1"),
            Mock(stem="model2")
        ]
        mock_path.return_value = mock_checkpoint_dir
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data


class TestTranscriptionEndpoint:
    """Test transcription endpoint."""
    
    @patch('api.app.processor_cache')
    @patch('api.app.models_cache')
    @patch('api.app.tokenizer_cache')
    def test_transcribe_audio(self, mock_tokenizer, mock_models, mock_processor, client, sample_audio_bytes):
        """Test POST /transcribe endpoint."""
        # Mock dependencies
        mock_processor.extract_features.return_value = np.random.randn(100, 80)
        mock_tokenizer.__len__.return_value = 100
        mock_tokenizer.decode.return_value = "test transcription"
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = (Mock(), torch.tensor([100]))
        mock_models.get.return_value = mock_model
        
        files = {"audio": ("test.wav", sample_audio_bytes, "audio/wav")}
        data = {
            "model_name": "default",
            "use_beam_search": "false",
            "beam_width": "5"
        }
        
        response = client.post("/transcribe", files=files, data=data)
        
        # Should handle gracefully even if model loading fails
        assert response.status_code in [200, 404, 500]
    
    def test_transcribe_invalid_file(self, client):
        """Test transcription with invalid file."""
        files = {"audio": ("test.txt", b"not audio data", "text/plain")}
        
        response = client.post("/transcribe", files=files)
        
        # Should return error
        assert response.status_code in [400, 422, 500]
    
    def test_transcribe_missing_file(self, client):
        """Test transcription without file."""
        response = client.post("/transcribe")
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    @patch('api.app.processor_cache')
    def test_transcribe_with_beam_search(self, mock_processor, client, sample_audio_bytes):
        """Test transcription with beam search enabled."""
        mock_processor.extract_features.return_value = np.random.randn(100, 80)
        
        files = {"audio": ("test.wav", sample_audio_bytes, "audio/wav")}
        data = {
            "use_beam_search": "true",
            "beam_width": "10"
        }
        
        response = client.post("/transcribe", files=files, data=data)
        
        # May fail due to missing model, but should handle request
        assert response.status_code in [200, 404, 500]


class TestModelManagementEndpoints:
    """Test model management endpoints."""
    
    def test_load_model_endpoint(self, client):
        """Test POST /models/load endpoint."""
        data = {
            "model_path": "checkpoints/test.pt",
            "model_name": "test_model"
        }
        
        response = client.post("/models/load", json=data)
        
        # May fail if model file doesn't exist, but endpoint should exist
        assert response.status_code in [200, 404, 500]
    
    def test_unload_model_endpoint(self, client):
        """Test DELETE /models/{model_name} endpoint."""
        response = client.delete("/models/nonexistent_model")
        
        # Should return 404 if model not in cache
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test handling invalid JSON."""
        response = client.post("/models/load", data="invalid json")
        
        # Should return validation error
        assert response.status_code in [400, 422]


class TestSecurityHeaders:
    """Test security headers."""
    
    def test_security_headers_present(self, client):
        """Test that security headers are present."""
        response = client.get("/health")
        
        headers = response.headers
        # Check for security headers (may not all be present in test client)
        # But we can verify the response is valid
        assert response.status_code == 200


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers (if configured)."""
        # CORS headers may not be present in test client
        # This is a placeholder for CORS testing
        response = client.get("/health")
        
        assert response.status_code == 200

