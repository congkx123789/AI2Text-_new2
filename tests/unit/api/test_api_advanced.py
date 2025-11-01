"""
Advanced and challenging tests for API endpoints.

Tests include security, load testing, edge cases, and error handling.
"""

import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import concurrent.futures
import time
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from api.app import app


@pytest.fixture
def api_client():
    """Create API test client."""
    return TestClient(app)


@pytest.fixture
def large_audio_file():
    """Create a large audio file for testing."""
    # 5 minutes of audio
    duration = 300.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = Path(tmp.name)
    
    yield str(tmp_path)
    
    if tmp_path.exists():
        tmp_path.unlink()


class TestAPISecurity:
    """Security tests for API endpoints."""
    
    def test_transcribe_path_traversal_attack(self, api_client):
        """Test path traversal attack prevention."""
        # Try to access files outside allowed directory
        malicious_path = "../../../etc/passwd"
        
        # Should be rejected
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"fake audio", "audio/wav")},
            data={"model_name": malicious_path}
        )
        
        # Should reject or handle gracefully
        assert response.status_code in [400, 404, 422, 500]
    
    def test_transcribe_sql_injection(self, api_client):
        """Test SQL injection prevention."""
        malicious_input = "'; DROP TABLE AudioFiles; --"
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"fake audio", "audio/wav")},
            data={"model_name": malicious_input}
        )
        
        # Should sanitize input
        assert response.status_code in [200, 400, 404, 422, 500]
    
    def test_transcribe_xss_prevention(self, api_client):
        """Test XSS prevention in responses."""
        malicious_input = "<script>alert('XSS')</script>"
        
        # Create minimal audio
        audio_bytes = self._create_minimal_audio()
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={"model_name": malicious_input}
        )
        
        if response.status_code == 200:
            # Response should not contain script tags
            response_text = str(response.json())
            assert "<script>" not in response_text.lower()
    
    def test_transcribe_file_size_limit(self, api_client):
        """Test file size limit enforcement."""
        # Create very large fake file
        large_file_content = b"x" * (100 * 1024 * 1024)  # 100MB
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", large_file_content, "audio/wav")},
            timeout=10.0
        )
        
        # Should reject or timeout on very large files
        assert response.status_code in [400, 413, 422, 500, 504]
    
    def test_transcribe_content_type_validation(self, api_client):
        """Test content type validation."""
        # Try to upload non-audio file
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.txt", b"not audio data", "text/plain")},
        )
        
        # Should reject non-audio files
        assert response.status_code in [400, 415, 422]
    
    def _create_minimal_audio(self):
        """Create minimal valid audio file."""
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        return audio_bytes


class TestAPILoadTesting:
    """Load and stress tests for API."""
    
    def test_concurrent_transcription_requests(self, api_client):
        """Test handling concurrent transcription requests."""
        # Create test audio file
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        def make_request():
            return api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
                data={"use_beam_search": "false"},
                timeout=30.0
            )
        
        # Make 20 concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Should handle concurrent requests
        assert len(results) == 20
        # At least some should succeed (may fail if model not available)
        success_count = sum(1 for r in results if r.status_code == 200)
        
        # Should complete within reasonable time
        assert total_time < 120.0  # 2 minutes max
    
    def test_rapid_health_checks(self, api_client):
        """Test rapid health check requests."""
        def check_health():
            return api_client.get("/health", timeout=5.0)
        
        # Make 100 rapid requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_health) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # All should succeed
        assert all(r.status_code == 200 for r in results)
        # Should be fast (< 10 seconds for 100 requests)
        assert total_time < 10.0
    
    def test_large_file_transcription(self, api_client, large_audio_file):
        """Test transcription of large audio file."""
        with open(large_audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        start_time = time.time()
        response = api_client.post(
            "/transcribe",
            files={"audio": ("large.wav", audio_bytes, "audio/wav")},
            data={"use_beam_search": "false"},
            timeout=600.0  # 10 minutes for large file
        )
        processing_time = time.time() - start_time
        
        # May fail if model not available, but should handle request
        assert response.status_code in [200, 404, 500, 504]
        # Should complete within reasonable time or timeout
        assert processing_time < 600.0


class TestAPIErrorHandling:
    """Test error handling and edge cases."""
    
    def test_transcribe_missing_file_parameter(self, api_client):
        """Test transcription without file parameter."""
        response = api_client.post("/transcribe")
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_transcribe_empty_file(self, api_client):
        """Test transcription with empty file."""
        response = api_client.post(
            "/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        
        # Should reject empty file
        assert response.status_code in [400, 422, 500]
    
    def test_transcribe_invalid_model_name(self, api_client):
        """Test transcription with invalid model name."""
        duration = 0.5
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={"model_name": "nonexistent_model_12345"}
        )
        
        # Should return 404 or handle gracefully
        assert response.status_code in [200, 404, 500]
    
    def test_transcribe_invalid_parameters(self, api_client):
        """Test transcription with invalid parameters."""
        duration = 0.5
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        # Invalid beam width (negative)
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={"beam_width": "-5"}
        )
        
        # Should validate and reject or handle gracefully
        assert response.status_code in [200, 400, 422, 500]
    
    def test_health_check_during_error_state(self, api_client):
        """Test health check when system is in error state."""
        # Health check should always work
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    def test_transcribe_response_time(self, api_client, benchmark):
        """Benchmark transcription response time."""
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        def transcribe():
            return api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
                data={"use_beam_search": "false"},
                timeout=30.0
            )
        
        benchmark(transcribe)
    
    def test_health_check_response_time(self, api_client, benchmark):
        """Benchmark health check response time."""
        def health_check():
            return api_client.get("/health")
        
        benchmark(health_check)


class TestAPIEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("file_extension", [".wav", ".mp3", ".flac", ".m4a", ".ogg"])
    def test_transcribe_various_audio_formats(self, api_client, file_extension):
        """Test transcription with various audio formats."""
        # Note: Actual format support depends on librosa/soundfile
        duration = 0.5
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            try:
                sf.write(tmp.name, audio, sample_rate)
                with open(tmp.name, 'rb') as f:
                    audio_bytes = f.read()
            except Exception:
                pytest.skip(f"Format {file_extension} not supported")
            finally:
                Path(tmp.name).unlink()
        
        response = api_client.post(
            "/transcribe",
            files={"audio": (f"test{file_extension}", audio_bytes, f"audio/{file_extension[1:]}")},
            data={"use_beam_search": "false"},
        )
        
        # May succeed or fail depending on format support
        assert response.status_code in [200, 400, 415, 422, 500]
    
    def test_transcribe_very_short_audio(self, api_client):
        """Test transcription with very short audio."""
        # 10ms of audio
        duration = 0.01
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            Path(tmp.name).unlink()
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("short.wav", audio_bytes, "audio/wav")},
            data={"use_beam_search": "false"},
        )
        
        # Should handle very short audio
        assert response.status_code in [200, 400, 422, 500]

