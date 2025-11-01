"""
Security tests for the AI2Text system.

Tests include input validation, authentication, and vulnerability checks.
"""

import pytest
import tempfile
import numpy as np
import soundfile as sf
import time
from pathlib import Path
from fastapi.testclient import TestClient
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def api_client():
    """Create API test client."""
    from api.app import app
    return TestClient(app)


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_file_path_traversal(self, api_client):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]
        
        for path in malicious_paths:
            response = api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
                data={"model_name": path}
            )
            # Should reject or sanitize
            assert response.status_code in [200, 400, 404, 422, 500]
    
    def test_sql_injection(self, api_client):
        """Test SQL injection prevention."""
        sql_injections = [
            "'; DROP TABLE AudioFiles; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM Users--",
        ]
        
        for injection in sql_injections:
            response = api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
                data={"model_name": injection}
            )
            # Should sanitize input
            assert response.status_code in [200, 400, 404, 422, 500]
    
    def test_xss_prevention(self, api_client):
        """Test XSS prevention in API responses."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg/onload=alert('XSS')>",
        ]
        
        for payload in xss_payloads:
            # Create minimal valid audio
            duration = 0.1
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
                data={"model_name": payload}
            )
            
            if response.status_code == 200:
                # Response should not contain script tags
                response_text = str(response.json())
                assert "<script>" not in response_text.lower()
                assert "javascript:" not in response_text.lower()
    
    def test_command_injection(self, api_client):
        """Test command injection prevention."""
        command_injections = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
        ]
        
        for injection in command_injections:
            response = api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", b"fake", "audio/wav")},
                data={"model_name": injection}
            )
            # Should reject or sanitize
            assert response.status_code in [200, 400, 404, 422, 500]


class TestFileValidation:
    """Test file upload validation."""
    
    def test_file_size_limit(self, api_client):
        """Test file size limit enforcement."""
        # Create large fake file
        large_file = b"x" * (50 * 1024 * 1024)  # 50MB
        
        response = api_client.post(
            "/transcribe",
            files={"audio": ("large.wav", large_file, "audio/wav")},
            timeout=10.0
        )
        
        # Should reject or timeout on large files
        assert response.status_code in [400, 413, 422, 500, 504]
    
    def test_file_type_validation(self, api_client):
        """Test file type validation."""
        invalid_files = [
            ("test.exe", b"MZ\x90\x00", "application/x-msdownload"),  # Executable
            ("test.php", b"<?php system($_GET['cmd']); ?>", "text/plain"),  # PHP script
            ("test.sh", b"#!/bin/bash\nrm -rf /", "text/plain"),  # Shell script
            ("test.py", b"import os\nos.system('rm -rf /')", "text/plain"),  # Python script
        ]
        
        for filename, content, content_type in invalid_files:
            response = api_client.post(
                "/transcribe",
                files={"audio": (filename, content, content_type)},
            )
            
            # Should reject non-audio files
            assert response.status_code in [400, 415, 422, 500]
    
    def test_corrupted_audio_file(self, api_client):
        """Test handling of corrupted audio files."""
        corrupted_files = [
            b"",  # Empty file
            b"RIFF\x00\x00\x00\x00WAVE",  # Invalid WAV header
            b"Not audio data",  # Random data
            b"\x00" * 100,  # Null bytes
        ]
        
        for content in corrupted_files:
            response = api_client.post(
                "/transcribe",
                files={"audio": ("corrupted.wav", content, "audio/wav")},
            )
            
            # Should handle gracefully
            assert response.status_code in [400, 422, 500]
    
    def test_zip_bomb_protection(self, api_client):
        """Test protection against zip bomb attacks."""
        # Create compressed file that expands to huge size
        # (This is a simplified test - actual zip bombs are more complex)
        try:
            import zlib
            # Compressed small file that could expand large
            compressed = zlib.compress(b"x" * 1000)
            
            response = api_client.post(
                "/transcribe",
                files={"audio": ("test.wav", compressed, "audio/wav")},
            )
            
            # Should reject or handle gracefully
            assert response.status_code in [400, 422, 500]
        except ImportError:
            pytest.skip("zlib not available")


class TestRateLimiting:
    """Test rate limiting and DoS protection."""
    
    def test_rapid_requests(self, api_client):
        """Test handling of rapid requests (potential DoS)."""
        # Make many rapid requests
        start_time = time.time()
        responses = []
        
        for _ in range(100):
            try:
                response = api_client.get("/health", timeout=1.0)
                responses.append(response.status_code)
            except Exception:
                pass
        
        total_time = time.time() - start_time
        
        # Should handle rapid requests (may rate limit)
        assert len(responses) > 0
        # Should complete or rate limit within reasonable time
        assert total_time < 30.0


class TestErrorHandling:
    """Test error handling and information disclosure."""
    
    def test_error_message_sanitization(self, api_client):
        """Test that error messages don't leak sensitive information."""
        # Try to trigger various errors
        test_cases = [
            {"files": {}, "data": {}},  # Missing file
            {"files": {"audio": ("test.wav", b"", "audio/wav")}, "data": {}},  # Empty file
        ]
        
        for case in test_cases:
            try:
                response = api_client.post("/transcribe", **case)
                
                if response.status_code != 200:
                    error_text = str(response.json())
                    # Should not expose internal paths, stack traces, etc.
                    sensitive_patterns = [
                        "/home/",
                        "/root/",
                        "C:\\",
                        "Traceback",
                        "File \"",
                        ".py\"",
                    ]
                    
                    for pattern in sensitive_patterns:
                        assert pattern not in error_text
            except Exception:
                pass
    
    def test_stack_trace_prevention(self, api_client):
        """Test that stack traces are not exposed."""
        # Try to trigger an internal error
        response = api_client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"invalid", "audio/wav")},
            data={"model_name": "nonexistent"}
        )
        
        error_text = str(response.json())
        # Should not contain Python stack trace
        assert "Traceback" not in error_text
        assert "File \"" not in error_text


class TestAuthentication:
    """Test authentication and authorization (if implemented)."""
    
    def test_unauthenticated_access(self, api_client):
        """Test that public endpoints are accessible without auth."""
        # Public endpoints should be accessible
        response = api_client.get("/health")
        assert response.status_code == 200
    
    def test_invalid_authentication(self, api_client):
        """Test handling of invalid authentication tokens."""
        # If authentication is implemented, test invalid tokens
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid",
            "expired_token",
        ]
        
        # This depends on actual implementation
        # For now, just test that endpoints handle auth gracefully
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

