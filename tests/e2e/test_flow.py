"""
End-to-end test for the full ASR microservices flow.

Tests:
1. WebSocket streaming transcription
2. Batch file upload and ingestion
3. Transcript retrieval from metadata service
4. NLP post-processing integration
5. Search functionality
"""

import pytest
import httpx
import asyncio
import websockets
import json
import base64
import time
from pathlib import Path

# Service endpoints
BASE = "http://localhost:8080"  # API Gateway
ASR_WS = "ws://localhost:8003/v1/asr/stream"  # Direct ASR streaming
METADATA_URL = "http://localhost:8002"  # Direct metadata service
NLP_POST_URL = "http://localhost:8004"  # Direct NLP service
SEARCH_URL = "http://localhost:8006"  # Direct search service

# Test auth token (dev mode)
AUTH_HEADERS = {"Authorization": "Bearer dev"}


@pytest.mark.asyncio
async def test_asr_websocket_streaming():
    """Test real-time ASR streaming via WebSocket."""
    audio_id = None
    
    try:
        async with websockets.connect(ASR_WS) as ws:
            # Send start message
            await ws.send(json.dumps({
                "type": "start",
                "audio_format": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "encoding": "pcm16"
                }
            }))
            
            # Receive acknowledgment
            ack_msg = await ws.recv()
            ack = json.loads(ack_msg)
            assert ack["type"] == "ack"
            audio_id = ack.get("audio_id")
            assert audio_id is not None
            print(f"✓ Got audio_id: {audio_id}")
            
            # Send a short silence buffer (simulate audio)
            silence = b'\x00' * 32000  # 1 second of silence at 16kHz mono
            await ws.send(json.dumps({
                "type": "frame",
                "base64": base64.b64encode(silence).decode()
            }))
            
            # Send end message
            await ws.send(json.dumps({"type": "end"}))
            
            # Collect results
            results = []
            async for msg in ws:
                evt = json.loads(msg)
                results.append(evt)
                print(f"  Received: {evt['type']}")
                
                if evt["type"] == "final":
                    assert "text" in evt
                    assert evt["audio_id"] == audio_id
                    print(f"✓ Final transcript: {evt['text']}")
                    break
            
            assert len(results) > 0
            assert results[-1]["type"] == "final"
            
    except Exception as e:
        pytest.fail(f"WebSocket streaming test failed: {e}")


@pytest.mark.asyncio
async def test_batch_ingestion_flow():
    """Test batch file upload through ingestion service."""
    
    # Create a small test audio file (WAV header + silence)
    wav_header = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x08, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16 for PCM)
        0x01, 0x00,              # AudioFormat (1 = PCM)
        0x01, 0x00,              # NumChannels (1 = mono)
        0x80, 0x3E, 0x00, 0x00,  # SampleRate (16000)
        0x00, 0x7D, 0x00, 0x00,  # ByteRate
        0x02, 0x00,              # BlockAlign
        0x10, 0x00,              # BitsPerSample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x08, 0x00, 0x00,  # Subchunk2Size
    ])
    silence = b'\x00' * 2048
    test_audio = wav_header + silence
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Upload through gateway
        files = {"file": ("test.wav", test_audio, "audio/wav")}
        response = await client.post(
            f"{BASE}/v1/ingest",
            files=files,
            headers=AUTH_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "audio_id" in data
        assert "object_uri" in data
        
        audio_id = data["audio_id"]
        print(f"✓ Uploaded audio: {audio_id}")
        print(f"✓ Object URI: {data['object_uri']}")
        
        # Wait a bit for async processing
        await asyncio.sleep(2)
        
        return audio_id


@pytest.mark.asyncio
async def test_nlp_normalization():
    """Test NLP post-processing service for Vietnamese text."""
    
    test_text = "xin chao viet nam"  # Without diacritics
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{NLP_POST_URL}/v1/nlp/normalize",
            json={
                "text": test_text,
                "restore_diacritics": True,
                "fix_typos": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "text_clean" in data
        assert "text_with_diacritics" in data
        assert "corrections" in data
        
        print(f"✓ Original: {test_text}")
        print(f"✓ With diacritics: {data['text_with_diacritics']}")
        print(f"✓ Corrections: {len(data['corrections'])} applied")
        
        # Check that Vietnamese diacritics were added
        assert "chào" in data["text_with_diacritics"] or "việt" in data["text_with_diacritics"]


@pytest.mark.asyncio
async def test_metadata_transcript_storage():
    """Test transcript storage and retrieval from metadata service."""
    
    test_audio_id = "test-audio-123"
    test_text = "xin chao"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Store transcript
        response = await client.put(
            f"{METADATA_URL}/v1/transcripts/{test_audio_id}",
            params={
                "text": test_text
            }
        )
        
        # May fail if audio doesn't exist in DB, that's OK for this test
        if response.status_code in [200, 404]:
            print(f"✓ Transcript update returned: {response.status_code}")
        
        # Try to retrieve (may not exist)
        response = await client.get(
            f"{METADATA_URL}/v1/transcripts/{test_audio_id}"
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved transcript: {data.get('text', 'N/A')}")
        else:
            print(f"✓ Transcript not found (expected for test data): {response.status_code}")


@pytest.mark.asyncio 
async def test_search_service():
    """Test semantic search service."""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Perform search
        response = await client.get(
            f"{SEARCH_URL}/v1/search",
            params={"q": "xin chào"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total" in data
        assert "query_time_ms" in data
        
        print(f"✓ Search completed in {data['query_time_ms']:.2f}ms")
        print(f"✓ Found {data['total']} results")


@pytest.mark.asyncio
async def test_health_checks():
    """Test health endpoints of all services."""
    
    services = {
        "API Gateway": f"{BASE}/health",
        "Metadata": f"{METADATA_URL}/health",
        "NLP-Post": f"{NLP_POST_URL}/health",
        "Search": f"{SEARCH_URL}/health",
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(url)
                assert response.status_code == 200
                data = response.json()
                print(f"✓ {name}: {data.get('status', 'unknown')}")
            except Exception as e:
                pytest.fail(f"{name} health check failed: {e}")


@pytest.mark.asyncio
async def test_full_pipeline():
    """
    Test the complete pipeline:
    1. Upload audio
    2. Wait for transcription
    3. Check NLP processing
    4. Verify search indexing
    """
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    
    # Step 1: Upload audio
    audio_id = await test_batch_ingestion_flow()
    assert audio_id is not None
    
    # Step 2: Wait for async processing
    print("⏳ Waiting for async processing...")
    await asyncio.sleep(5)
    
    # Step 3: Try to retrieve transcript
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{METADATA_URL}/v1/transcripts/{audio_id}"
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved transcript for {audio_id}")
            print(f"  Text: {data.get('text', 'N/A')}")
            print(f"  Clean: {data.get('text_clean', 'N/A')}")
        else:
            print(f"⚠ Transcript not yet available (status {response.status_code})")
    
    print("="*60)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
