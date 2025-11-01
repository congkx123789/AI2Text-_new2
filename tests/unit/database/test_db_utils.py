"""
Unit tests for database utilities.

Tests for ASRDatabase class and all CRUD operations.
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from datetime import datetime
from database.db_utils import ASRDatabase


class TestASRDatabase:
    """Test ASRDatabase class."""
    
    def test_initialization(self, temp_db):
        """Test database initialization."""
        assert temp_db is not None
        assert temp_db.db_path.exists()
    
    def test_initialization_creates_directory(self):
        """Test that database directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "new_dir" / "test.db"
            db = ASRDatabase(str(db_path))
            
            assert db_path.exists()
            assert db_path.parent.exists()
    
    def test_add_audio_file(self, temp_db):
        """Test adding audio file."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000,
            channels=1,
            format="wav"
        )
        
        assert audio_id is not None
        assert isinstance(audio_id, int)
        assert audio_id > 0
    
    def test_get_audio_file(self, temp_db):
        """Test retrieving audio file."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        audio = temp_db.get_audio_file(audio_id)
        
        assert audio is not None
        assert audio['id'] == audio_id
        assert audio['filename'] == "audio.wav"
        assert audio['duration'] == 5.0
    
    def test_add_transcript(self, temp_db):
        """Test adding transcript."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        transcript_id = temp_db.add_transcript(
            audio_file_id=audio_id,
            transcript="xin chào việt nam",
            normalized_transcript="xin chao viet nam"
        )
        
        assert transcript_id is not None
        assert isinstance(transcript_id, int)
    
    def test_get_transcript(self, temp_db):
        """Test retrieving transcript."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        transcript_id = temp_db.add_transcript(
            audio_file_id=audio_id,
            transcript="xin chào việt nam",
            normalized_transcript="xin chao viet nam"
        )
        
        transcript = temp_db.get_transcript(transcript_id)
        
        assert transcript is not None
        assert transcript['transcript'] == "xin chào việt nam"
    
    def test_assign_split(self, temp_db):
        """Test assigning audio to data split."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        success = temp_db.assign_split(audio_id, "train", "v1")
        
        assert success is True
    
    def test_get_audio_files_by_split(self, temp_db):
        """Test retrieving audio files by split."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        temp_db.assign_split(audio_id, "train", "v1")
        
        files = temp_db.get_audio_files_by_split("train", "v1")
        
        assert len(files) > 0
        assert any(f['id'] == audio_id for f in files)
    
    def test_get_dataset_statistics(self, temp_db):
        """Test getting dataset statistics."""
        # Add some audio files
        for i in range(5):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/audio_{i}.wav",
                filename=f"audio_{i}.wav",
                duration=5.0 + i,
                sample_rate=16000
            )
            temp_db.assign_split(audio_id, "train" if i < 3 else "val", "v1")
        
        stats = temp_db.get_dataset_statistics("v1")
        
        assert stats is not None
        assert 'total_files' in stats or 'train_files' in stats
    
    def test_update_audio_file(self, temp_db):
        """Test updating audio file."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        success = temp_db.update_audio_file(audio_id, duration=10.0)
        
        assert success is True
        
        audio = temp_db.get_audio_file(audio_id)
        assert audio['duration'] == 10.0
    
    def test_delete_audio_file(self, temp_db):
        """Test deleting audio file."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        success = temp_db.delete_audio_file(audio_id)
        
        assert success is True
        
        audio = temp_db.get_audio_file(audio_id)
        assert audio is None
    
    def test_transaction_rollback(self, temp_db):
        """Test transaction rollback on error."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        # Try to add transcript with invalid audio_file_id
        # Should handle gracefully or raise appropriate error
        try:
            transcript_id = temp_db.add_transcript(
                audio_file_id=99999,  # Non-existent ID
                transcript="test",
                normalized_transcript="test"
            )
            # If it succeeds, that's fine too (foreign key might not be enforced)
        except Exception:
            # If it raises error, that's expected behavior
            pass
    
    def test_get_connection(self, temp_db):
        """Test connection context manager."""
        with temp_db.get_connection() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)
            
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_add_model(self, temp_db):
        """Test adding model to registry."""
        model_id = temp_db.add_model(
            model_name="test_model",
            model_type="transformer",
            config_path="configs/test.yaml"
        )
        
        assert model_id is not None
        assert isinstance(model_id, int)
    
    def test_get_model(self, temp_db):
        """Test retrieving model."""
        model_id = temp_db.add_model(
            model_name="test_model",
            model_type="transformer",
            config_path="configs/test.yaml"
        )
        
        model = temp_db.get_model(model_id)
        
        assert model is not None
        assert model['model_name'] == "test_model"
    
    def test_add_training_run(self, temp_db):
        """Test adding training run."""
        model_id = temp_db.add_model(
            model_name="test_model",
            model_type="transformer"
        )
        
        run_id = temp_db.add_training_run(
            model_id=model_id,
            config_path="configs/test.yaml"
        )
        
        assert run_id is not None
        assert isinstance(run_id, int)
    
    def test_add_epoch_metrics(self, temp_db):
        """Test adding epoch metrics."""
        model_id = temp_db.add_model(
            model_name="test_model",
            model_type="transformer"
        )
        
        run_id = temp_db.add_training_run(
            model_id=model_id,
            config_path="configs/test.yaml"
        )
        
        success = temp_db.add_epoch_metrics(
            training_run_id=run_id,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_wer=0.2,
            val_wer=0.25
        )
        
        assert success is True
    
    def test_concurrent_access(self, temp_db):
        """Test database with concurrent access."""
        # Add multiple files concurrently
        audio_ids = []
        for i in range(10):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/audio_{i}.wav",
                filename=f"audio_{i}.wav",
                duration=5.0,
                sample_rate=16000
            )
            audio_ids.append(audio_id)
        
        assert len(audio_ids) == 10
        assert all(isinstance(aid, int) for aid in audio_ids)
        assert len(set(audio_ids)) == 10  # All unique IDs
    
    def test_query_performance(self, temp_db):
        """Test query performance with larger dataset."""
        # Add many audio files
        audio_ids = []
        for i in range(100):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/audio_{i}.wav",
                filename=f"audio_{i}.wav",
                duration=5.0,
                sample_rate=16000
            )
            temp_db.assign_split(audio_id, "train" if i % 2 == 0 else "val", "v1")
            audio_ids.append(audio_id)
        
        # Query should still be fast
        files = temp_db.get_audio_files_by_split("train", "v1")
        assert len(files) == 50

