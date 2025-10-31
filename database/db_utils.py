"""
Database utilities for ASR training system.
Provides easy-to-use functions for database operations.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import pandas as pd
from collections import defaultdict


class ASRDatabase:
    """Main database interface for ASR training system."""
    
    def __init__(self, db_path: str = "database/asr_training.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database with schema if not exists."""
        schema_path = Path(__file__).parent / "init_db.sql"
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # AudioFiles operations
    def add_audio_file(self, file_path: str, filename: str, 
                       duration: float, sample_rate: int,
                       channels: int = 1, format: str = "wav",
                       language: str = "vi", speaker_id: Optional[str] = None,
                       audio_quality: str = "medium", 
                       skip_duplicate: bool = True) -> Optional[int]:
        """Add a new audio file to database.
        
        Args:
            skip_duplicate: If True, skip if file_path already exists (returns None)
        
        Returns:
            audio_file_id: ID of inserted audio file, or None if duplicate
        """
        with self.get_connection() as conn:
            # Check for duplicate if requested
            if skip_duplicate:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", (file_path,)
                )
                existing = cursor.fetchone()
                if existing:
                    return None  # Skip duplicate
            
            cursor = conn.execute("""
                INSERT INTO AudioFiles 
                (file_path, filename, duration_seconds, sample_rate, channels, 
                 format, language, speaker_id, audio_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, filename, duration, sample_rate, channels, 
                  format, language, speaker_id, audio_quality))
            conn.commit()
            return cursor.lastrowid
    
    def batch_add_audio_files(self, audio_data: List[Dict]) -> List[int]:
        """Batch add multiple audio files efficiently.
        
        Args:
            audio_data: List of dicts with audio file information
            
        Returns:
            List of audio_file_ids (None for skipped duplicates)
        """
        ids = []
        with self.get_connection() as conn:
            for data in audio_data:
                # Check for duplicate
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", 
                    (data['file_path'],)
                )
                if cursor.fetchone():
                    ids.append(None)
                    continue
                
                cursor = conn.execute("""
                    INSERT INTO AudioFiles 
                    (file_path, filename, duration_seconds, sample_rate, channels, 
                     format, language, speaker_id, audio_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['file_path'], data['filename'], data['duration'],
                    data['sample_rate'], data['channels'], data['format'],
                    data['language'], data.get('speaker_id'), data['audio_quality']
                ))
                ids.append(cursor.lastrowid)
            conn.commit()
        return ids
    
    def get_audio_file(self, audio_file_id: int) -> Optional[Dict]:
        """Get audio file by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM AudioFiles WHERE id = ?", (audio_file_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # Transcripts operations
    def add_transcript(self, audio_file_id: int, transcript: str,
                       normalized_transcript: Optional[str] = None,
                       annotation_type: str = "manual",
                       confidence_score: Optional[float] = None,
                       annotator_id: Optional[str] = None) -> int:
        """Add transcript for an audio file.
        
        Returns:
            transcript_id: ID of inserted transcript
        """
        word_count = len(transcript.split())
        char_count = len(transcript)
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Transcripts 
                (audio_file_id, transcript, normalized_transcript, 
                 word_count, char_count, annotation_type, 
                 confidence_score, annotator_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (audio_file_id, transcript, normalized_transcript,
                  word_count, char_count, annotation_type,
                  confidence_score, annotator_id))
            conn.commit()
            return cursor.lastrowid
    
    # DataSplits operations
    def assign_split(self, audio_file_id: int, split_type: str, 
                     split_version: str = "v1") -> int:
        """Assign audio file to train/val/test split.
        
        Args:
            split_type: One of 'train', 'val', 'test'
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO DataSplits 
                (audio_file_id, split_type, split_version)
                VALUES (?, ?, ?)
            """, (audio_file_id, split_type, split_version))
            conn.commit()
            return cursor.lastrowid
    
    def get_split_data(self, split_type: str, split_version: str = "v1") -> pd.DataFrame:
        """Get all audio files and transcripts for a split."""
        query = """
            SELECT 
                af.id as audio_file_id,
                af.file_path,
                af.filename,
                af.duration_seconds,
                af.sample_rate,
                af.speaker_id,
                t.transcript,
                t.normalized_transcript
            FROM DataSplits ds
            JOIN AudioFiles af ON ds.audio_file_id = af.id
            JOIN Transcripts t ON af.id = t.audio_file_id
            WHERE ds.split_type = ? AND ds.split_version = ?
            ORDER BY af.id
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(split_type, split_version))
    
    # Models operations
    def add_model(self, model_name: str, model_type: str,
                  architecture: str, version: str,
                  checkpoint_path: Optional[str] = None,
                  config: Optional[Dict] = None,
                  total_parameters: Optional[int] = None) -> int:
        """Register a new model."""
        config_json = json.dumps(config) if config else None
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Models 
                (model_name, model_type, architecture, version, 
                 checkpoint_path, config_json, total_parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_name, model_type, architecture, version,
                  checkpoint_path, config_json, total_parameters))
            conn.commit()
            return cursor.lastrowid
    
    # TrainingRuns operations
    def create_training_run(self, model_id: int, run_name: str,
                            config: Dict, batch_size: int,
                            learning_rate: float, num_epochs: int,
                            optimizer: str, description: Optional[str] = None,
                            split_version: str = "v1",
                            gpu_name: Optional[str] = None) -> int:
        """Create a new training run."""
        config_json = json.dumps(config)
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO TrainingRuns 
                (model_id, run_name, description, config_json, split_version,
                 batch_size, learning_rate, num_epochs, optimizer, gpu_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_id, run_name, description, config_json, split_version,
                  batch_size, learning_rate, num_epochs, optimizer, gpu_name))
            conn.commit()
            return cursor.lastrowid
    
    def update_training_run(self, run_id: int, **kwargs):
        """Update training run with results."""
        set_clauses = []
        values = []
        
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)
        
        if not set_clauses:
            return
        
        values.append(run_id)
        query = f"UPDATE TrainingRuns SET {', '.join(set_clauses)} WHERE id = ?"
        
        with self.get_connection() as conn:
            conn.execute(query, values)
            conn.commit()
    
    def complete_training_run(self, run_id: int, final_train_loss: float,
                              final_val_loss: float, best_val_loss: float,
                              best_epoch: int, wer: float, cer: float,
                              total_time: float):
        """Mark training run as completed with final metrics."""
        self.update_training_run(
            run_id,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            word_error_rate=wer,
            character_error_rate=cer,
            total_training_time_seconds=total_time,
            status='completed',
            completed_at=datetime.now().isoformat()
        )
    
    # EpochMetrics operations
    def add_epoch_metrics(self, training_run_id: int, epoch: int,
                          train_loss: float, val_loss: float,
                          learning_rate: float, wer: Optional[float] = None,
                          cer: Optional[float] = None,
                          epoch_time: Optional[float] = None):
        """Log metrics for an epoch."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO EpochMetrics 
                (training_run_id, epoch, train_loss, val_loss, 
                 learning_rate, wer, cer, epoch_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (training_run_id, epoch, train_loss, val_loss,
                  learning_rate, wer, cer, epoch_time))
            conn.commit()
    
    def get_training_history(self, training_run_id: int) -> pd.DataFrame:
        """Get all epoch metrics for a training run."""
        query = """
            SELECT * FROM EpochMetrics 
            WHERE training_run_id = ? 
            ORDER BY epoch
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(training_run_id,))
    
    # Predictions operations
    def add_prediction(self, training_run_id: int, audio_file_id: int,
                       predicted_text: str, ground_truth: str,
                       confidence_score: Optional[float] = None,
                       inference_time_ms: Optional[float] = None,
                       wer: Optional[float] = None,
                       cer: Optional[float] = None) -> int:
        """Store a model prediction."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Predictions 
                (training_run_id, audio_file_id, predicted_text, ground_truth,
                 confidence_score, inference_time_ms, wer, cer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (training_run_id, audio_file_id, predicted_text, ground_truth,
                  confidence_score, inference_time_ms, wer, cer))
            conn.commit()
            return cursor.lastrowid
    
    # Analytics and reporting
    def get_dataset_statistics(self, split_version: str = "v1") -> pd.DataFrame:
        """Get dataset statistics using the view."""
        query = "SELECT * FROM DatasetStats WHERE split_version = ?"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(split_version,))
    
    def validate_data_for_training(self, split_version: str = "v1") -> Dict:
        """Validate data quality and prepare training readiness report.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'is_ready': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        with self.get_connection() as conn:
            # Check splits exist
            cursor = conn.execute("""
                SELECT split_type, COUNT(*) as count 
                FROM DataSplits 
                WHERE split_version = ?
                GROUP BY split_type
            """, (split_version,))
            
            splits = {row['split_type']: row['count'] for row in cursor.fetchall()}
            
            if 'train' not in splits or splits['train'] == 0:
                validation_results['is_ready'] = False
                validation_results['issues'].append("No training data found!")
            
            if 'val' not in splits or splits['val'] == 0:
                validation_results['warnings'].append("No validation data found!")
            
            # Check for transcripts
            cursor = conn.execute("""
                SELECT COUNT(*) as count_without_transcript
                FROM AudioFiles af
                LEFT JOIN Transcripts t ON af.id = t.audio_file_id
                WHERE t.id IS NULL
            """)
            no_transcript = cursor.fetchone()['count_without_transcript']
            
            if no_transcript > 0:
                validation_results['issues'].append(
                    f"{no_transcript} audio files without transcripts!"
                )
                validation_results['is_ready'] = False
            
            # Check audio quality distribution
            cursor = conn.execute("""
                SELECT audio_quality, COUNT(*) as count
                FROM AudioFiles
                GROUP BY audio_quality
            """)
            quality_dist = {row['audio_quality']: row['count'] for row in cursor.fetchall()}
            
            # Check duration statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(duration_seconds) as avg_duration,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration,
                    SUM(CASE WHEN duration_seconds < 0.5 THEN 1 ELSE 0 END) as too_short,
                    SUM(CASE WHEN duration_seconds > 30 THEN 1 ELSE 0 END) as too_long
                FROM AudioFiles
            """)
            duration_stats = dict(cursor.fetchone())
            
            # Compile statistics
            validation_results['statistics'] = {
                'splits': splits,
                'quality_distribution': quality_dist,
                'duration_stats': duration_stats,
                'files_without_transcripts': no_transcript
            }
            
            # Generate recommendations
            total_files = sum(splits.values())
            if total_files < 100:
                validation_results['recommendations'].append(
                    f"Only {total_files} files. Recommend at least 100 files for basic training."
                )
            elif total_files < 1000:
                validation_results['recommendations'].append(
                    f"{total_files} files is good for initial training. More data will improve results."
                )
            
            if duration_stats['too_short'] > 0:
                validation_results['warnings'].append(
                    f"{duration_stats['too_short']} files are too short (<0.5s). Consider filtering."
                )
            
            if duration_stats['too_long'] > 0:
                validation_results['warnings'].append(
                    f"{duration_stats['too_long']} files are too long (>30s). Consider splitting."
                )
            
            train_count = splits.get('train', 0)
            if train_count < 50:
                validation_results['is_ready'] = False
                validation_results['issues'].append(
                    f"Only {train_count} training samples. Need at least 50."
                )
        
        return validation_results
    
    def get_data_summary(self, split_version: str = "v1") -> Dict:
        """Get comprehensive data summary for training preparation."""
        with self.get_connection() as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(DISTINCT af.id) as total_files,
                    COUNT(DISTINCT af.speaker_id) as total_speakers,
                    SUM(af.duration_seconds) as total_duration_hours,
                    AVG(af.duration_seconds) as avg_duration,
                    AVG(t.word_count) as avg_words_per_transcript
                FROM AudioFiles af
                JOIN Transcripts t ON af.id = t.audio_file_id
            """)
            overall = dict(cursor.fetchone())
            overall['total_duration_hours'] = overall['total_duration_hours'] / 3600 if overall['total_duration_hours'] else 0
            
            # Split statistics
            stats = self.get_dataset_statistics(split_version)
            stats_dict = stats.to_dict('records')
            
            return {
                'overall': overall,
                'splits': {row['split_type']: dict(row) for row in stats_dict},
                'validation': self.validate_data_for_training(split_version)
            }
    
    def get_training_runs_summary(self, limit: int = 10) -> pd.DataFrame:
        """Get summary of recent training runs."""
        query = """
            SELECT * FROM TrainingRunSummary 
            ORDER BY started_at DESC 
            LIMIT ?
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_best_model(self, metric: str = "word_error_rate") -> Optional[Dict]:
        """Get best performing model based on a metric."""
        query = f"""
            SELECT * FROM TrainingRuns 
            WHERE status = 'completed' AND {metric} IS NOT NULL
            ORDER BY {metric} ASC 
            LIMIT 1
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None


# Convenience functions
def get_db(db_path: str = "database/asr_training.db") -> ASRDatabase:
    """Get database instance."""
    return ASRDatabase(db_path)


if __name__ == "__main__":
    # Test database setup
    db = ASRDatabase("database/asr_training.db")
    print("Database initialized successfully!")
    print(f"Database location: {db.db_path.absolute()}")

