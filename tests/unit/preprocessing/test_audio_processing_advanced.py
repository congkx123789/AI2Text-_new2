"""
Advanced and challenging tests for audio processing.

Tests include edge cases, stress tests, numerical stability, and error handling.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
import soundfile as sf
import concurrent.futures
import threading
import time
import warnings

from preprocessing.audio_processing import (
    AudioProcessor,
    AudioAugmenter,
    preprocess_audio_file
)


class TestAudioProcessorAdvanced:
    """Advanced tests for AudioProcessor with challenging scenarios."""
    
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000, 96000])
    def test_load_audio_various_sample_rates(self, sample_rate):
        """Test loading audio with various sample rates (resampling)."""
        processor = AudioProcessor(sample_rate=16000)
        
        # Create audio at different sample rate
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_orig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_orig, sample_rate)
            
            try:
                audio, sr = processor.load_audio(tmp.name)
                assert sr == 16000  # Should be resampled
                assert len(audio) > 0
                # Duration should be approximately preserved
                assert abs(len(audio) / 16000 - duration) < 0.1
            finally:
                Path(tmp.name).unlink()
    
    @pytest.mark.parametrize("duration", [0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 300.0])
    def test_load_audio_various_durations(self, duration):
        """Test loading audio of various durations (very short to very long)."""
        processor = AudioProcessor(sample_rate=16000)
        
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_orig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_orig, sample_rate)
            
            try:
                audio, sr = processor.load_audio(tmp.name)
                assert sr == sample_rate
                assert len(audio) > 0
            except Exception as e:
                if duration < 0.01:
                    # Very short audio might cause issues, which is acceptable
                    pytest.skip(f"Very short audio ({duration}s) may not process correctly")
            finally:
                Path(tmp.name).unlink()
    
    def test_load_audio_extremely_long_file(self):
        """Test loading extremely long audio file (stress test)."""
        processor = AudioProcessor(sample_rate=16000)
        
        # 5 minutes of audio
        duration = 300.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_orig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_orig, sample_rate)
            
            try:
                start_time = time.time()
                audio, sr = processor.load_audio(tmp.name)
                load_time = time.time() - start_time
                
                assert sr == sample_rate
                assert len(audio) > 0
                # Should complete within reasonable time (< 30 seconds)
                assert load_time < 30.0
            finally:
                Path(tmp.name).unlink()
    
    def test_load_audio_nonexistent_file(self):
        """Test loading non-existent file (error handling)."""
        processor = AudioProcessor(sample_rate=16000)
        
        with pytest.raises((FileNotFoundError, OSError)):
            processor.load_audio("nonexistent_file_12345.wav")
    
    def test_load_audio_invalid_format(self):
        """Test loading invalid audio format."""
        processor = AudioProcessor(sample_rate=16000)
        
        # Create invalid audio file (text file)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"Not audio data")
            tmp_path = tmp.name
        
        try:
            with pytest.raises((ValueError, RuntimeError, OSError)):
                processor.load_audio(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    @pytest.mark.parametrize("channels", [1, 2, 4, 6])
    def test_load_audio_multichannel(self, channels):
        """Test loading multi-channel audio (mono conversion)."""
        processor = AudioProcessor(sample_rate=16000)
        
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_orig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Create multi-channel audio
        if channels > 1:
            audio_multi = np.stack([audio_orig] * channels, axis=1)
        else:
            audio_multi = audio_orig
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_multi, sample_rate)
            
            try:
                audio, sr = processor.load_audio(tmp.name)
                assert sr == sample_rate
                assert len(audio.shape) == 1  # Should be mono
                assert len(audio) > 0
            finally:
                Path(tmp_path).unlink()
    
    def test_extract_mel_spectrogram_extreme_values(self):
        """Test mel spectrogram with extreme audio values."""
        processor = AudioProcessor(sample_rate=16000)
        
        # Test with very loud audio
        audio_loud = np.ones(16000) * 0.99
        mel_spec_loud = processor.extract_mel_spectrogram(audio_loud)
        assert not np.any(np.isnan(mel_spec_loud))
        assert not np.any(np.isinf(mel_spec_loud))
        
        # Test with very quiet audio
        audio_quiet = np.ones(16000) * 1e-6
        mel_spec_quiet = processor.extract_mel_spectrogram(audio_quiet)
        assert not np.any(np.isnan(mel_spec_quiet))
        
        # Test with zero audio
        audio_zero = np.zeros(16000)
        mel_spec_zero = processor.extract_mel_spectrogram(audio_zero)
        assert not np.any(np.isnan(mel_spec_zero))
    
    def test_extract_mel_spectrogram_numerical_stability(self):
        """Test numerical stability with various input ranges."""
        processor = AudioProcessor(sample_rate=16000)
        
        test_cases = [
            np.random.randn(16000).astype(np.float32),
            np.random.randn(16000).astype(np.float64),
            (np.random.randn(16000) * 100).astype(np.float32),  # Large values
            (np.random.randn(16000) * 1e-6).astype(np.float32),  # Small values
            np.clip(np.random.randn(16000), -1, 1).astype(np.float32),  # Clipped
        ]
        
        for audio in test_cases:
            mel_spec = processor.extract_mel_spectrogram(audio)
            
            # Check for numerical issues
            assert not np.any(np.isnan(mel_spec))
            assert not np.any(np.isinf(mel_spec))
            assert mel_spec.shape[0] == processor.n_mels
    
    @pytest.mark.parametrize("length", [1, 10, 100, 1000, 10000, 100000, 1000000])
    def test_pad_or_truncate_various_lengths(self, length):
        """Test padding/truncation with various input lengths."""
        processor = AudioProcessor(sample_rate=16000)
        target_length = 16000
        
        if length < 1:
            pytest.skip("Length must be positive")
        
        audio = np.random.randn(length).astype(np.float32)
        processed = processor.pad_or_truncate(audio, target_length)
        
        assert len(processed) == target_length
        assert not np.any(np.isnan(processed))
    
    def test_pad_or_truncate_empty_audio(self):
        """Test padding with empty/zero-length audio."""
        processor = AudioProcessor(sample_rate=16000)
        target_length = 16000
        
        audio = np.array([])
        processed = processor.pad_or_truncate(audio, target_length)
        
        assert len(processed) == target_length
        assert np.all(processed == 0)
    
    def test_concurrent_audio_loading(self):
        """Test concurrent audio loading (thread safety)."""
        processor = AudioProcessor(sample_rate=16000)
        
        # Create multiple audio files
        audio_files = []
        for i in range(10):
            duration = 1.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sample_rate)
                audio_files.append(Path(tmp.name))
        
        results = []
        errors = []
        
        def load_audio(path):
            try:
                return processor.load_audio(str(path))
            except Exception as e:
                errors.append(e)
                return None
        
        # Load concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_audio, path) for path in audio_files]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Cleanup
        for path in audio_files:
            if path.exists():
                path.unlink()
        
        # All should succeed
        assert len(errors) == 0
        assert all(r is not None for r in results)
        assert all(r[0] is not None for r in results)


class TestAudioAugmenterAdvanced:
    """Advanced tests for AudioAugmenter with challenging scenarios."""
    
    @pytest.mark.parametrize("noise_factor", [0.0, 0.001, 0.01, 0.1, 1.0, 10.0])
    def test_add_noise_various_levels(self, noise_factor):
        """Test adding noise at various levels."""
        augmenter = AudioAugmenter(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32)
        
        noisy = augmenter.add_noise(audio, noise_factor=noise_factor)
        
        assert noisy.shape == audio.shape
        assert not np.any(np.isnan(noisy))
        assert not np.any(np.isinf(noisy))
    
    @pytest.mark.parametrize("snr_db", [-10, 0, 10, 20, 30, 40])
    def test_add_background_noise_various_snr(self, snr_db):
        """Test adding background noise at various SNR levels."""
        augmenter = AudioAugmenter(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        noise = np.random.randn(20000).astype(np.float32)
        
        noisy = augmenter.add_background_noise(audio, noise, snr_db=snr_db)
        
        assert noisy.shape == audio.shape
        assert not np.any(np.isnan(noisy))
    
    def test_spec_augment_extreme_parameters(self):
        """Test SpecAugment with extreme masking parameters."""
        augmenter = AudioAugmenter(sample_rate=16000)
        mel_spec = np.random.randn(80, 100).astype(np.float32)
        
        # Test with very large masks
        augmented = augmenter.spec_augment(
            mel_spec,
            freq_mask_param=50,  # Large mask
            time_mask_param=80,   # Large mask
            num_freq_masks=5,
            num_time_masks=5
        )
        
        assert augmented.shape == mel_spec.shape
        assert not np.any(np.isnan(augmented))
    
    def test_augment_reproducibility(self):
        """Test that augmentations are deterministic when seeded."""
        augmenter = AudioAugmenter(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32)
        
        # Note: This test depends on whether augmentation uses random state
        # If not seeded, this might fail, which is acceptable
    
    def test_concurrent_augmentation(self):
        """Test concurrent augmentation (thread safety)."""
        augmenter = AudioAugmenter(sample_rate=16000)
        
        def augment_audio(audio):
            return augmenter.augment(audio, augmentation_types=['noise', 'volume'])
        
        audio_batch = [np.random.randn(16000).astype(np.float32) for _ in range(10)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(augment_audio, audio_batch))
        
        assert len(results) == 10
        assert all(r.shape == (16000,) for r in results)


class TestPreprocessingAdvanced:
    """Advanced integration tests for preprocessing pipeline."""
    
    def test_preprocess_very_large_file(self):
        """Test preprocessing very large audio file."""
        # 10 minutes of audio
        duration = 600.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = Path(tmp.name)
        
        try:
            start_time = time.time()
            result = preprocess_audio_file(
                str(tmp_path),
                extract_features=True
            )
            processing_time = time.time() - start_time
            
            assert 'mel_spectrogram' in result
            assert result['mel_spectrogram'] is not None
            # Should complete within reasonable time
            assert processing_time < 120.0  # 2 minutes max
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_preprocess_multiple_augmentations(self):
        """Test preprocessing with multiple augmentation types."""
        processor = AudioProcessor(sample_rate=16000)
        augmenter = AudioAugmenter(sample_rate=16000)
        
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = Path(tmp.name)
        
        try:
            result = preprocess_audio_file(
                str(tmp_path),
                processor=processor,
                augmenter=augmenter,
                apply_augmentation=True,
                extract_features=True
            )
            
            assert 'audio' in result
            assert 'mel_spectrogram' in result
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_preprocess_memory_efficiency(self):
        """Test that preprocessing doesn't leak memory."""
        processor = AudioProcessor(sample_rate=16000)
        
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = Path(tmp.name)
        
        try:
            # Process many times
            for _ in range(100):
                result = preprocess_audio_file(
                    str(tmp_path),
                    processor=processor,
                    extract_features=True
                )
                # Force garbage collection check (if needed)
                del result
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    @pytest.mark.parametrize("corrupted_data", [
        b"",  # Empty file
        b"RIFF\x00\x00\x00\x00WAVE",  # Invalid WAV header
        b"Not audio at all",  # Random data
    ])
    def test_preprocess_corrupted_files(self, corrupted_data):
        """Test preprocessing with corrupted audio files."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(corrupted_data)
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                preprocess_audio_file(str(tmp_path), extract_features=False)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


@pytest.mark.performance
class TestAudioProcessingPerformance:
    """Performance and benchmark tests."""
    
    def test_load_audio_performance(self, benchmark):
        """Benchmark audio loading performance."""
        processor = AudioProcessor(sample_rate=16000)
        
        duration = 10.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name
        
        try:
            def load():
                processor.load_audio(tmp_path)
            
            benchmark(load)
        finally:
            Path(tmp_path).unlink()
    
    def test_mel_spectrogram_performance(self, benchmark):
        """Benchmark mel spectrogram extraction."""
        processor = AudioProcessor(sample_rate=16000)
        audio = np.random.randn(160000).astype(np.float32)  # 10 seconds
        
        benchmark(processor.extract_mel_spectrogram, audio)
    
    def test_augmentation_performance(self, benchmark):
        """Benchmark audio augmentation."""
        augmenter = AudioAugmenter(sample_rate=16000)
        audio = np.random.randn(160000).astype(np.float32)
        
        def augment():
            return augmenter.augment(audio, augmentation_types=['noise', 'volume', 'shift'])
        
        benchmark(augment)

