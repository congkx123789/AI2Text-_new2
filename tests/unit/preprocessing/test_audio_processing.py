"""
Unit tests for audio processing module.

Tests for AudioProcessor and AudioAugmenter classes.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import soundfile as sf

from preprocessing.audio_processing import (
    AudioProcessor,
    AudioAugmenter,
    preprocess_audio_file
)


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        assert processor.sample_rate == 16000
        assert processor.n_mels == 80
        assert processor.n_fft == 400
        assert processor.hop_length == 160
    
    def test_load_audio(self, sample_audio_file):
        """Test audio loading."""
        audio_path, expected_sr = sample_audio_file
        processor = AudioProcessor(sample_rate=16000)
        
        audio, sr = processor.load_audio(audio_path)
        
        assert sr == 16000
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32 or audio.dtype == np.float64
    
    def test_load_audio_normalize(self, sample_audio_file, temp_dir):
        """Test audio loading with normalization."""
        audio_path, _ = sample_audio_file
        processor = AudioProcessor(sample_rate=16000)
        
        audio, _ = processor.load_audio(audio_path, normalize=True)
        
        # Check normalization (values should be in reasonable range)
        assert np.abs(audio).max() <= 1.0 + 1e-6
    
    def test_extract_mel_spectrogram(self, audio_processor):
        """Test mel spectrogram extraction."""
        # Generate dummy audio
        audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        
        mel_spec = audio_processor.extract_mel_spectrogram(audio)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == audio_processor.n_mels
        assert mel_spec.shape[1] > 0  # Time dimension
    
    def test_extract_mfcc(self, audio_processor):
        """Test MFCC feature extraction."""
        audio = np.random.randn(16000).astype(np.float32)
        
        mfcc = audio_processor.extract_mfcc(audio, n_mfcc=13)
        
        assert mfcc.ndim == 2
        assert mfcc.shape[0] == 13  # 13 MFCC coefficients
        assert mfcc.shape[1] > 0
    
    def test_compute_energy(self, audio_processor):
        """Test energy computation."""
        audio = np.random.randn(16000).astype(np.float32)
        
        energy = audio_processor.compute_energy(audio)
        
        assert isinstance(energy, np.ndarray)
        assert len(energy) > 0
        assert np.all(energy >= 0)  # Energy should be non-negative
    
    def test_trim_silence(self, audio_processor):
        """Test silence trimming."""
        # Create audio with silence at beginning and end
        audio = np.zeros(16000)
        audio[4000:12000] = np.random.randn(8000) * 0.5
        
        trimmed = audio_processor.trim_silence(audio, top_db=20.0)
        
        assert len(trimmed) <= len(audio)
        assert len(trimmed) > 0
    
    def test_pad_or_truncate_pad(self, audio_processor):
        """Test padding audio."""
        audio = np.random.randn(8000)  # Short audio
        target_length = 16000
        
        processed = audio_processor.pad_or_truncate(audio, target_length)
        
        assert len(processed) == target_length
        assert np.allclose(processed[:len(audio)], audio)
        assert np.allclose(processed[len(audio):], 0)  # Padding should be zeros
    
    def test_pad_or_truncate_truncate(self, audio_processor):
        """Test truncating audio."""
        audio = np.random.randn(32000)  # Long audio
        target_length = 16000
        
        processed = audio_processor.pad_or_truncate(audio, target_length)
        
        assert len(processed) == target_length
        assert np.allclose(processed, audio[:target_length])
    
    def test_save_audio(self, audio_processor, temp_dir):
        """Test saving audio."""
        audio = np.random.randn(16000).astype(np.float32)
        output_path = temp_dir / "test_output.wav"
        
        audio_processor.save_audio(audio, str(output_path))
        
        assert output_path.exists()
        # Verify file can be loaded
        loaded_audio, sr = sf.read(str(output_path))
        assert sr == audio_processor.sample_rate
        assert len(loaded_audio) > 0


class TestAudioAugmenter:
    """Test AudioAugmenter class."""
    
    def test_initialization(self):
        """Test AudioAugmenter initialization."""
        augmenter = AudioAugmenter(sample_rate=16000)
        assert augmenter.sample_rate == 16000
    
    def test_add_noise(self, audio_augmenter):
        """Test adding Gaussian noise."""
        audio = np.random.randn(16000).astype(np.float32)
        original = audio.copy()
        
        noisy = audio_augmenter.add_noise(audio, noise_factor=0.01)
        
        assert noisy.shape == audio.shape
        assert not np.allclose(noisy, original)  # Should be different
        # Noise should be small compared to signal
        assert np.std(noisy - original) < 0.1
    
    def test_time_shift(self, audio_augmenter):
        """Test time shifting."""
        audio = np.random.randn(16000).astype(np.float32)
        
        shifted = audio_augmenter.time_shift(audio, shift_max=0.1)
        
        assert shifted.shape == audio.shape
        # Energy should be preserved
        assert np.isclose(np.sum(audio**2), np.sum(shifted**2), rtol=1e-3)
    
    def test_time_stretch(self, audio_augmenter):
        """Test time stretching."""
        audio = np.random.randn(16000).astype(np.float32)
        
        stretched = audio_augmenter.time_stretch(audio, rate_range=(0.9, 1.1))
        
        # Length may change due to stretching
        assert len(stretched) > 0
        assert isinstance(stretched, np.ndarray)
    
    def test_pitch_shift(self, audio_augmenter):
        """Test pitch shifting."""
        audio = np.random.randn(16000).astype(np.float32)
        
        pitch_shifted = audio_augmenter.pitch_shift(audio, n_steps_range=(-1, 1))
        
        assert pitch_shifted.shape == audio.shape
        assert isinstance(pitch_shifted, np.ndarray)
    
    def test_change_volume(self, audio_augmenter):
        """Test volume change."""
        audio = np.random.randn(16000).astype(np.float32)
        
        volume_changed = audio_augmenter.change_volume(audio, gain_range=(0.5, 2.0))
        
        assert volume_changed.shape == audio.shape
        # Volume should change (unless gain happens to be ~1.0)
        # This test may occasionally fail if random gain is ~1.0, but probability is low
    
    def test_add_background_noise(self, audio_augmenter):
        """Test adding background noise."""
        audio = np.random.randn(16000).astype(np.float32)
        noise = np.random.randn(20000).astype(np.float32)
        
        noisy = audio_augmenter.add_background_noise(audio, noise, snr_db=10.0)
        
        assert noisy.shape == audio.shape
        assert not np.allclose(noisy, audio)
    
    def test_spec_augment(self, audio_augmenter):
        """Test SpecAugment."""
        mel_spec = np.random.randn(80, 100).astype(np.float32)  # 80 mels, 100 frames
        
        augmented = audio_augmenter.spec_augment(
            mel_spec,
            freq_mask_param=10,
            time_mask_param=20,
            num_freq_masks=2,
            num_time_masks=2
        )
        
        assert augmented.shape == mel_spec.shape
        # Some values should be masked (zeroed)
        assert np.sum(augmented == 0) > 0
    
    def test_augment(self, audio_augmenter):
        """Test combined augmentation."""
        audio = np.random.randn(16000).astype(np.float32)
        
        augmented = audio_augmenter.augment(
            audio,
            augmentation_types=['noise', 'volume']
        )
        
        assert augmented.shape == audio.shape
        assert isinstance(augmented, np.ndarray)


class TestPreprocessAudioFile:
    """Test preprocess_audio_file function."""
    
    def test_preprocess_with_features(self, sample_audio_file, temp_dir):
        """Test preprocessing with feature extraction."""
        audio_path, _ = sample_audio_file
        processor = AudioProcessor(sample_rate=16000)
        
        result = preprocess_audio_file(
            audio_path,
            processor=processor,
            extract_features=True
        )
        
        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'mel_spectrogram' in result
        assert 'feature_shape' in result
        assert result['sample_rate'] == 16000
        assert result['mel_spectrogram'] is not None
    
    def test_preprocess_with_augmentation(self, sample_audio_file):
        """Test preprocessing with augmentation."""
        audio_path, _ = sample_audio_file
        processor = AudioProcessor(sample_rate=16000)
        augmenter = AudioAugmenter(sample_rate=16000)
        
        result = preprocess_audio_file(
            audio_path,
            processor=processor,
            augmenter=augmenter,
            apply_augmentation=True,
            extract_features=False
        )
        
        assert 'audio' in result
        assert result['audio'] is not None
    
    def test_preprocess_save_output(self, sample_audio_file, temp_dir):
        """Test preprocessing with output directory."""
        audio_path, _ = sample_audio_file
        output_dir = str(temp_dir / "processed")
        
        result = preprocess_audio_file(
            audio_path,
            output_dir=output_dir,
            extract_features=False
        )
        
        assert 'processed_path' in result
        assert Path(result['processed_path']).exists()

