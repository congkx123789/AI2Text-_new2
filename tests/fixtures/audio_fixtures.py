"""
Audio fixtures for testing.

Provides reusable audio data for tests.
"""

import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path


def create_sine_wave(duration=1.0, sample_rate=16000, frequency=440.0, amplitude=0.5):
    """Create a sine wave audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def create_noise(duration=1.0, sample_rate=16000, noise_level=0.1):
    """Create white noise audio signal."""
    samples = int(sample_rate * duration)
    audio = np.random.randn(samples).astype(np.float32) * noise_level
    return audio


def create_mixed_audio(duration=1.0, sample_rate=16000):
    """Create mixed audio signal (sine + noise)."""
    sine = create_sine_wave(duration, sample_rate, frequency=440.0)
    noise = create_noise(duration, sample_rate, noise_level=0.05)
    return (sine + noise).astype(np.float32)


def save_audio_file(audio, sample_rate=16000, suffix='.wav'):
    """Save audio to temporary file and return path."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    sf.write(tmp_file.name, audio, sample_rate)
    return Path(tmp_file.name)


def create_silent_audio(duration=1.0, sample_rate=16000):
    """Create silent audio (all zeros)."""
    samples = int(sample_rate * duration)
    return np.zeros(samples, dtype=np.float32)

