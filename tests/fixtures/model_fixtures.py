"""
Model fixtures for testing.

Provides reusable model instances for tests.
"""

import torch
from models.asr_base import ASRModel
from models.enhanced_asr import EnhancedASRModel


def create_small_asr_model(input_dim=80, vocab_size=100, d_model=128):
    """Create a small ASR model for testing."""
    return ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1
    )


def create_small_enhanced_model(input_dim=80, vocab_size=100, d_model=128):
    """Create a small enhanced ASR model for testing."""
    return EnhancedASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1,
        use_contextual_embeddings=False,  # Disable for faster tests
        use_cross_modal_attention=False
    )


def create_dummy_input(batch_size=2, seq_len=50, input_dim=80):
    """Create dummy input features for testing."""
    return torch.randn(batch_size, seq_len, input_dim)


def create_dummy_lengths(batch_size=2, seq_len=50):
    """Create dummy sequence lengths for testing."""
    return torch.tensor([seq_len] * batch_size)

