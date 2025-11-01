"""
Unit tests for base ASR model.

Tests for ASRModel, ASREncoder, and ASRDecoder classes.
"""

import pytest
import torch
import numpy as np
from models.asr_base import ASRModel, ASREncoder, ASRDecoder


class TestASREncoder:
    """Test ASREncoder class."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=256,
            num_layers=4,
            num_heads=4,
            d_ff=1024,
            dropout=0.1
        )
        assert encoder.input_dim == 80
        assert encoder.d_model == 256
    
    def test_forward_pass(self):
        """Test encoder forward pass."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            dropout=0.1
        )
        
        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len - 10])
        
        output, output_lengths = encoder(input_features, lengths)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert len(output_lengths) == batch_size
        assert torch.all(output_lengths <= lengths)
    
    def test_forward_with_mask(self):
        """Test encoder with sequence masking."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=128,
            num_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0  # Disable dropout for testing
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([50, 30])
        
        output, output_lengths = encoder(input_features, lengths)
        
        # Output length should match input length
        assert output.shape[1] == seq_len
        assert output.shape[0] == batch_size


class TestASRDecoder:
    """Test ASRDecoder class."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = ASRDecoder(d_model=256, vocab_size=100)
        assert decoder.d_model == 256
        assert decoder.vocab_size == 100
    
    def test_forward_pass(self):
        """Test decoder forward pass."""
        decoder = ASRDecoder(d_model=256, vocab_size=100)
        
        batch_size = 2
        seq_len = 100
        encoder_output = torch.randn(batch_size, seq_len, 256)
        
        logits = decoder(encoder_output)
        
        assert logits.shape == (batch_size, seq_len, 100)
        assert isinstance(logits, torch.Tensor)


class TestASRModel:
    """Test ASRModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            num_encoder_layers=4,
            num_heads=4,
            d_ff=1024,
            dropout=0.1
        )
        assert model.input_dim == 80
        assert model.vocab_size == 100
        assert model.d_model == 256
    
    def test_forward_pass(self, asr_model):
        """Test model forward pass."""
        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len - 20])
        
        logits, output_lengths = asr_model(input_features, lengths)
        
        assert logits.shape == (batch_size, seq_len, asr_model.vocab_size)
        assert len(output_lengths) == batch_size
        assert isinstance(logits, torch.Tensor)
    
    def test_forward_pass_no_lengths(self, asr_model):
        """Test forward pass without length tensor."""
        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        
        logits, output_lengths = asr_model(input_features)
        
        assert logits.shape == (batch_size, seq_len, asr_model.vocab_size)
        assert len(output_lengths) == batch_size
    
    def test_gradient_flow(self, asr_model):
        """Test that gradients flow correctly."""
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        logits, _ = asr_model(input_features, lengths)
        loss = logits.mean()
        loss.backward()
        
        # Check that gradients exist
        assert input_features.grad is not None
        # Check that model parameters have gradients
        has_grad = any(p.grad is not None for p in asr_model.parameters() if p.requires_grad)
        assert has_grad
    
    def test_model_eval_mode(self, asr_model):
        """Test model in eval mode."""
        asr_model.eval()
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len])
        
        with torch.no_grad():
            logits, _ = asr_model(input_features, lengths)
        
        assert logits.requires_grad == False
    
    def test_model_train_mode(self, asr_model):
        """Test model in train mode."""
        asr_model.train()
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len])
        
        logits, _ = asr_model(input_features, lengths)
        
        assert logits.requires_grad == True
    
    def test_parameter_count(self, asr_model):
        """Test parameter counting."""
        total_params = sum(p.numel() for p in asr_model.parameters())
        trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_different_batch_sizes(self, asr_model):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            seq_len = 50
            input_features = torch.randn(batch_size, seq_len, 80)
            lengths = torch.tensor([seq_len] * batch_size)
            
            logits, output_lengths = asr_model(input_features, lengths)
            
            assert logits.shape[0] == batch_size
            assert len(output_lengths) == batch_size
    
    def test_different_sequence_lengths(self, asr_model):
        """Test model with varying sequence lengths."""
        batch_size = 2
        input_features = torch.randn(batch_size, 200, 80)
        lengths = torch.tensor([200, 150])
        
        logits, output_lengths = asr_model(input_features, lengths)
        
        assert logits.shape[1] == 200  # Max sequence length
        assert output_lengths[1] <= output_lengths[0]  # Shorter sequence should have smaller length
    
    def test_output_shapes(self, asr_model):
        """Test output shapes are correct."""
        batch_size = 3
        seq_len = 75
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([75, 60, 50])
        
        logits, output_lengths = asr_model(input_features, lengths)
        
        assert logits.shape == (batch_size, seq_len, asr_model.vocab_size)
        assert output_lengths.shape == (batch_size,)
        assert torch.all(output_lengths <= lengths)

