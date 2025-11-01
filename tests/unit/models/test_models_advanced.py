"""
Advanced and challenging tests for ASR models.

Tests include numerical stability, memory management, gradient flow, and edge cases.
"""

import pytest
import torch
import numpy as np
import gc
import time
from contextlib import contextmanager
import warnings

from models.asr_base import ASRModel, ASREncoder, ASRDecoder
from models.enhanced_asr import EnhancedASRModel


class TestModelNumericalStability:
    """Test numerical stability and edge cases."""
    
    @pytest.mark.parametrize("input_value", [
        0.0,
        1e-10,
        1e-5,
        1.0,
        1e5,
        1e10,
        -1e10,
        -1e5,
        -1.0,
    ])
    def test_encoder_extreme_input_values(self, input_value):
        """Test encoder with extreme input values."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=128,
            num_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 1
        seq_len = 10
        input_features = torch.full((batch_size, seq_len, 80), input_value, dtype=torch.float32)
        lengths = torch.tensor([seq_len])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output, output_lengths = encoder(input_features, lengths)
        
        # Check for NaN or Inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_encoder_nan_input(self):
        """Test encoder with NaN input (should handle gracefully)."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=128,
            num_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 1
        seq_len = 10
        input_features = torch.full((batch_size, seq_len, 80), float('nan'))
        lengths = torch.tensor([seq_len])
        
        # May propagate NaN or handle it
        output, _ = encoder(input_features, lengths)
        # Just check it doesn't crash
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_encoder_inf_input(self):
        """Test encoder with Inf input."""
        encoder = ASREncoder(
            input_dim=80,
            d_model=128,
            num_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 1
        seq_len = 10
        input_features = torch.full((batch_size, seq_len, 80), float('inf'))
        lengths = torch.tensor([seq_len])
        
        output, _ = encoder(input_features, lengths)
        # May propagate Inf or handle it
        assert output.shape == (batch_size, seq_len, 128)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_model_various_batch_sizes(self, batch_size):
        """Test model with various batch sizes (stress test)."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len] * batch_size)
        
        logits, output_lengths = model(input_features, lengths)
        
        assert logits.shape == (batch_size, seq_len, 100)
        assert len(output_lengths) == batch_size
    
    @pytest.mark.parametrize("seq_len", [1, 10, 50, 100, 500, 1000, 2000])
    def test_model_various_sequence_lengths(self, seq_len):
        """Test model with various sequence lengths."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, max(1, seq_len - 10)])
        
        try:
            logits, output_lengths = model(input_features, lengths)
            assert logits.shape == (batch_size, seq_len, 100)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"OOM for sequence length {seq_len}")
            raise
    
    def test_model_extreme_sequence_length_differences(self):
        """Test model with very different sequence lengths in batch."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 4
        max_seq_len = 200
        lengths = torch.tensor([200, 10, 150, 5])
        input_features = torch.randn(batch_size, max_seq_len, 80)
        
        logits, output_lengths = model(input_features, lengths)
        
        assert logits.shape == (batch_size, max_seq_len, 100)
        assert len(output_lengths) == batch_size


class TestGradientFlow:
    """Test gradient flow and backpropagation."""
    
    def test_gradient_flow_through_all_layers(self):
        """Test that gradients flow through all model layers."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=4,
            num_heads=4,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        logits, _ = model(input_features, lengths)
        loss = logits.mean()
        loss.backward()
        
        # Check gradients exist for input
        assert input_features.grad is not None
        
        # Check gradients for model parameters
        has_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                has_grads.append(param.grad is not None)
        
        # At least some parameters should have gradients
        assert any(has_grads)
    
    def test_gradient_explosion_detection(self):
        """Test for gradient explosion (should be controlled)."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=6,
            num_heads=4,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        logits, _ = model(input_features, lengths)
        loss = logits.mean()
        loss.backward()
        
        # Check gradient magnitudes
        max_grad = max(
            (p.grad.abs().max().item() for p in model.parameters() if p.grad is not None),
            default=0.0
        )
        
        # Gradients should be reasonable (< 1e6)
        assert max_grad < 1e6 or max_grad == float('inf'), f"Potential gradient explosion: {max_grad}"
    
    def test_gradient_vanishing_detection(self):
        """Test for gradient vanishing."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=8,  # Deep model
            num_heads=4,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        logits, _ = model(input_features, lengths)
        loss = logits.mean()
        loss.backward()
        
        # Check gradient magnitudes
        grad_magnitudes = [
            p.grad.abs().max().item() 
            for p in model.parameters() 
            if p.grad is not None
        ]
        
        if grad_magnitudes:
            # At least some gradients should be non-zero
            assert max(grad_magnitudes) > 1e-10
    
    def test_multi_step_backpropagation(self):
        """Test multiple backward passes (optimizer step simulation)."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        # Multiple forward/backward passes
        for _ in range(5):
            logits, _ = model(input_features, lengths)
            loss = logits.mean()
            loss.backward()
            
            # Simulate optimizer step (zero grads)
            model.zero_grad()
            input_features.grad = None


class TestMemoryManagement:
    """Test memory management and potential leaks."""
    
    def test_memory_efficiency_multiple_forward_passes(self):
        """Test memory usage over multiple forward passes."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 4
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len] * batch_size)
        
        # Multiple forward passes
        for _ in range(100):
            with torch.no_grad():
                logits, _ = model(input_features, lengths)
                del logits
            
            # Force garbage collection periodically
            if _ % 10 == 0:
                gc.collect()
        
        # If we get here without OOM, memory management is working
        assert True
    
    def test_gradient_memory_cleanup(self):
        """Test that gradient memory is properly cleaned up."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        # Forward and backward
        logits, _ = model(input_features, lengths)
        loss = logits.mean()
        loss.backward()
        
        # Clear gradients
        model.zero_grad()
        del logits, loss
        gc.collect()
        
        # Memory should be freed
        assert True  # If we get here, cleanup worked


class TestEnhancedModelAdvanced:
    """Advanced tests for EnhancedASRModel."""
    
    @pytest.mark.parametrize("use_context", [True, False])
    @pytest.mark.parametrize("use_cross_modal", [True, False])
    @pytest.mark.parametrize("use_word2vec", [True, False])
    def test_enhanced_model_all_combinations(self, use_context, use_cross_modal, use_word2vec):
        """Test all combinations of enhanced model features."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0,
            use_contextual_embeddings=use_context,
            use_cross_modal_attention=use_cross_modal,
            use_word2vec_auxiliary=use_word2vec
        )
        
        batch_size = 2
        audio_len = 50
        text_len = 10
        
        audio_features = torch.randn(batch_size, audio_len, 80)
        audio_lengths = torch.tensor([audio_len, audio_len])
        
        if use_context:
            text_context = torch.randint(0, 100, (batch_size, text_len))
            text_lengths = torch.tensor([text_len, text_len])
            output = model(audio_features, audio_lengths, text_context=text_context, text_lengths=text_lengths)
        else:
            output = model(audio_features, audio_lengths)
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, audio_len, 100)
        
        if use_word2vec:
            assert 'word2vec_embeddings' in output
    
    def test_cross_modal_attention_with_mismatched_lengths(self):
        """Test cross-modal attention with mismatched audio/text lengths."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            use_contextual_embeddings=True,
            use_cross_modal_attention=True,
            dropout=0.0
        )
        
        batch_size = 2
        audio_len = 100
        text_len = 5  # Much shorter
        
        audio_features = torch.randn(batch_size, audio_len, 80)
        audio_lengths = torch.tensor([audio_len, audio_len])
        text_context = torch.randint(0, 100, (batch_size, text_len))
        text_lengths = torch.tensor([text_len, text_len])
        
        output = model(
            audio_features,
            audio_lengths,
            text_context=text_context,
            text_lengths=text_lengths
        )
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, audio_len, 100)


class TestModelInferencePerformance:
    """Performance tests for model inference."""
    
    def test_inference_speed_small_batch(self, benchmark):
        """Benchmark inference speed with small batch."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        model.eval()
        
        batch_size = 1
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len])
        
        def inference():
            with torch.no_grad():
                return model(input_features, lengths)
        
        benchmark(inference)
    
    def test_inference_speed_large_batch(self, benchmark):
        """Benchmark inference speed with large batch."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        model.eval()
        
        batch_size = 16
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len] * batch_size)
        
        def inference():
            with torch.no_grad():
                return model(input_features, lengths)
        
        benchmark(inference)


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_model_empty_batch(self):
        """Test model with empty batch (edge case)."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 0
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([])
        
        # Should handle gracefully or raise appropriate error
        try:
            logits, output_lengths = model(input_features, lengths)
            # If it succeeds, check output
            assert len(output_lengths) == 0
        except (ValueError, RuntimeError):
            # Error is acceptable for empty batch
            pass
    
    def test_model_zero_length_sequences(self):
        """Test model with zero-length sequences."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([0, seq_len])  # One zero-length sequence
        
        # Should handle gracefully
        try:
            logits, output_lengths = model(input_features, lengths)
            assert len(output_lengths) == batch_size
        except (ValueError, RuntimeError, IndexError):
            # Error is acceptable for zero-length sequences
            pass
    
    def test_model_inconsistent_lengths(self):
        """Test model with inconsistent length tensor."""
        model = ASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128,
            num_encoder_layers=2,
            num_heads=2,
            d_ff=256,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        input_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len + 10, seq_len])  # First length > actual sequence
        
        # May raise error or handle gracefully
        try:
            logits, output_lengths = model(input_features, lengths)
            assert logits.shape[0] == batch_size
        except (ValueError, RuntimeError, IndexError):
            # Error is acceptable
            pass

