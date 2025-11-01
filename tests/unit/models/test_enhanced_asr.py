"""
Unit tests for enhanced ASR model.

Tests for EnhancedASRModel, ContextualEmbedding, and CrossModalAttention.
"""

import pytest
import torch
import numpy as np
from models.enhanced_asr import (
    EnhancedASRModel,
    ContextualEmbedding,
    CrossModalAttention
)


class TestContextualEmbedding:
    """Test ContextualEmbedding class."""
    
    def test_initialization(self):
        """Test contextual embedding initialization."""
        embedding = ContextualEmbedding(
            vocab_size=100,
            embedding_dim=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        assert embedding is not None
    
    def test_forward_pass(self):
        """Test contextual embedding forward pass."""
        embedding = ContextualEmbedding(
            vocab_size=100,
            embedding_dim=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 20
        token_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        output = embedding(token_ids)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert isinstance(output, torch.Tensor)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        embedding = ContextualEmbedding(
            vocab_size=100,
            embedding_dim=128,
            d_model=256,
            num_layers=2,
            num_heads=4,
            dropout=0.0
        )
        
        batch_size = 2
        token_ids = torch.randint(0, 100, (batch_size, 50))
        
        output = embedding(token_ids)
        
        assert output.shape == (batch_size, 50, 256)


class TestCrossModalAttention:
    """Test CrossModalAttention class."""
    
    def test_initialization(self):
        """Test cross-modal attention initialization."""
        attention = CrossModalAttention(
            d_model=256,
            num_heads=4,
            dropout=0.1
        )
        assert attention.d_model == 256
        assert attention.num_heads == 4
    
    def test_forward_pass(self):
        """Test cross-modal attention forward pass."""
        attention = CrossModalAttention(
            d_model=256,
            num_heads=4,
            dropout=0.0
        )
        
        batch_size = 2
        audio_len = 100
        text_len = 20
        audio_features = torch.randn(batch_size, audio_len, 256)
        text_features = torch.randn(batch_size, text_len, 256)
        
        attended_audio, attended_text = attention(audio_features, text_features)
        
        assert attended_audio.shape == (batch_size, audio_len, 256)
        assert attended_text.shape == (batch_size, text_len, 256)
    
    def test_forward_with_masks(self):
        """Test cross-modal attention with masks."""
        attention = CrossModalAttention(
            d_model=128,
            num_heads=2,
            dropout=0.0
        )
        
        batch_size = 2
        audio_len = 50
        text_len = 10
        audio_features = torch.randn(batch_size, audio_len, 128)
        text_features = torch.randn(batch_size, text_len, 128)
        
        # Create masks (1 = valid, 0 = padding)
        audio_mask = torch.ones(batch_size, audio_len, dtype=torch.bool)
        text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
        
        attended_audio, attended_text = attention(
            audio_features, text_features,
            audio_mask=audio_mask,
            text_mask=text_mask
        )
        
        assert attended_audio.shape == audio_features.shape
        assert attended_text.shape == text_features.shape


class TestEnhancedASRModel:
    """Test EnhancedASRModel class."""
    
    def test_initialization_default(self):
        """Test enhanced model initialization with defaults."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256
        )
        assert model.input_dim == 80
        assert model.vocab_size == 100
        assert model.d_model == 256
    
    def test_initialization_with_contextual(self):
        """Test initialization with contextual embeddings."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            use_contextual_embeddings=True,
            use_cross_modal_attention=True
        )
        assert model.use_contextual_embeddings is True
        assert model.use_cross_modal_attention is True
        assert model.contextual_embedding is not None
        assert model.cross_modal_attention is not None
    
    def test_initialization_without_contextual(self):
        """Test initialization without contextual features."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            use_contextual_embeddings=False,
            use_cross_modal_attention=False
        )
        assert model.contextual_embedding is None
        assert model.cross_modal_attention is None
    
    def test_forward_pass_audio_only(self):
        """Test forward pass with audio only."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            use_contextual_embeddings=False
        )
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len - 20])
        
        output = model(audio_features, lengths)
        
        assert 'logits' in output
        assert 'output_lengths' in output
        assert output['logits'].shape == (batch_size, seq_len, 100)
    
    def test_forward_pass_with_context(self):
        """Test forward pass with text context."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            use_contextual_embeddings=True,
            use_cross_modal_attention=True
        )
        
        batch_size = 2
        audio_len = 100
        text_len = 20
        audio_features = torch.randn(batch_size, audio_len, 80)
        audio_lengths = torch.tensor([audio_len, audio_len - 20])
        text_context = torch.randint(0, 100, (batch_size, text_len))
        text_lengths = torch.tensor([text_len, text_len - 5])
        
        output = model(
            audio_features,
            audio_lengths,
            text_context=text_context,
            text_lengths=text_lengths
        )
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, audio_len, 100)
    
    def test_forward_pass_with_word2vec(self):
        """Test forward pass with Word2Vec auxiliary task."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256,
            use_word2vec_auxiliary=True,
            word2vec_dim=128
        )
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len])
        
        output = model(audio_features, lengths)
        
        assert 'word2vec_embeddings' in output
        assert output['word2vec_embeddings'].shape == (batch_size, seq_len, 128)
    
    def test_parameter_counting(self):
        """Test parameter counting methods."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=256
        )
        
        total_params = model.get_num_params()
        trainable_params = model.get_num_trainable_params()
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_different_configurations(self):
        """Test different model configurations."""
        configs = [
            {'use_contextual_embeddings': False, 'use_cross_modal_attention': False},
            {'use_contextual_embeddings': True, 'use_cross_modal_attention': False},
            {'use_contextual_embeddings': True, 'use_cross_modal_attention': True},
            {'use_contextual_embeddings': True, 'use_cross_modal_attention': True, 'use_word2vec_auxiliary': True},
        ]
        
        for config in configs:
            model = EnhancedASRModel(
                input_dim=80,
                vocab_size=100,
                d_model=128,  # Smaller for faster tests
                **config
            )
            
            batch_size = 1
            seq_len = 50
            audio_features = torch.randn(batch_size, seq_len, 80)
            lengths = torch.tensor([seq_len])
            
            output = model(audio_features, lengths)
            assert 'logits' in output
            assert output['logits'].shape[2] == 100  # vocab_size
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = EnhancedASRModel(
            input_dim=80,
            vocab_size=100,
            d_model=128
        )
        
        batch_size = 2
        seq_len = 50
        audio_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        lengths = torch.tensor([seq_len, seq_len])
        
        output = model(audio_features, lengths)
        loss = output['logits'].mean()
        loss.backward()
        
        # Check gradients
        assert audio_features.grad is not None
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

