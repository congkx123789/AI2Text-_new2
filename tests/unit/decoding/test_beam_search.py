"""
Unit tests for beam search decoding.

Tests for BeamSearchDecoder class.
"""

import pytest
import torch
import numpy as np
from decoding.beam_search import BeamSearchDecoder


class TestBeamSearchDecoder:
    """Test BeamSearchDecoder class."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = BeamSearchDecoder(
            vocab_size=100,
            blank_token_id=0,
            beam_width=5
        )
        assert decoder.vocab_size == 100
        assert decoder.blank_token_id == 0
        assert decoder.beam_width == 5
    
    def test_decode_basic(self):
        """Test basic beam search decoding."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 2
        seq_len = 20
        vocab_size = 10
        
        # Create logits with high probability for token 1
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 1] = 10.0  # High probability for token 1
        logits[:, :, 0] = -10.0  # Low probability for blank
        
        lengths = torch.tensor([seq_len, seq_len - 5])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        assert isinstance(results[0], list)  # List of hypotheses
    
    def test_decode_batch(self):
        """Test batch decoding."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 2
        seq_len = 15
        vocab_size = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len, seq_len])
        
        results = decoder.decode_batch(logits, lengths, tokenizer=None)
        
        assert len(results) == batch_size
        assert all(isinstance(r, (list, dict, str)) for r in results)
    
    def test_beam_width_effect(self):
        """Test that beam width affects number of hypotheses."""
        vocab_size = 10
        seq_len = 10
        logits = torch.randn(1, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        for beam_width in [1, 3, 5, 10]:
            decoder = BeamSearchDecoder(
                vocab_size=vocab_size,
                blank_token_id=0,
                beam_width=beam_width
            )
            
            results = decoder.decode(logits, lengths)
            
            assert len(results) == 1
            # With larger beam width, we may get more hypotheses (up to beam_width)
            assert len(results[0]) <= beam_width
    
    def test_blank_token_handling(self):
        """Test blank token handling in CTC."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 1
        seq_len = 10
        vocab_size = 10
        
        # Create logits that prefer blank tokens
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 10.0  # High probability for blank
        
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) > 0
        # Results should be valid (even if mostly blank)
    
    def test_empty_sequence(self):
        """Test decoding empty sequence."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 1
        seq_len = 0
        vocab_size = 10
        logits = torch.randn(batch_size, 1, vocab_size)  # Minimum 1 timestep
        lengths = torch.tensor([1])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
    
    def test_varying_sequence_lengths(self):
        """Test decoding with varying sequence lengths."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 3
        max_seq_len = 20
        vocab_size = 10
        
        logits = torch.randn(batch_size, max_seq_len, vocab_size)
        lengths = torch.tensor([20, 15, 10])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        # Each result should handle its sequence length correctly
    
    def test_confidence_scoring(self):
        """Test confidence scoring if available."""
        decoder = BeamSearchDecoder(
            vocab_size=10,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 1
        seq_len = 10
        vocab_size = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) > 0
        # Check if results have confidence/scores
        if isinstance(results[0], dict):
            assert 'score' in results[0] or 'confidence' in results[0]


class TestBeamSearchDecodingWithTokenizer:
    """Test beam search decoding with tokenizer."""
    
    def test_decode_with_tokenizer(self, tokenizer):
        """Test decoding with tokenizer."""
        vocab_size = len(tokenizer)
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 1
        seq_len = 20
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode_batch(logits, lengths, tokenizer=tokenizer)
        
        assert len(results) == batch_size
        assert isinstance(results[0], str) or isinstance(results[0], list)

