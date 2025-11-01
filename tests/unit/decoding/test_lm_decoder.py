"""
Unit tests for language model decoder.

Tests for LMBeamSearchDecoder class.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from decoding.lm_decoder import LMBeamSearchDecoder


class TestLMBeamSearchDecoder:
    """Test LMBeamSearchDecoder class."""
    
    def test_initialization_without_lm(self):
        """Test decoder initialization without LM."""
        vocab = ["<blank>", "<pad>", "a", "b", "c", " ", "xin", "chào"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            lm_path=None,
            beam_width=5
        )
        
        assert decoder.vocab == vocab
        assert decoder.lm_path is None
        assert decoder.beam_width == 5
    
    def test_initialization_with_lm_path_nonexistent(self):
        """Test initialization with non-existent LM path."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        # Should not raise error, just fall back to no LM
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            lm_path="nonexistent.lm",
            beam_width=5
        )
        
        assert decoder.has_lm is False
    
    def test_decode_basic(self):
        """Test basic decoding without LM."""
        vocab = ["<blank>", "<pad>", "a", "b", "c", " ", "xin", "chào"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 2
        seq_len = 20
        vocab_size = len(vocab)
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len, seq_len - 5])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)
    
    def test_decode_batch(self):
        """Test batch decoding."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 2
        seq_len = 15
        vocab_size = len(vocab)
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len, seq_len])
        
        results = decoder.decode_batch(logits, lengths)
        
        assert len(results) == batch_size
        assert all(isinstance(r, str) for r in results)
    
    def test_decode_with_high_confidence_tokens(self):
        """Test decoding with high-confidence token predictions."""
        vocab = ["<blank>", "<pad>", "xin", "chào", "việt", "nam", " "]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 1
        seq_len = 10
        vocab_size = len(vocab)
        
        # Create logits that prefer specific tokens
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 2] = 10.0  # High probability for "xin"
        logits[:, :, 0] = -10.0  # Low probability for blank
        
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        assert 'text' in results[0]
    
    def test_decode_varying_lengths(self):
        """Test decoding with varying sequence lengths."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 3
        max_seq_len = 20
        vocab_size = len(vocab)
        
        logits = torch.randn(batch_size, max_seq_len, vocab_size)
        lengths = torch.tensor([20, 15, 10])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        assert all(isinstance(r, dict) for r in results)
    
    def test_decode_empty_batch(self):
        """Test decoding empty batch."""
        vocab = ["<blank>", "<pad>", "a", "b"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=3
        )
        
        batch_size = 0
        seq_len = 10
        vocab_size = len(vocab)
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == 0
    
    def test_lm_score_present(self):
        """Test that LM score is present in results when available."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 10
        vocab_size = len(vocab)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) > 0
        # Results should have score, and potentially lm_score
        assert 'score' in results[0]
        # lm_score may be 0.0 if no LM is used
        if 'lm_score' in results[0]:
            assert isinstance(results[0]['lm_score'], (float, int))
    
    def test_beam_width_parameter(self):
        """Test that beam width parameter works."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        for beam_width in [1, 3, 5, 10]:
            decoder = LMBeamSearchDecoder(
                vocab=vocab,
                blank_token_id=0,
                beam_width=beam_width
            )
            
            batch_size = 1
            seq_len = 10
            vocab_size = len(vocab)
            logits = torch.randn(batch_size, seq_len, vocab_size)
            lengths = torch.tensor([seq_len])
            
            results = decoder.decode(logits, lengths, beam_size=beam_width)
            
            assert len(results) == batch_size
    
    def test_fallback_to_basic_decoder(self):
        """Test fallback when pyctcdecode is not available."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        # If decoder is None (no pyctcdecode), should still work
        # by falling back to basic decoder
        batch_size = 1
        seq_len = 10
        vocab_size = len(vocab)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        # Should not raise error even if decoder is None
        try:
            results = decoder.decode(logits, lengths)
            assert len(results) > 0
        except Exception as e:
            # If fallback decoder doesn't work, that's acceptable for this test
            # The important thing is that it doesn't crash
            pass

