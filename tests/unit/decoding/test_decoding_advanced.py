"""
Advanced and challenging tests for decoding algorithms.

Tests include edge cases, performance, numerical stability, and complex scenarios.
"""

import pytest
import torch
import numpy as np
import time
from decoding.beam_search import BeamSearchDecoder
from decoding.lm_decoder import LMBeamSearchDecoder


class TestBeamSearchAdvanced:
    """Advanced tests for beam search decoding."""
    
    @pytest.mark.parametrize("beam_width", [1, 2, 5, 10, 20, 50, 100, 200])
    def test_beam_search_various_widths(self, beam_width):
        """Test beam search with various beam widths."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=beam_width
        )
        
        batch_size = 2
        seq_len = 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len, seq_len - 10])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        assert all(len(hypotheses) <= beam_width for hypotheses in results if isinstance(results[0], list))
    
    def test_beam_search_very_long_sequences(self):
        """Test beam search with very long sequences."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 1000  # Very long sequence
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        start_time = time.time()
        results = decoder.decode(logits, lengths)
        decode_time = time.time() - start_time
        
        assert len(results) == batch_size
        # Should complete within reasonable time (< 30 seconds)
        assert decode_time < 30.0
    
    def test_beam_search_all_blanks(self):
        """Test beam search when all predictions are blank tokens."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 2
        seq_len = 50
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 10.0  # High probability for blank
        
        lengths = torch.tensor([seq_len, seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        # Results should be valid even if mostly blank
    
    def test_beam_search_uniform_distribution(self):
        """Test beam search with uniform probability distribution."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        # Uniform logits (no preference)
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
    
    def test_beam_search_extreme_probabilities(self):
        """Test beam search with extreme probability values."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        
        # Very high probabilities for one token
        logits = torch.full((batch_size, seq_len, vocab_size), -100.0)
        logits[:, :, 1] = 100.0  # Very high probability for token 1
        
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        # Should handle extreme values without numerical issues
    
    def test_beam_search_numerical_stability(self):
        """Test numerical stability with various logit ranges."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        
        test_cases = [
            torch.randn(batch_size, seq_len, vocab_size),  # Normal range
            torch.randn(batch_size, seq_len, vocab_size) * 100,  # Large values
            torch.randn(batch_size, seq_len, vocab_size) * 1e-6,  # Small values
        ]
        
        for logits in test_cases:
            lengths = torch.tensor([seq_len])
            results = decoder.decode(logits, lengths)
            
            assert len(results) == batch_size
            # Should not produce NaN or Inf
            if isinstance(results[0], dict) and 'text' in results[0]:
                assert results[0]['text'] is not None
    
    def test_beam_search_empty_vocabulary(self):
        """Test beam search edge case with minimal vocabulary."""
        vocab_size = 2  # Minimal vocabulary
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=2
        )
        
        batch_size = 1
        seq_len = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
    
    def test_beam_search_concurrent_decoding(self):
        """Test concurrent beam search decoding."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        def decode_batch(logits, lengths):
            return decoder.decode(logits, lengths)
        
        batch_size = 1
        seq_len = 50
        logits_batch = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(10)]
        lengths_batch = [torch.tensor([seq_len]) for _ in range(10)]
        
        # Decode concurrently (if decoder is thread-safe)
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(decode_batch, logits_batch, lengths_batch))
        
        assert len(results) == 10
        assert all(len(r) == batch_size for r in results)


class TestLMDecoderAdvanced:
    """Advanced tests for LM decoder."""
    
    def test_lm_decoder_very_large_vocabulary(self):
        """Test LM decoder with very large vocabulary."""
        vocab_size = 10000  # Large vocabulary
        vocab = [f"token_{i}" for i in range(vocab_size)]
        vocab[0] = "<blank>"
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=10
        )
        
        batch_size = 1
        seq_len = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        # May be slow with large vocab, but should work
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
    
    def test_lm_decoder_empty_text_output(self):
        """Test LM decoder when output is empty."""
        vocab = ["<blank>", "<pad>", "a", "b", "c"]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        # All blanks - should produce empty or minimal output
        logits = torch.zeros(batch_size, seq_len, len(vocab))
        logits[:, :, 0] = 10.0  # All blanks
        
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
        # Text may be empty, which is acceptable
    
    def test_lm_decoder_performance_large_batch(self):
        """Test LM decoder performance with large batch."""
        vocab = ["<blank>", "<pad>"] + [f"token_{i}" for i in range(100)]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 32  # Large batch
        seq_len = 100
        logits = torch.randn(batch_size, seq_len, len(vocab))
        lengths = torch.tensor([seq_len] * batch_size)
        
        start_time = time.time()
        results = decoder.decode(logits, lengths)
        decode_time = time.time() - start_time
        
        assert len(results) == batch_size
        # Should complete within reasonable time
        assert decode_time < 60.0  # 1 minute max
    
    @pytest.mark.parametrize("beam_width", [1, 5, 10, 20, 50, 100])
    def test_lm_decoder_beam_width_scaling(self, beam_width):
        """Test how beam width affects decoding time."""
        vocab = ["<blank>", "<pad>"] + [f"token_{i}" for i in range(50)]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=beam_width
        )
        
        batch_size = 1
        seq_len = 50
        logits = torch.randn(batch_size, seq_len, len(vocab))
        lengths = torch.tensor([seq_len])
        
        start_time = time.time()
        results = decoder.decode(logits, lengths, beam_size=beam_width)
        decode_time = time.time() - start_time
        
        assert len(results) == batch_size
        # Decode time should increase with beam width (but not linearly)
        # Just check it completes


class TestDecodingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_decode_empty_sequence(self):
        """Test decoding empty sequence."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 1  # Minimal sequence
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        results = decoder.decode(logits, lengths)
        
        assert len(results) == batch_size
    
    def test_decode_single_token_vocabulary(self):
        """Test decoding with single token vocabulary (edge case)."""
        vocab_size = 1
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=1
        )
        
        batch_size = 1
        seq_len = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        # Should handle gracefully
        try:
            results = decoder.decode(logits, lengths)
            assert len(results) == batch_size
        except (ValueError, IndexError):
            # Error is acceptable for edge case
            pass
    
    def test_decode_mismatched_lengths(self):
        """Test decoding with mismatched sequence lengths."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 2
        max_seq_len = 50
        logits = torch.randn(batch_size, max_seq_len, vocab_size)
        lengths = torch.tensor([max_seq_len + 10, max_seq_len - 10])  # Mismatched
        
        # Should handle gracefully
        try:
            results = decoder.decode(logits, lengths)
            assert len(results) == batch_size
        except (ValueError, IndexError, RuntimeError):
            # Error is acceptable
            pass
    
    def test_decode_nan_logits(self):
        """Test decoding with NaN logits (error handling)."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        logits = torch.full((batch_size, seq_len, vocab_size), float('nan'))
        lengths = torch.tensor([seq_len])
        
        # May propagate NaN or handle it
        try:
            results = decoder.decode(logits, lengths)
            # If succeeds, just check structure
            assert len(results) == batch_size
        except (ValueError, RuntimeError):
            # Error is acceptable for NaN input
            pass
    
    def test_decode_inf_logits(self):
        """Test decoding with Inf logits."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=5
        )
        
        batch_size = 1
        seq_len = 50
        logits = torch.full((batch_size, seq_len, vocab_size), float('inf'))
        lengths = torch.tensor([seq_len])
        
        try:
            results = decoder.decode(logits, lengths)
            assert len(results) == batch_size
        except (ValueError, RuntimeError):
            # Error is acceptable
            pass


@pytest.mark.performance
class TestDecodingPerformance:
    """Performance tests for decoding."""
    
    def test_beam_search_performance(self, benchmark):
        """Benchmark beam search decoding."""
        vocab_size = 100
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=0,
            beam_width=10
        )
        
        batch_size = 1
        seq_len = 200
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len])
        
        def decode():
            return decoder.decode(logits, lengths)
        
        benchmark(decode)
    
    def test_lm_decoder_performance(self, benchmark):
        """Benchmark LM decoder."""
        vocab = ["<blank>", "<pad>"] + [f"token_{i}" for i in range(100)]
        
        decoder = LMBeamSearchDecoder(
            vocab=vocab,
            blank_token_id=0,
            beam_width=10
        )
        
        batch_size = 1
        seq_len = 200
        logits = torch.randn(batch_size, seq_len, len(vocab))
        lengths = torch.tensor([seq_len])
        
        def decode():
            return decoder.decode(logits, lengths)
        
        benchmark(decode)

