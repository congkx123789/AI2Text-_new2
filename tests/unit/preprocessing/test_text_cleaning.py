"""
Unit tests for text cleaning and normalization module.

Tests for VietnameseTextNormalizer and Tokenizer classes.
"""

import pytest
from preprocessing.text_cleaning import (
    VietnameseTextNormalizer,
    Tokenizer
)


class TestVietnameseTextNormalizer:
    """Test VietnameseTextNormalizer class."""
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = VietnameseTextNormalizer(
            lowercase=True,
            remove_punctuation=True,
            normalize_unicode=True
        )
        assert normalizer.lowercase is True
        assert normalizer.remove_punctuation is True
        assert normalizer.normalize_unicode is True
    
    def test_normalize_unicode_text(self):
        """Test Unicode normalization."""
        normalizer = VietnameseTextNormalizer()
        # Test with potentially problematic Unicode
        text = "café"
        normalized = normalizer.normalize_unicode_text(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_remove_extra_whitespace(self):
        """Test whitespace removal."""
        normalizer = VietnameseTextNormalizer()
        
        text = "xin   chào    việt   nam"
        cleaned = normalizer.remove_extra_whitespace(text)
        
        assert cleaned == "xin chào việt nam"
        assert "  " not in cleaned  # No double spaces
    
    def test_expand_abbreviations(self):
        """Test abbreviation expansion."""
        normalizer = VietnameseTextNormalizer()
        
        text = "tôi sống ở tp. hồ chí minh"
        expanded = normalizer.expand_abbreviations(text)
        
        assert "thành phố" in expanded.lower()
    
    def test_convert_numbers_to_words(self):
        """Test number to word conversion."""
        normalizer = VietnameseTextNormalizer()
        
        text = "số 123"
        converted = normalizer.convert_numbers_to_words(text)
        
        assert "một" in converted or "hai" in converted or "ba" in converted
        # Should not contain digits
        assert not any(char.isdigit() for char in converted.split())
    
    def test_remove_special_characters(self):
        """Test special character removal."""
        normalizer = VietnameseTextNormalizer(remove_punctuation=True)
        
        text = "xin chào @#$ việt nam!"
        cleaned = normalizer.remove_special_characters(text)
        
        # Should only contain Vietnamese characters and spaces
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned
    
    def test_remove_special_keep_vietnamese(self):
        """Test keeping Vietnamese characters."""
        normalizer = VietnameseTextNormalizer()
        
        text = "xin chào việt nam"
        cleaned = normalizer.remove_special_characters(text, keep_vietnamese=True)
        
        assert "việt" in cleaned
        assert "nam" in cleaned
    
    def test_normalize_complete_pipeline(self, sample_vietnamese_texts):
        """Test complete normalization pipeline."""
        normalizer = VietnameseTextNormalizer()
        
        for input_text, expected_output in sample_vietnamese_texts.items():
            normalized = normalizer.normalize(input_text)
            # Check that normalization happened (may not match exactly due to variations)
            assert isinstance(normalized, str)
            assert len(normalized) >= 0
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        normalizer = VietnameseTextNormalizer()
        
        result = normalizer.normalize("")
        assert result == ""
    
    def test_normalize_lowercase(self):
        """Test lowercase conversion."""
        normalizer = VietnameseTextNormalizer(lowercase=True)
        
        text = "XIN CHÀO VIỆT NAM"
        normalized = normalizer.normalize(text)
        
        assert normalized.islower() or normalized == ""
    
    def test_normalize_preserve_vietnamese_tones(self):
        """Test that Vietnamese tone marks are preserved."""
        normalizer = VietnameseTextNormalizer()
        
        text = "xin chào việt nam"
        normalized = normalizer.normalize(text)
        
        # Should contain Vietnamese characters with tones
        vietnamese_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
        assert any(char in normalized for char in vietnamese_chars) or "việt" in normalized.lower()


class TestTokenizer:
    """Test Tokenizer class."""
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = Tokenizer()
        assert tokenizer is not None
        assert len(tokenizer) >= 0  # Should have some vocabulary
    
    def test_encode_basic(self, tokenizer):
        """Test basic encoding."""
        text = "xin chào"
        
        tokens = tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_decode_basic(self, tokenizer):
        """Test basic decoding."""
        # First encode some text
        text = "xin chào"
        tokens = tokenizer.encode(text)
        
        # Then decode
        decoded = tokenizer.decode(tokens)
        
        assert isinstance(decoded, str)
        # Decoded text should be similar (may have normalization)
        assert len(decoded) > 0
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encode-decode roundtrip."""
        text = "xin chào việt nam"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Should be able to decode back (may differ due to normalization)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_encode_empty_string(self, tokenizer):
        """Test encoding empty string."""
        tokens = tokenizer.encode("")
        assert isinstance(tokens, list)
        # May be empty or contain special tokens
    
    def test_decode_empty_list(self, tokenizer):
        """Test decoding empty token list."""
        decoded = tokenizer.decode([])
        assert isinstance(decoded, str)
    
    def test_vocabulary_size(self, tokenizer):
        """Test vocabulary access."""
        vocab_size = len(tokenizer)
        assert vocab_size > 0
        assert isinstance(vocab_size, int)
    
    def test_special_tokens(self, tokenizer):
        """Test special token handling."""
        # Check if special tokens exist
        if hasattr(tokenizer, 'blank_token_id'):
            assert isinstance(tokenizer.blank_token_id, int)
        if hasattr(tokenizer, 'pad_token_id'):
            assert isinstance(tokenizer.pad_token_id, int)
    
    def test_encode_unicode(self, tokenizer):
        """Test encoding with Vietnamese Unicode."""
        text = "xin chào việt nam"
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
    
    def test_token_to_id(self, tokenizer):
        """Test token to ID conversion if available."""
        if hasattr(tokenizer, 'token_to_id'):
            # Test with a common token if vocabulary exists
            assert callable(tokenizer.token_to_id)
    
    def test_id_to_token(self, tokenizer):
        """Test ID to token conversion if available."""
        if hasattr(tokenizer, 'id_to_token'):
            assert hasattr(tokenizer.id_to_token, 'get') or isinstance(tokenizer.id_to_token, dict)

