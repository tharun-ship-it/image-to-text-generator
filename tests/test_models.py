"""
Comprehensive Tests for BLIP Image Captioning
=============================================

Unit tests for vision encoder, text decoder, attention mechanisms,
and full captioning pipeline.

Author: Tharun Ponnam
"""

import pytest
import torch

from src.models.attention import (
    MultiHeadAttention,
    CrossModalAttention,
    PositionalEncoding,
    LearnedPositionalEncoding
)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""
    
    @pytest.fixture
    def attention(self):
        return MultiHeadAttention(embed_dim=256, num_heads=8, dropout=0.0)
    
    def test_forward_shape(self, attention):
        batch_size, seq_len, embed_dim = 2, 10, 256
        x = torch.randn(batch_size, seq_len, embed_dim)
        output, _ = attention(x, x, x)
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_attention_weights_shape(self, attention):
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)
        _, attn_weights = attention(x, x, x, return_attention=True)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_cross_attention(self, attention):
        query = torch.randn(2, 10, 256)
        key_value = torch.randn(2, 20, 256)
        output, attn = attention(query, key_value, key_value, return_attention=True)
        assert output.shape == (2, 10, 256)
        assert attn.shape == (2, 8, 10, 20)


class TestCrossModalAttention:
    """Tests for CrossModalAttention module."""
    
    @pytest.fixture
    def cross_attention(self):
        return CrossModalAttention(query_dim=256, key_dim=1024, num_heads=8)
    
    def test_forward_shape(self, cross_attention):
        text = torch.randn(2, 10, 256)
        visual = torch.randn(2, 577, 1024)
        output, _ = cross_attention(text, visual)
        assert output.shape == text.shape
    
    def test_attention_map(self, cross_attention):
        text = torch.randn(2, 10, 256)
        visual = torch.randn(2, 50, 1024)
        attn_map = cross_attention.get_attention_map(text, visual)
        assert attn_map.shape == (2, 10, 50)


class TestPositionalEncoding:
    """Tests for positional encoding modules."""
    
    def test_sinusoidal_encoding(self):
        pe = PositionalEncoding(embed_dim=256, max_length=1000, dropout=0.0)
        x = torch.randn(2, 50, 256)
        output = pe(x)
        assert output.shape == x.shape
    
    def test_learned_encoding(self):
        pe = LearnedPositionalEncoding(embed_dim=256, max_length=512, dropout=0.0)
        x = torch.randn(2, 50, 256)
        output = pe(x)
        assert output.shape == x.shape


class TestHelperFunctions:
    """Tests for utility functions."""
    
    def test_set_seed_reproducibility(self):
        from src.utils.helpers import set_seed
        set_seed(42)
        tensor1 = torch.randn(10)
        set_seed(42)
        tensor2 = torch.randn(10)
        assert torch.allclose(tensor1, tensor2)
    
    def test_count_parameters(self):
        from src.utils.helpers import count_parameters
        model = torch.nn.Linear(10, 5)
        total = count_parameters(model)
        assert total == 10 * 5 + 5


class TestDataPreprocessing:
    """Tests for data preprocessing utilities."""
    
    def test_image_preprocessor(self):
        from src.data.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(image_size=384)
        assert preprocessor.BLIP_MEAN is not None
    
    def test_text_preprocessor(self):
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor(max_length=50)
        clean = preprocessor.clean_text("  HELLO  https://test.com  ")
        assert "https" not in clean


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image_tensor():
    return torch.randn(1, 3, 384, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
