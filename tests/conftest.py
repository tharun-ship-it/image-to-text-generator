"""
Test Fixtures and Configuration
===============================

Pytest fixtures for BLIP Image Captioning tests.

Author: Tharun Ponnam
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image():
    """Create sample image tensor (BLIP format: 384x384)."""
    return torch.randn(1, 3, 384, 384)


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    return {
        'pixel_values': torch.randn(4, 3, 384, 384),
        'input_ids': torch.randint(0, 30000, (4, 20)),
        'attention_mask': torch.ones(4, 20, dtype=torch.long)
    }


@pytest.fixture
def sample_visual_features():
    """Create sample visual features (BLIP-Large output)."""
    # BLIP-Large: 577 tokens (576 patches + 1 CLS), 1024 dim
    return torch.randn(2, 577, 1024)


@pytest.fixture
def sample_text_features():
    """Create sample text features."""
    return torch.randn(2, 20, 768)


@pytest.fixture(scope="session")
def blip_processor():
    """Load BLIP processor (session-scoped for efficiency)."""
    try:
        from transformers import BlipProcessor
        return BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    except Exception:
        pytest.skip("BLIP processor not available")
