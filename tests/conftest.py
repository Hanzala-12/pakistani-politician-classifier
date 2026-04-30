"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
from pathlib import Path


@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def num_classes():
    """Number of politician classes"""
    return 16


@pytest.fixture
def batch_size():
    """Default batch size for testing"""
    return 4


@pytest.fixture
def image_size():
    """Default image size"""
    return (224, 224)


@pytest.fixture
def dummy_image(batch_size):
    """Create dummy image tensor"""
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def model_dir():
    """Model directory path"""
    return Path('models/saved')


@pytest.fixture
def dataset_dir():
    """Dataset directory path"""
    return Path('dataset')
