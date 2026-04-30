"""
Tests for dataset loading and transforms
"""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.train import PoliticianDataset, get_transforms, CLASS_NAMES


class TestDataset:
    """Test dataset functionality"""
    
    def test_class_names_count(self):
        """Test that we have exactly 16 classes"""
        assert len(CLASS_NAMES) == 16
    
    def test_class_names_sorted(self):
        """Test that class names are sorted"""
        assert CLASS_NAMES == sorted(CLASS_NAMES)
    
    def test_class_names_unique(self):
        """Test that class names are unique"""
        assert len(CLASS_NAMES) == len(set(CLASS_NAMES))
    
    def test_train_transform_output_shape(self):
        """Test train transform output shape"""
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        transform = get_transforms('train')
        output = transform(dummy_img)
        
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32
    
    def test_val_transform_output_shape(self):
        """Test validation transform output shape"""
        from PIL import Image
        import numpy as np
        
        dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        transform = get_transforms('val')
        output = transform(dummy_img)
        
        assert output.shape == (3, 224, 224)
    
    def test_test_transform_output_shape(self):
        """Test test transform output shape"""
        from PIL import Image
        import numpy as np
        
        dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        transform = get_transforms('test')
        output = transform(dummy_img)
        
        assert output.shape == (3, 224, 224)
    
    def test_transform_normalization_range(self):
        """Test that transforms normalize values properly"""
        from PIL import Image
        import numpy as np
        
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        transform = get_transforms('test')
        output = transform(dummy_img)
        
        # After normalization, values should be roughly in [-3, 3] range
        assert output.min() >= -5.0
        assert output.max() <= 5.0
    
    @pytest.mark.skipif(not Path('dataset/train').exists(), reason="Dataset not available")
    def test_dataset_loading(self):
        """Test dataset can be loaded"""
        dataset = PoliticianDataset('dataset/train', transform=get_transforms('train'))
        
        assert len(dataset) > 0
    
    @pytest.mark.skipif(not Path('dataset/train').exists(), reason="Dataset not available")
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format"""
        dataset = PoliticianDataset('dataset/train', transform=get_transforms('train'))
        
        if len(dataset) > 0:
            image, label, path = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 224, 224)
            assert isinstance(label, int)
            assert 0 <= label < 16
            assert isinstance(path, str)
