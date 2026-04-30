"""
Tests for model architecture and output shapes
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.train import get_model


class TestModelArchitecture:
    """Test model architectures"""
    
    def test_resnet50_output_shape(self, dummy_image, num_classes, device):
        """Test ResNet50 output shape"""
        model = get_model("resnet50", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
        
        assert output.shape == (dummy_image.size(0), num_classes)
        assert not torch.isnan(output).any()
    
    def test_resnet152_output_shape(self, dummy_image, num_classes, device):
        """Test ResNet152 output shape"""
        model = get_model("resnet152", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
        
        assert output.shape == (dummy_image.size(0), num_classes)
    
    def test_efficientnet_b3_output_shape(self, dummy_image, num_classes, device):
        """Test EfficientNet-B3 output shape"""
        model = get_model("efficientnet_b3", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
        
        assert output.shape == (dummy_image.size(0), num_classes)
    
    def test_vgg16_output_shape(self, dummy_image, num_classes, device):
        """Test VGG16 output shape"""
        model = get_model("vgg16", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
        
        assert output.shape == (dummy_image.size(0), num_classes)
    
    def test_convnext_base_output_shape(self, dummy_image, num_classes, device):
        """Test ConvNeXt Base output shape"""
        model = get_model("convnext_base", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
        
        assert output.shape == (dummy_image.size(0), num_classes)
    
    def test_invalid_model_name(self, num_classes):
        """Test invalid model name raises error"""
        with pytest.raises(ValueError):
            get_model("invalid_model", num_classes=num_classes)
    
    def test_model_parameters_trainable(self, num_classes):
        """Test that model has trainable parameters"""
        model = get_model("resnet50", num_classes=num_classes)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
    
    def test_softmax_output_sums_to_one(self, dummy_image, num_classes, device):
        """Test that softmax probabilities sum to 1"""
        model = get_model("resnet50", num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_image.to(device))
            probs = torch.softmax(output, dim=1)
        
        # Check each sample's probabilities sum to ~1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
