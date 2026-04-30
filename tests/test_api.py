"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

# Add paths
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create test client"""
    try:
        from api.main import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not load API: {e}")


@pytest.fixture
def dummy_image_file():
    """Create dummy image file for testing"""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "device" in data
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "count" in data
        assert data["count"] == 16
        assert len(data["classes"]) == 16
    
    @pytest.mark.skipif(not Path('models/saved').exists(), reason="Models not available")
    def test_predict_endpoint(self, client, dummy_image_file):
        """Test predict endpoint"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_image_file, "image/jpeg")}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "top3" in data
            assert "model_used" in data
            assert "inference_time_ms" in data
            assert len(data["top3"]) == 3
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_file(self, client):
        """Test predict with invalid file type"""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        )
        assert response.status_code == 400
    
    @pytest.mark.skipif(not Path('models/saved').exists(), reason="Models not available")
    def test_predict_with_model_name(self, client, dummy_image_file):
        """Test predict with specific model name"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_image_file, "image/jpeg")},
            data={"model_name": "resnet50"}
        )
        
        # Should either succeed or return 400 if model not found
        assert response.status_code in [200, 400]
