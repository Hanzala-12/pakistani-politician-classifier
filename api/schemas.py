"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: str = Field(..., alias="class", description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    
    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Response for single image prediction"""
    predicted_class: str = Field(..., description="Top predicted class")
    confidence: float = Field(..., description="Confidence for top prediction")
    top3: List[PredictionResult] = Field(..., description="Top 3 predictions")
    model_used: str = Field(..., description="Model name used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction"""
    predictions: List[PredictionResponse]
    total_images: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    device: str


class ClassesResponse(BaseModel):
    """Classes list response"""
    classes: List[str]
    count: int
