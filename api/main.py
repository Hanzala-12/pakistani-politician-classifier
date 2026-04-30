"""
FastAPI Application for Pakistani Politician Image Classification
"""

import torch
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from typing import Optional, List
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.train import get_transforms, CLASS_NAMES
from api.model_loader import model_loader
from api.schemas import (
    PredictionResponse, PredictionResult,
    BatchPredictionResponse, HealthResponse, ClassesResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Pakistani Politician Image Classifier API",
    description="Deep Learning API for classifying Pakistani politician images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get transform
transform = get_transforms('test')


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Pakistani Politician Image Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models_loaded=model_loader.get_available_models(),
        device=model_loader.device
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Info"])
async def get_classes():
    """Get list of politician classes"""
    return ClassesResponse(
        classes=CLASS_NAMES,
        count=len(CLASS_NAMES)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image file"),
    model_name: Optional[str] = Form(None, description="Model name (optional)")
):
    """
    Predict politician from uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
        model_name: Optional model name (uses best model if not specified)
    
    Returns:
        Prediction results with top-3 classes
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(model_loader.device)
        
        # Get model
        try:
            model = model_loader.get_model(model_name)
            used_model_name = model_name if model_name else model_loader.get_best_model_name()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Predict
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get top-3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        top3_results = []
        for prob, idx in zip(top_probs, top_indices):
            top3_results.append(
                PredictionResult(
                    class_name=CLASS_NAMES[idx],
                    confidence=prob.item()
                )
            )
        
        return PredictionResponse(
            predicted_class=CLASS_NAMES[top_indices[0]],
            confidence=top_probs[0].item(),
            top3=top3_results,
            model_used=used_model_name,
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of image files"),
    model_name: Optional[str] = Form(None, description="Model name (optional)")
):
    """
    Predict politicians from multiple images
    
    Args:
        files: List of image files
        model_name: Optional model name
    
    Returns:
        Batch prediction results
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    start_time = time.time()
    predictions = []
    
    for file in files:
        try:
            # Use single prediction endpoint logic
            result = await predict(file, model_name)
            predictions.append(result)
        except Exception as e:
            # Skip failed images
            print(f"Failed to process {file.filename}: {e}")
            continue
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(predictions),
        total_time_ms=round(total_time, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
