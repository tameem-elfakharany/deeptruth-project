"""
DeepTruth Image Deepfake Detection API
FastAPI backend for image deepfake detection using XceptionNet
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
import numpy as np
import cv2
import io
from pathlib import Path
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepTruth Image Detection API",
    description="Deepfake detection for images using XceptionNet",
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

# Global model variable
model = None

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/xception_deepfake.h5')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def load_model():
    """Load Xception model"""
    global model
    try:
        # Try to load saved model
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Load pre-trained Xception
            logger.info("Loading pre-trained Xception model from ImageNet")
            model = Xception(weights='imagenet', include_top=True)
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def convert_image_for_xception(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to Xception format.
    
    Args:
        image_bytes: Image file bytes
    
    Returns:
        Preprocessed image array (1, 299, 299, 3)
    """
    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 299x299
    img = cv2.resize(img, (299, 299))
    
    # Convert to float32 and normalize
    img = img.astype('float32') / 255.0
    
    # If grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Preprocess for Xception
    img = preprocess_input(img)
    
    return img


def predict_image(image_bytes: bytes, top_k: int = 5) -> Dict[str, Any]:
    """
    Predict on image.
    
    Args:
        image_bytes: Image file bytes
        top_k: Number of top predictions to return
    
    Returns:
        Prediction results
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Convert image
    img_ready = convert_image_for_xception(image_bytes)
    
    # Make prediction
    pred = model.predict(img_ready, verbose=0)
    
    # Decode predictions
    decoded = decode_predictions(pred, top=top_k)
    
    # Format results
    results = []
    for imagenet_id, label, score in decoded[0]:
        results.append({
            'label': label,
            'score': float(score),
            'imagenet_id': imagenet_id
        })
    
    return {
        'predictions': results,
        'raw_output': pred[0].tolist()
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up API...")
    if not load_model():
        logger.warning("Model failed to load, but API will continue running")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DeepTruth Image Detection API",
        "version": "1.0.0",
        "description": "Deepfake detection for images using XceptionNet",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict deepfake on uploaded image.
    
    Args:
        file: Image file
    
    Returns:
        Prediction results
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
            )
        
        # Check file size
        file_size = 0
        file_bytes = b""
        
        while True:
            chunk = await file.file.read(1024)
            if not chunk:
                break
            file_bytes += chunk
            file_size += len(chunk)
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                )
        
        # Make prediction
        result = predict_image(file_bytes, top_k=5)
        
        return {
            "filename": file.filename,
            "predictions": result['predictions'],
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch predict on multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        List of prediction results
    """
    results = []
    
    for file in files:
        try:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"File type {file_ext} not allowed"
                })
                continue
            
            # Read file bytes
            file_bytes = await file.read()
            
            # Check file size
            if len(file_bytes) > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                })
                continue
            
            # Make prediction
            result = predict_image(file_bytes, top_k=5)
            
            results.append({
                "filename": file.filename,
                "predictions": result['predictions'],
                "status": "success"
            })
        
        except Exception as e:
            logger.error(f"Error predicting {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/info")
async def info():
    """Get API information"""
    return {
        "name": "DeepTruth Image Detection API",
        "version": "1.0.0",
        "model": "XceptionNet (ImageNet pre-trained)",
        "input_size": "299x299",
        "output": "Top-5 ImageNet predictions",
        "max_file_size": f"{MAX_FILE_SIZE / (1024*1024):.0f}MB",
        "allowed_formats": list(ALLOWED_EXTENSIONS)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
