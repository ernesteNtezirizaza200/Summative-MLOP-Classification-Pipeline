"""
FastAPI endpoints for brain tumor classification.
Provides prediction and retraining APIs.
"""

import os
# Suppress TensorFlow CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import cv2
from datetime import datetime
import pandas as pd
import logging
try:
    from src.prediction import BrainTumorPredictor
except ImportError:
    from prediction import BrainTumorPredictor
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Brain Tumor Classification API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Model uptime tracking
start_time = time.time()
request_count = 0


def load_predictor():
    """Load the predictor model."""
    global predictor
    try:
        logger.info("Loading predictor model...")
        predictor = BrainTumorPredictor()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting application...")
    load_predictor()
    logger.info("Application startup complete")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Brain Tumor Classification API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "retrain": "/retrain",
            "health": "/health",
            "uptime": "/uptime"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/uptime")
async def get_uptime():
    """Get API uptime and statistics."""
    global request_count
    uptime_seconds = time.time() - start_time
    uptime_hours = uptime_seconds / 3600
    uptime_minutes = (uptime_seconds % 3600) / 60
    
    return {
        "uptime_seconds": int(uptime_seconds),
        "uptime_formatted": f"{int(uptime_hours)}h {int(uptime_minutes)}m",
        "total_requests": request_count,
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor class from an uploaded image.
    """
    global request_count
    request_count += 1
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        start_pred_time = time.time()
        result = predictor.predict(img, return_probabilities=True)
        prediction_time = time.time() - start_pred_time
        
        return JSONResponse({
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "prediction_time_seconds": round(prediction_time, 4),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict classes for multiple images.
    """
    global request_count
    request_count += len(files)
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            result = predictor.predict(img, return_probabilities=True)
            result['filename'] = file.filename
            results.append(result)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "results": results,
        "total_images": len(files),
        "timestamp": datetime.now().isoformat()
    })


@app.post("/retrain")
async def trigger_retrain(files: list[UploadFile] = File(...), class_name: str = "unknown"):
    """
    Upload images for retraining and save to database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Import database
    try:
        from src.database import get_database
    except ImportError:
        from database import get_database
    
    db = get_database()
    
    # Create upload directory with class subfolder
    upload_dir = f"data/retrain_uploads/{class_name}"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    saved_to_db = []
    
    try:
        # Save uploaded files
        for file in files:
            contents = await file.read()
            file_path = os.path.join(upload_dir, file.filename)
            
            # Save to disk
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Save to database
            image_id = db.save_uploaded_image(
                filename=file.filename,
                class_name=class_name,
                file_path=file_path,
                file_size=len(contents),
                metadata={
                    'upload_source': 'api',
                    'upload_timestamp': datetime.now().isoformat()
                }
            )
            
            saved_files.append(file_path)
            saved_to_db.append(image_id)
        
        return JSONResponse({
            "message": "Files uploaded successfully and saved to database.",
            "files_uploaded": len(saved_files),
            "database_ids": saved_to_db,
            "file_paths": saved_files,
            "note": "Files are ready for retraining. Use retrain endpoint to trigger training.",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/retrain/status")
async def retrain_status():
    """Get retraining status from database."""
    try:
        from src.database import get_database
    except ImportError:
        from database import get_database
    
    db = get_database()
    sessions = db.get_training_sessions(limit=1)
    
    if sessions:
        latest = sessions[0]
        return {
            "status": latest['status'],
            "session_id": latest['id'],
            "timestamp": latest['session_timestamp'],
            "final_accuracy": latest.get('final_accuracy'),
            "images_used": latest.get('images_used'),
            "message": f"Latest training session: {latest['status']}"
        }
    else:
        return {
            "status": "no_sessions",
            "message": "No training sessions found",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics."""
    try:
        from src.database import get_database
    except ImportError:
        from database import get_database
    
    db = get_database()
    stats = db.get_training_statistics()
    return stats


@app.get("/visualizations/data")
async def get_visualization_data():
    """Get feature data for visualizations."""
    feature_files = [
        'data/processed/image_features_train.csv',
        'data/processed/image_features.csv',
        'image_features_train.csv',
        'image_features.csv'
    ]
    
    feature_file = None
    for f in feature_files:
        if os.path.exists(f):
            feature_file = f
            break
    
    if not feature_file:
        return {
            "features": [],
            "total_samples": 0,
            "source": "none",
            "has_mean_intensity": False,
            "has_std_intensity": False,
            "has_gradient_mean": False
        }
    
    try:
        df = pd.read_csv(feature_file)
        
        # Prepare data
        result = {
            "features": df.to_dict('records'),
            "total_samples": len(df),
            "source": feature_file,
            "has_mean_intensity": 'mean_intensity' in df.columns and 'class' in df.columns,
            "has_std_intensity": 'std_intensity' in df.columns and 'class' in df.columns,
            "has_gradient_mean": 'gradient_mean' in df.columns and 'mean_intensity' in df.columns and 'class' in df.columns
        }
        
        # Add specific data arrays for charts
        if result["has_mean_intensity"]:
            result["mean_intensity_data"] = df[['class', 'mean_intensity']].to_dict('records')
        
        if result["has_std_intensity"]:
            result["std_intensity_data"] = df[['class', 'std_intensity']].to_dict('records')
        
        if result["has_gradient_mean"]:
            result["gradient_data"] = df[['class', 'mean_intensity', 'gradient_mean']].to_dict('records')
        
        # Class distribution
        if 'class' in df.columns:
            class_counts = df['class'].value_counts().to_dict()
            result["class_distribution"] = class_counts
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading visualization data: {str(e)}")


@app.get("/retrain/sessions")
async def get_training_sessions(limit: int = 5):
    """Get recent training sessions."""
    try:
        from src.database import get_database
    except ImportError:
        from database import get_database
    
    db = get_database()
    sessions = db.get_training_sessions(limit=limit)
    return sessions


class RetrainRequest(BaseModel):
    epochs: int = 10
    fine_tune_epochs: int = 3

@app.post("/retrain/trigger")
async def trigger_retraining(request: RetrainRequest):
    """Trigger model retraining."""
    try:
        from retrain import retrain_model
    except ImportError:
        raise HTTPException(status_code=500, detail="Retraining module not found")
    
    try:
        result = retrain_model(
            retrain_data_dir='data/retrain_uploads',
            epochs=request.epochs,
            fine_tune_epochs=request.fine_tune_epochs
        )
        
        if result:
            # Reload predictor
            load_predictor()
            return {
                "message": "Retraining completed successfully",
                "epochs": request.epochs,
                "fine_tune_epochs": request.fine_tune_epochs,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Retraining failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


if __name__ == "__main__":
    # Load predictor before starting server
    load_predictor()
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    # Run server with proper logging
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )

