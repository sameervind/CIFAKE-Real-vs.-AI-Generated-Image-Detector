"""
FastAPI REST API for CIFAKE AI-Generated vs. Real Image Detection.
Provides endpoints for single image inference, batch inference, model performance metrics,
and health checks.
"""

import sys
import os
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure root workspace is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_image
from src.dataset import TARGET_SIZE

app = FastAPI(
    title="CIFAKE AI vs. Real Image Detection API",
    description="High-performance classification API using ResNet50 Transfer Learning & 2D FFT Frequency Domain Analysis to detect synthetic diffusion images vs real photographs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_DIR = ROOT_DIR / "assets"
SAMPLES_DIR = ROOT_DIR / "samples"


class PredictionResponse(BaseModel):
    filename: Optional[str] = None
    label: str
    is_ai: bool
    is_real: bool
    confidence: float
    confidence_percent: str
    probabilities: Dict[str, float]
    artifacts: Dict[str, Any]
    explanation: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    input_resolution: List[int]
    device: str


@app.get("/", summary="Root Endpoint")
def read_root():
    return {
        "title": "CIFAKE AI vs. Real Image Detection API",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health Check")
def get_health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": "ResNet50 + 2D FFT Frequency Domain Engine",
        "input_resolution": list(TARGET_SIZE) + [3],
        "device": "CPU"
    }


@app.get("/metrics", summary="Model Evaluation Benchmarks")
def get_metrics():
    metrics_file = ASSETS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Run evaluation pipeline first.")
    with open(metrics_file, "r") as f:
        data = json.load(f)
    return data


@app.post("/predict", response_model=PredictionResponse, summary="Detect AI vs. Real for a Single Image")
async def predict_single(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for REAL class")
):
    # Validate mime type
    valid_content_types = ["image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"]
    if file.content_type and file.content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Must be JPEG, PNG, or WEBP."
        )

    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        img_array = np.array(pil_image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}"
        )

    try:
        result = predict_image(img_array, threshold=threshold)
        return {
            "filename": file.filename,
            "label": result["label"],
            "is_ai": result["is_ai"],
            "is_real": result["is_real"],
            "confidence": result["confidence"],
            "confidence_percent": result["confidence_percent"],
            "probabilities": result["probabilities"],
            "artifacts": result["artifacts"],
            "explanation": result["explanation"],
            "processing_time_ms": result["processing_time_ms"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/batch-predict", response_model=List[PredictionResponse], summary="Batch Detection for Multiple Images")
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum batch limit is 50 images per request."
        )

    results = []
    for file in files:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
            img_array = np.array(pil_image)
            res = predict_image(img_array, threshold=threshold)
            results.append({
                "filename": file.filename,
                "label": res["label"],
                "is_ai": res["is_ai"],
                "is_real": res["is_real"],
                "confidence": res["confidence"],
                "confidence_percent": res["confidence_percent"],
                "probabilities": res["probabilities"],
                "artifacts": res["artifacts"],
                "explanation": res["explanation"],
                "processing_time_ms": res["processing_time_ms"]
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "label": "ERROR",
                "is_ai": False,
                "is_real": False,
                "confidence": 0.0,
                "confidence_percent": "0.0%",
                "probabilities": {"REAL": 0.0, "AI": 0.0},
                "artifacts": {},
                "explanation": f"Failed: {str(e)}",
                "processing_time_ms": 0.0
            })
    return results


@app.get("/samples", summary="List Available Sample Images")
def list_samples():
    real_files = [f.name for f in (SAMPLES_DIR / "real").glob("*.jpg")]
    ai_files = [f.name for f in (SAMPLES_DIR / "ai").glob("*.jpg")]
    return {
        "real_count": len(real_files),
        "ai_count": len(ai_files),
        "real_samples": real_files,
        "ai_samples": ai_files
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
