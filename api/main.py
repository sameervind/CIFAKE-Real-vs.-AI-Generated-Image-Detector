"""
FastAPI REST API for CIFAKE AI-Generated vs. Real Image Detection.
Provides endpoints for single image inference, batch inference, model performance metrics,
health checks, and sample library listings.
Uses ResNet50 as the authoritative classifier with supplementary 2D Fourier/sensor noise diagnostics.
Zero silent fallback.
"""

import sys
import os
import json
import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root workspace is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import TARGET_SIZE
from src.model import ModelUnavailableError
from src.predict import predict_image, get_inference_service

app = FastAPI(
    title="CIFAKE AI vs. Real Image Detection API",
    description="High-performance forensic classification API using ResNet50 Transfer Learning & 2D FFT Frequency Domain Analysis to detect synthetic diffusion images vs real photographs.",
    version="2.0.0",
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
MODEL_DIR = ROOT_DIR / "model"
SAMPLES_DIR = ROOT_DIR / "samples"


class PredictionResponse(BaseModel):
    filename: Optional[str] = None
    label: str
    verdict: Optional[str] = None
    primary_driver: Optional[str] = None
    is_ai: bool
    is_real: bool
    is_ood: Optional[bool] = None
    ood_score: Optional[float] = None
    ood_verdict: Optional[str] = None
    ood_telemetry: Optional[Dict[str, Any]] = None
    forensic_telemetry: Optional[Dict[str, Any]] = None
    confidence: float
    confidence_percent: str
    probabilities: Dict[str, float]
    backend: str
    decision_threshold: float
    artifacts: Dict[str, Any]
    explanation: str
    processing_time_ms: float
    scientific_note: str


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    model_loaded: bool
    model_path: str
    framework: str
    input_resolution: List[int]
    device: str


@app.get("/", summary="Root Endpoint")
def read_root():
    service = get_inference_service()
    return {
        "title": "CIFAKE AI vs. Real Image Detection API",
        "version": "2.0.0",
        "status": "online",
        "model_loaded": service.is_model_loaded(),
        "backend": service.active_backend,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "generalization_metrics": "/generalization-metrics",
        "samples": "/samples"
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health Check")
def get_health():
    service = get_inference_service()
    model_loaded = service.is_model_loaded()
    model_path = str(service.model_path) if service.model_path else "None"
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        framework_str = f"TensorFlow {tf.__version__} / Keras 3"
    except Exception:
        framework_str = "TensorFlow Unavailable"

    return {
        "status": "healthy" if model_loaded else "degraded",
        "version": "2.0.0",
        "backend": service.active_backend,
        "model_loaded": model_loaded,
        "model_path": model_path,
        "framework": framework_str,
        "input_resolution": list(TARGET_SIZE) + [3],
        "device": "CPU (oneDNN Optimized)"
    }


@app.get("/metrics", summary="Model Evaluation Benchmarks")
def get_metrics():
    # Priority 1: Active trained checkpoint evaluation metrics
    active_metrics_file = MODEL_DIR / "evaluation_metrics.json"
    if active_metrics_file.exists():
        try:
            with open(active_metrics_file, "r") as f:
                data = json.load(f)
            data["metrics_source"] = "active_trained_checkpoint"
            return data
        except Exception:
            pass

    # Priority 2: Benchmark reference metrics
    benchmark_metrics_file = ASSETS_DIR / "metrics.json"
    if benchmark_metrics_file.exists():
        try:
            with open(benchmark_metrics_file, "r") as f:
                data = json.load(f)
            data["metrics_source"] = "cifake_benchmark_reference"
            return data
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No evaluation metrics found. Run training or evaluation pipeline to generate metrics."
    )


@app.get("/generalization-metrics", summary="Real-World Multi-Domain Generalization Benchmarks")
def get_generalization_metrics():
    """Retrieve multi-domain generalization evaluation across benchmark and external categories."""
    try:
        from src.generalization_benchmark import load_generalization_metrics
        return load_generalization_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load generalization metrics: {str(e)}"
        )


def _sanitize_artifacts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure artifacts dictionary contains only JSON-serializable primitives."""
    sanitized = {}
    for k, v in artifacts.items():
        if isinstance(v, np.ndarray):
            continue  # Exclude large binary arrays from JSON API payload
        elif isinstance(v, (np.floating, float)):
            sanitized[k] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            sanitized[k] = int(v)
        elif isinstance(v, list):
            if len(v) <= 50 and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
                sanitized[k] = [round(float(x), 4) for x in v]
        elif isinstance(v, (str, bool)):
            sanitized[k] = v
    return sanitized


@app.post("/predict", response_model=PredictionResponse, summary="Detect AI vs. Real for a Single Image")
async def predict_single(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for REAL class")
):
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
        label_val = "AI-GENERATED" if result["label"] == "AI" else result["label"]
        return {
            "filename": file.filename,
            "label": label_val,
            "verdict": result.get("verdict", label_val),
            "primary_driver": result.get("primary_driver", "CIFAKE_RESNET50"),
            "is_ai": result["is_ai"],
            "is_real": result["is_real"],
            "is_ood": result.get("is_ood", False),
            "ood_score": result.get("ood_score", 0.0),
            "ood_verdict": result.get("ood_verdict", "IN_DISTRIBUTION"),
            "ood_telemetry": _sanitize_artifacts(result.get("ood_telemetry", {})),
            "forensic_telemetry": _sanitize_artifacts(result.get("forensic_telemetry", {})),
            "confidence": result["confidence"],
            "confidence_percent": result["confidence_percent"],
            "probabilities": result["probabilities"],
            "backend": result.get("backend", "ResNet50"),
            "decision_threshold": result.get("decision_threshold", threshold),
            "artifacts": _sanitize_artifacts(result.get("artifacts", {})),
            "explanation": result["explanation"],
            "processing_time_ms": result["processing_time_ms"],
            "scientific_note": (
                "Classification reflects statistical resemblance to the CIFAKE benchmark dataset "
                "(CIFAR-10 real photographs vs. Stable Diffusion v1.4 synthetic images at 32x32 resolution). "
                "The platform applies Out-of-Distribution detection and Multi-Scale Physical Forensics "
                "to prevent false REAL predictions on modern high-resolution diffusion models."
            )
        }
    except ModelUnavailableError as mue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ResNet50 model unavailable: {str(mue)}. Train model using 'python -m src.train' before requesting inference."
        )
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

    # Pre-check model availability before looping
    service = get_inference_service()
    if not service.is_model_loaded():
        try:
            service.load_model()
        except ModelUnavailableError as mue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ResNet50 model unavailable: {str(mue)}. Train model using 'python -m src.train' before batch inference."
            )

    results = []
    for file in files:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
            img_array = np.array(pil_image)
            res = predict_image(img_array, threshold=threshold)
            label_val = "AI-GENERATED" if res["label"] == "AI" else res["label"]
            results.append({
                "filename": file.filename,
                "label": label_val,
                "verdict": res.get("verdict", label_val),
                "primary_driver": res.get("primary_driver", "CIFAKE_RESNET50"),
                "is_ai": res["is_ai"],
                "is_real": res["is_real"],
                "is_ood": res.get("is_ood", False),
                "ood_score": res.get("ood_score", 0.0),
                "ood_verdict": res.get("ood_verdict", "IN_DISTRIBUTION"),
                "confidence": res["confidence"],
                "confidence_percent": res["confidence_percent"],
                "probabilities": res["probabilities"],
                "backend": res.get("backend", "ResNet50"),
                "decision_threshold": res.get("decision_threshold", threshold),
                "artifacts": _sanitize_artifacts(res.get("artifacts", {})),
                "explanation": res["explanation"],
                "processing_time_ms": res["processing_time_ms"],
                "scientific_note": "Resemblance to CIFAKE benchmark (CIFAR-10 vs. Stable Diffusion v1.4)."
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
                "backend": "None",
                "decision_threshold": threshold,
                "artifacts": {},
                "explanation": f"Failed: {str(e)}",
                "processing_time_ms": 0.0,
                "scientific_note": "Processing error"
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
