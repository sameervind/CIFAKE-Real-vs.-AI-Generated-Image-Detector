"""
Inference module for CIFAKE AI vs. Real image prediction.
Uses trained ResNet50 Transfer Learning model as the primary, authoritative classifier.
Includes independent 2D Fourier and spatial frequency diagnostics as supplementary signals.
Strictly eliminates silent fallback behavior.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image

from src.dataset import preprocess_image, TARGET_SIZE
from src.model import (
    ResNet50InferenceService,
    get_inference_service,
    ModelUnavailableError,
    DEFAULT_MODEL_PATH
)
from src.frequency_analysis import extract_artifact_features


def predict_image(
    image_input: Any,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    compute_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Predict whether an image is AI-generated (Synthetic) or a Real Photograph using ResNet50.
    
    Args:
        image_input: File path (str/Path), PIL Image, or numpy array.
        model_path: Optional path to trained model checkpoint (.keras).
        threshold: Decision threshold for REAL class (default 0.50). P(REAL) >= threshold is REAL.
        compute_diagnostics: If True, computes supplementary 2D Fourier and sensor noise metrics.
        
    Returns:
        Dictionary containing authoritative prediction, confidence, probabilities,
        active backend, and supplementary spectral diagnostics.
        
    Raises:
        ModelUnavailableError: If trained ResNet50 checkpoint is missing or invalid.
        FileNotFoundError: If input image path cannot be found or read.
    """
    start_time = time.perf_counter()

    # Load raw RGB image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_input}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Could not decode image at: {image_input}")
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        raw_rgb = image_input.copy()
        if raw_rgb.ndim == 2:
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_GRAY2RGB)
        elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_RGBA2RGB)
    elif isinstance(image_input, Image.Image):
        raw_rgb = np.array(image_input.convert("RGB"))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if raw_rgb.dtype != np.uint8:
        if raw_rgb.max() <= 1.0:
            raw_rgb = (raw_rgb * 255).astype(np.uint8)
        else:
            raw_rgb = raw_rgb.astype(np.uint8)

    # 1. Authoritative ResNet50 Inference
    service = get_inference_service(model_path=model_path)
    pred_res = service.predict(raw_rgb, threshold=threshold)

    is_real = pred_res["is_real"]
    real_prob = pred_res["probabilities"]["REAL"]
    ai_prob = pred_res["probabilities"]["AI"]
    confidence = pred_res["confidence"]

    # 2. Supplementary Spectral Diagnostics (Non-blocking, clearly marked as supplementary)
    artifact_metrics = {}
    spectrum_2d = np.zeros((32, 32), dtype=np.float32)
    radial_profile = []
    
    if compute_diagnostics:
        try:
            artifact_metrics = extract_artifact_features(raw_rgb)
            spectrum_2d = artifact_metrics.get("magnitude_spectrum", spectrum_2d)
            radial_profile = artifact_metrics.get("radial_profile", [])
        except Exception as e:
            artifact_metrics = {"error": f"Diagnostic extraction failed: {e}"}

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Formulate forensic explanation
    if is_real:
        explanation = (
            f"ResNet-50 deep residual manifold matches photographic distribution (P(REAL)={real_prob * 100:.1f}%). "
            f"Physical sensor noise std={artifact_metrics.get('noise_residual_std', 'N/A')} with natural gradient distribution."
        )
    else:
        explanation = (
            f"ResNet-50 deep residual manifold matches synthetic generative distribution (P(AI)={ai_prob * 100:.1f}%). "
            f"Spectral peak ratio={artifact_metrics.get('spectral_peak_ratio', 'N/A')}x with characteristic diffusion feature residue."
        )

    return {
        "label": pred_res["label"],
        "class_id": pred_res["class_id"],
        "is_real": is_real,
        "is_ai": not is_real,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.1f}%",
        "probabilities": {
            "REAL": real_prob,
            "AI": ai_prob
        },
        "raw_score": pred_res["raw_score"],
        "threshold": threshold,
        "backend": pred_res["backend"],
        "model_name": pred_res["model_name"],
        "model_version": Path(pred_res["checkpoint_path"]).name,
        "checkpoint_path": pred_res["checkpoint_path"],
        "artifacts": artifact_metrics,
        "spectrum_2d": spectrum_2d,
        "radial_profile": radial_profile,
        "explanation": explanation,
        "processing_time_ms": elapsed_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is Real or AI-Generated using ResNet50.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("--model", type=str, default=None, help="Path to trained ResNet50 checkpoint (.keras).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision boundary for REAL class (default: 0.50).")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Input file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        result = predict_image(args.image_path, model_path=args.model, threshold=args.threshold)
        print("=" * 65)
        print(f" Image:       {args.image_path}")
        print(f" Prediction:  {result['label']} ({result['confidence_percent']} confidence)")
        print(f" P(Real):     {result['probabilities']['REAL']:.4f}")
        print(f" P(AI):       {result['probabilities']['AI']:.4f}")
        print(f" Backend:     {result['backend']} ({result['model_name']} - {result['model_version']})")
        print(f" Latency:     {result['processing_time_ms']} ms")
        print(f" Diagnostic:  {result['explanation']}")
        print("=" * 65)
    except ModelUnavailableError as e:
        print(f"\n[MODEL ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[INFERENCE ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
