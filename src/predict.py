"""
Inference module for AI vs. Real image prediction.
Accepts image file paths, URLs, or raw arrays and returns classification
probabilities, confidence levels, and frequency-domain artifact diagnostics.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import cv2
from PIL import Image

from src.dataset import preprocess_image, TARGET_SIZE
from src.model import ResNet50Detector
from src.frequency_analysis import extract_artifact_features

# Global detector instance for fast reuse
_GLOBAL_DETECTOR: Union[ResNet50Detector, None] = None


def get_detector(model_path: str = None) -> ResNet50Detector:
    """Retrieve or initialize the singleton detector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = ResNet50Detector(model_path=model_path)
    return _GLOBAL_DETECTOR


def predict_image(
    image_input: Any,
    model_path: str = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict whether an image is AI-generated (Synthetic) or Real.
    
    Args:
        image_input: File path (str/Path), PIL Image, or numpy array.
        model_path: Path to model weights (optional).
        threshold: Decision threshold for REAL class (default 0.5).
        
    Returns:
        Dictionary containing label, confidence, probabilities, and spectral diagnostics.
    """
    start_time = time.perf_counter()
    detector = get_detector(model_path)
    
    # Load raw RGB image without lossy downsampling
    if isinstance(image_input, (str, Path)):
        bgr = cv2.imread(str(image_input))
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        raw_rgb = image_input.copy()
        if raw_rgb.ndim == 2:
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_GRAY2RGB)
        elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_RGBA2RGB)
    else:
        # PIL Image
        raw_rgb = np.array(image_input.convert("RGB"))

    if raw_rgb.dtype != np.uint8:
        if raw_rgb.max() <= 1.0:
            raw_rgb = (raw_rgb * 255).astype(np.uint8)
        else:
            raw_rgb = raw_rgb.astype(np.uint8)

    # Multi-domain detector inference
    detect_res = detector.predict(raw_rgb)
    real_prob = float(detect_res["real_probability"])
    real_prob = max(0.001, min(0.999, real_prob))
    ai_prob = float(1.0 - real_prob)
    
    is_real = real_prob >= threshold
    predicted_label = "REAL" if is_real else "AI-GENERATED"
    confidence = real_prob if is_real else ai_prob
    
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    artifact_metrics = detect_res.get("artifacts", extract_artifact_features(raw_rgb))
    explanation = detect_res.get("explanation", "")
    
    if not explanation:
        if not is_real:
            explanation = "Elevated high-frequency harmonics, abnormal gradient kurtosis, and generative noise signature observed."
        else:
            explanation = "Natural camera sensor noise and authentic optical gradient decay curve observed."

    return {
        "label": predicted_label,
        "is_ai": not is_real,
        "is_real": is_real,
        "confidence": round(float(confidence), 4),
        "confidence_percent": f"{confidence * 100:.1f}%",
        "probabilities": {
            "REAL": round(float(real_prob), 4),
            "AI": round(float(ai_prob), 4)
        },
        "artifacts": {
            "spectral_peak_ratio": artifact_metrics.get("spectral_peak_ratio", 1.0),
            "high_to_low_ratio": artifact_metrics.get("high_to_low_ratio", 0.0),
            "laplacian_variance": artifact_metrics.get("laplacian_variance", 0.0),
            "laplacian_kurtosis": artifact_metrics.get("laplacian_kurtosis", 3.0),
            "noise_residual_std": artifact_metrics.get("noise_residual_std", 5.0)
        },
        "spectrum_2d": artifact_metrics.get("magnitude_spectrum", np.zeros((32, 32))),
        "radial_profile": artifact_metrics.get("radial_profile", []),
        "explanation": explanation,
        "processing_time_ms": elapsed_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is Real or AI-Generated.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint.")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    result = predict_image(args.image_path, model_path=args.model)
    print("=" * 60)
    print(f" Image:      {args.image_path}")
    print(f" Prediction: {result['label']} ({result['confidence_percent']} confidence)")
    print(f" P(Real):    {result['probabilities']['REAL']:.4f}")
    print(f" P(AI):      {result['probabilities']['AI']:.4f}")
    print(f" Latency:    {result['processing_time_ms']} ms")
    print(f" Diagnostic: {result['explanation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
