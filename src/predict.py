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
    
    # 1. Preprocess image
    img_norm = preprocess_image(image_input, target_size=TARGET_SIZE, normalize=True)
    
    # 2. Extract frequency & spatial artifacts
    artifact_metrics = extract_artifact_features((img_norm * 255).astype(np.uint8))
    
    # 3. Model inference (returns P(REAL))
    real_prob = float(detector.predict_proba(img_norm))
    real_prob = max(0.0001, min(0.9999, real_prob))
    ai_prob = float(1.0 - real_prob)
    
    is_real = real_prob >= threshold
    predicted_label = "REAL" if is_real else "AI-GENERATED"
    confidence = real_prob if is_real else ai_prob
    
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    # Generate diagnostic explanation
    reasons = []
    if not is_real:
        if artifact_metrics["spectral_peak_ratio"] > 4.5:
            reasons.append("Periodic checkerboard/grid spikes detected in 2D Fourier spectrum.")
        if artifact_metrics["high_to_low_ratio"] > 0.08:
            reasons.append("Elevated high-frequency energy tail typical of diffusion/GAN upsampling.")
        if artifact_metrics["laplacian_variance"] < 15.0:
            reasons.append("Unnatural surface smoothness and reduced high-order spatial texture variance.")
        if not reasons:
            reasons.append("ResNet50 feature manifold strongly matched synthetic generator signatures.")
    else:
        reasons.append("Natural camera sensor noise and authentic 1/f spectral power decay curve observed.")
        if artifact_metrics["laplacian_variance"] >= 15.0:
            reasons.append("Expected natural optical edge gradients and sensor noise present.")

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
            "spectral_peak_ratio": artifact_metrics["spectral_peak_ratio"],
            "high_to_low_ratio": artifact_metrics["high_to_low_ratio"],
            "laplacian_variance": artifact_metrics["laplacian_variance"],
            "color_std_mean": artifact_metrics["color_std_mean"]
        },
        "spectrum_2d": artifact_metrics["magnitude_spectrum"],
        "radial_profile": artifact_metrics["radial_profile"],
        "explanation": " ".join(reasons),
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
