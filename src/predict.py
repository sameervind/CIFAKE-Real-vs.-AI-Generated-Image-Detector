"""
Inference module for CIFAKE AI vs. Real image prediction.

Integrates:
1. Native CIFAR-CNN Benchmark Classifier (32x32 native convolutional architecture).
2. Out-of-Distribution (OOD) & Uncertainty Engine (Manifold distance & scale shift).
3. Supplementary Modern Multi-Scale Optical Diagnostics (PRNU noise, kurtosis, plasticity, TCA).
4. Supplementary 2D Fourier & spatial frequency diagnostics.
5. Transparent Multi-Evidence Decision Arbiter.

Strictly eliminates silent fallback behavior and overconfident false predictions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
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
from src.ood_detector import get_ood_detector
from src.forensics import get_forensic_analyzer
from src.decision_engine import get_decision_engine


def predict_image(
    image_input: Any,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    compute_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Predict whether an image is AI-generated (Synthetic) or a Real Photograph
    using the synthesized ResNet50 and Multi-Scale Forensic Decision Engine.
    
    Args:
        image_input: File path (str/Path), PIL Image, or numpy array.
        model_path: Optional path to trained model checkpoint (.keras).
        threshold: Decision threshold for REAL class (default 0.50). P(REAL) >= threshold is REAL.
        compute_diagnostics: If True, computes supplementary 2D Fourier and sensor noise metrics.
        
    Returns:
        Dictionary containing authoritative prediction, confidence, probabilities,
        active backend, OOD telemetry, forensic metrics, and transparent decision rationale.
        
    Raises:
        ModelUnavailableError: If trained ResNet50 checkpoint is missing or invalid.
        FileNotFoundError: If input image path cannot be found or read.
    """
    start_time = time.perf_counter()

    # Load raw RGB image at native resolution
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

    # 1. Authoritative ResNet50 Inference (on 224x224 preprocessed manifold)
    service = get_inference_service(model_path=model_path)
    pred_res = service.predict(raw_rgb, threshold=threshold)
    feature_emb = pred_res.get("feature_embedding")

    # 2. Out-of-Distribution & Uncertainty Evaluation
    ood_detector = get_ood_detector()
    ood_res = ood_detector.evaluate(
        raw_rgb,
        feature_embedding=feature_emb,
        raw_score=pred_res["raw_score"]
    )

    # 3. Modern Multi-Scale Forensic Analysis (on native un-downsampled image)
    forensic_analyzer = get_forensic_analyzer()
    forensic_res = forensic_analyzer.analyze(raw_rgb)

    # 4. Supplementary Spectral Diagnostics
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

    # 5. Transparent Multi-Evidence Arbitration
    decision_engine = get_decision_engine()
    decision = decision_engine.arbitrate(
        cifake_res=pred_res,
        ood_res=ood_res,
        forensic_res=forensic_res,
        spectral_metrics=artifact_metrics
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "label": decision.label,
        "verdict": decision.verdict,
        "class_id": decision.class_id,
        "is_real": decision.is_real,
        "is_ai": decision.is_ai,
        "is_inconclusive": decision.is_inconclusive,
        "confidence": decision.confidence,
        "confidence_percent": decision.confidence_percent,
        "primary_driver": decision.primary_driver,
        "probabilities": decision.probabilities,
        "raw_probabilities": decision.raw_probabilities,
        "raw_confidence": pred_res["confidence"],
        "raw_confidence_percent": pred_res["confidence_percent"],
        "raw_score": pred_res["raw_score"],
        "threshold": threshold,
        "backend": pred_res["backend"],
        "model_name": pred_res["model_name"],
        "architecture": "cifar_cnn (Native 32x32 Convolutional Neural Network)",
        "model_version": Path(pred_res["checkpoint_path"]).name,
        "checkpoint_path": pred_res["checkpoint_path"],
        "is_ood": ood_res.is_ood,
        "ood_score": ood_res.ood_score,
        "ood_verdict": ood_res.verdict,
        "ood_telemetry": decision.ood_telemetry,
        "forensic_telemetry": decision.forensic_telemetry,
        "cifake_telemetry": decision.cifake_telemetry,
        "artifacts": artifact_metrics,
        "spectrum_2d": spectrum_2d,
        "radial_profile": radial_profile,
        "explanation": decision.synthesis_rationale,
        "processing_time_ms": elapsed_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is Real or AI-Generated using CIFAKE Model.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint (.keras).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision boundary for REAL class (default: 0.50).")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Input file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        result = predict_image(args.image_path, model_path=args.model, threshold=args.threshold)
        print("\n================ CIFAKE PREDICTION RESULT ================")
        print(f"Verdict:             {result['verdict']}")
        print(f"Label:               {result['label']}")
        print(f"Confidence:          {result['confidence_percent']}")
        print(f"Decision Driver:     {result['primary_driver']}")
        print(f"P(REAL):             {result['probabilities']['REAL'] * 100:.2f}%")
        print(f"P(AI):               {result['probabilities']['AI'] * 100:.2f}%")
        print(f"OOD Status:          {result['ood_verdict']} (score: {result['ood_score']:.2f})")
        print(f"Backend:             {result['backend']}")
        print(f"Latency:             {result['processing_time_ms']} ms")
        print("----------------------------------------------------------")
        print(f"Rationale: {result['explanation']}")
        print("==========================================================\n")
    except ModelUnavailableError as e:
        print(f"\n[ERROR] Model unavailable: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
