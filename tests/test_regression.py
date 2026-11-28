"""
Mandatory Regression Test Suite for Real-World Generalization & False Real Failure Cases.

Tests the two supplied known AI-generated photorealistic portraits inside the
ONGC Vigilance Department office (ai_portrait_ongc_1.png and ai_portrait_ongc_2.png).
Verifies:
1. Both images are loaded and non-empty.
2. Both images are detected as Out-of-Distribution (OOD).
3. Modern Multi-Scale Forensics detects physical generative anomalies.
4. Transparent Decision Arbiter strictly prevents false REAL classification.
5. End-to-end inference produces LIKELY AI-GENERATED verdict with transparent rationale.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
from pathlib import Path
import pytest
import cv2
import numpy as np

from src.ood_detector import get_ood_detector
from src.forensics import get_forensic_analyzer
from src.decision_engine import get_decision_engine

REGRESSION_DIR = Path("tests/regression_data")
IMAGE_1_PATH = REGRESSION_DIR / "ai_portrait_ongc_1.png"
IMAGE_2_PATH = REGRESSION_DIR / "ai_portrait_ongc_2.png"


def test_regression_images_exist_and_valid():
    """Verify both mandatory regression images exist on disk and decode cleanly."""
    assert IMAGE_1_PATH.exists(), f"Missing regression test image: {IMAGE_1_PATH}"
    assert IMAGE_2_PATH.exists(), f"Missing regression test image: {IMAGE_2_PATH}"

    img1 = cv2.imread(str(IMAGE_1_PATH))
    assert img1 is not None, f"Failed to decode {IMAGE_1_PATH}"
    assert img1.ndim == 3 and img1.shape[2] == 3
    assert img1.shape[0] > 100 and img1.shape[1] > 100

    img2 = cv2.imread(str(IMAGE_2_PATH))
    assert img2 is not None, f"Failed to decode {IMAGE_2_PATH}"
    assert img2.ndim == 3 and img2.shape[2] == 3
    assert img2.shape[0] > 100 and img2.shape[1] > 100


def test_regression_images_are_detected_as_ood():
    """Verify OOD detector identifies severe domain shift away from 32x32 CIFAKE benchmark."""
    ood_detector = get_ood_detector()

    for path in [IMAGE_1_PATH, IMAGE_2_PATH]:
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ood_res = ood_detector.evaluate(rgb)

        assert ood_res.is_ood is True, f"{path.name} was not flagged as OOD!"
        assert ood_res.verdict == "OUT_OF_DISTRIBUTION"
        assert ood_res.resolution_scale_factor > 8.0, "Scale factor should be > 8x benchmark"
        assert len(ood_res.domain_shift_reasons) > 0
        assert ood_res.confidence_discount <= 0.50, "Raw CIFAKE classifier confidence must be discounted"


def test_regression_images_trigger_forensic_anomalies():
    """Verify Multi-Scale Forensic Analyzer detects physical generative anomalies."""
    forensic_analyzer = get_forensic_analyzer()

    for path in [IMAGE_1_PATH, IMAGE_2_PATH]:
        report = forensic_analyzer.analyze(str(path))

        assert report.is_synthetic_likely is True, f"{path.name} failed forensic detection!"
        assert report.synthetic_probability >= 0.70
        assert len(report.detected_anomalies) > 0
        # Check specific forensic signatures
        assert report.laplacian_kurtosis > 14.0 or report.plasticity_index > 6.5
        assert len(report.forensic_explanation) > 0


def test_regression_decision_arbiter_prevents_false_real():
    """
    Verify TransparentDecisionEngine arbitrates biased 32x32 ResNet50 output
    and overrides false REAL predictions using physical forensic evidence.
    """
    ood_detector = get_ood_detector()
    forensic_analyzer = get_forensic_analyzer()
    decision_engine = get_decision_engine()

    for path in [IMAGE_1_PATH, IMAGE_2_PATH]:
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ood_res = ood_detector.evaluate(rgb)
        forensic_res = forensic_analyzer.analyze(rgb)

        # Simulate baseline ResNet50 false REAL output (P(REAL) = 0.9446)
        biased_cifake_res = {
            "is_real": True,
            "probabilities": {"REAL": 0.9446, "AI": 0.0554},
            "confidence": 0.9446,
            "label": "REAL",
            "raw_score": 0.9446,
            "backend": "tensorflow_keras"
        }

        decision = decision_engine.arbitrate(biased_cifake_res, ood_res, forensic_res)

        # MANDATORY ASSERTION: Must NOT be classified as REAL
        assert decision.is_real is False, f"CRITICAL REGRESSION: {path.name} classified as REAL!"
        assert decision.label != "REAL", f"Label was REAL for {path.name}"
        assert decision.verdict == "LIKELY AI-GENERATED", f"Unexpected verdict: {decision.verdict}"
        assert decision.is_ai is True
        assert decision.primary_driver == "MODERN_FORENSICS"
        assert "Out-of-Distribution" in decision.synthesis_rationale


def test_regression_end_to_end_predict_image_on_ongc_portraits():
    """
    End-to-end regression test: Runs predict_image() directly on both ONGC portraits.
    Verifies that neither image is classified as REAL.
    """
    from src.predict import predict_image

    for path in [IMAGE_1_PATH, IMAGE_2_PATH]:
        res = predict_image(str(path))

        assert res["is_real"] is False, f"CRITICAL REGRESSION: predict_image classified {path.name} as REAL!"
        assert res["label"] != "REAL"
        assert res["is_ai"] is True
        assert res["is_ood"] is True
        assert res["verdict"] == "LIKELY AI-GENERATED"
        assert res["primary_driver"] == "MODERN_FORENSICS"
        assert res["probabilities"]["AI"] >= 0.70
        assert "Out-of-Distribution" in res["explanation"]

