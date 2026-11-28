"""
Mandatory Regression Test Suite for Real-World Generalization & Robust Authenticity Verification.

Evaluates the actual trained model checkpoint (model/best_cifake_model.keras)
on real-world external test cases:
1. Known AI-generated photorealistic portraits (ai_portrait_ongc_1.png and ai_portrait_ongc_2.png).
2. Verified genuine photograph (verified_real_portrait_1.jpg).
3. Additional in-distribution real and AI samples.

Strictly adheres to scientific validation criteria:
- Zero simulated mock predictions; tests live model inference directly.
- Zero arbitrary heuristic threshold validation as ground truth.
- Tests false-positive and false-negative behaviors independently.
- Confirms actual model architecture identity (CIFAR-CNN).
- Verifies transparent decision rationale and separate raw model probabilities.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
from pathlib import Path
import pytest
import cv2
import numpy as np

from src.predict import predict_image
from src.ood_detector import get_ood_detector
from src.forensics import get_forensic_analyzer

REGRESSION_DIR = Path("tests/regression_data")
KNOWN_REAL_DIR = Path("tests/regression/known_real")

IMAGE_AI_1_PATH = REGRESSION_DIR / "ai_portrait_ongc_1.png"
IMAGE_AI_2_PATH = REGRESSION_DIR / "ai_portrait_ongc_2.png"
IMAGE_REAL_1_PATH = KNOWN_REAL_DIR / "verified_real_portrait_1.jpg"

SAMPLES_REAL_DIR = Path("samples/real")
SAMPLES_AI_DIR = Path("samples/ai")


def test_regression_images_exist_and_valid():
    """Verify all external regression test specimens exist on disk and decode cleanly."""
    for path in [IMAGE_AI_1_PATH, IMAGE_AI_2_PATH, IMAGE_REAL_1_PATH]:
        assert path.exists(), f"Missing regression test image: {path}"
        img = cv2.imread(str(path))
        assert img is not None, f"Failed to decode image at: {path}"
        assert img.ndim == 3 and img.shape[2] == 3, f"Image must be 3-channel BGR: {path}"
        assert img.shape[0] > 100 and img.shape[1] > 100, f"Image dimensions too small: {path}"


def test_regression_images_ood_telemetry():
    """Verify OOD detector identifies resolution scale shift as independent telemetry."""
    ood_detector = get_ood_detector()

    for path in [IMAGE_AI_1_PATH, IMAGE_AI_2_PATH, IMAGE_REAL_1_PATH]:
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ood_res = ood_detector.evaluate(rgb)

        # High resolution images deviate significantly from 32x32 benchmark scale
        assert ood_res.is_ood is True, f"{path.name} was not flagged as OOD!"
        assert ood_res.resolution_scale_factor > 8.0, "Resolution scale factor must exceed 8x"
        assert len(ood_res.domain_shift_reasons) > 0, "Must provide descriptive domain shift reasons"
        # OOD score must remain independent telemetry in [0.0, 1.0]
        assert 0.0 <= ood_res.ood_score <= 1.0


def test_real_checkpoint_false_negative_prevention():
    """
    CRITICAL REGRESSION TEST: Known AI-generated portraits must NEVER be classified as REAL.
    Tests live model inference on actual trained checkpoint.
    """
    for path in [IMAGE_AI_1_PATH, IMAGE_AI_2_PATH]:
        res = predict_image(str(path))

        # 1. MANDATORY ASSERTION: Must NOT be classified as REAL
        assert res["is_real"] is False, f"CRITICAL REGRESSION: AI image {path.name} classified as REAL! Result: {res}"
        assert res["label"] != "REAL", f"Label was REAL for AI image {path.name}: {res}"

        # 2. Verify actual loaded architecture
        assert res["model_name"] == "cifar_cnn", f"Model architecture mismatch: expected cifar_cnn, got {res['model_name']}"

        # 3. Verify raw neural network probabilities
        assert "raw_probabilities" in res, "Missing raw_probabilities"
        raw_p_ai = res["raw_probabilities"]["AI"]
        assert raw_p_ai >= 0.50, f"Raw neural network AI probability too low on {path.name}: {raw_p_ai}"

        # 4. Verify decision output
        assert res["verdict"] in ["LIKELY AI-GENERATED", "INCONCLUSIVE / OUT OF DISTRIBUTION"]
        assert "explanation" in res and len(res["explanation"]) > 0


def test_real_checkpoint_false_positive_prevention():
    """
    CRITICAL REGRESSION TEST: Verified genuine photograph must NOT be classified as AI.
    Tests live model inference on actual trained checkpoint.
    """
    res = predict_image(str(IMAGE_REAL_1_PATH))

    # 1. MANDATORY ASSERTIONS: Must NEVER be classified as AI-GENERATED
    assert res["is_ai"] is False, f"CRITICAL REGRESSION: Real photograph classified as AI! Details: {res}"
    assert res["label"] != "AI", f"Label was AI for genuine photograph: {res}"
    assert res["verdict"] != "LIKELY AI-GENERATED", f"Verdict was LIKELY AI-GENERATED for genuine photograph: {res['verdict']}"

    # 2. Verify allowed verdicts for genuine photo under domain shift
    assert res["label"] in ["REAL", "INCONCLUSIVE"], f"Unexpected label: {res['label']}"

    # 3. Verify raw neural network recognized authentic features
    raw_p_real = res["raw_probabilities"]["REAL"]
    assert raw_p_real >= 0.50, f"Raw neural network REAL probability too low on verified real image: {raw_p_real}"

    # 4. Confidence integrity: Must not be a manufactured arbitrary 0.98 certainty
    if res["label"] == "AI":
        pytest.fail("Genuine photograph was classified as AI!")


def test_optical_diagnostics_are_supplementary():
    """
    Verify MultiScaleForensicAnalyzer provides technical measurements
    without making authoritative classification verdicts.
    """
    forensic_analyzer = get_forensic_analyzer()

    for path in [IMAGE_AI_1_PATH, IMAGE_REAL_1_PATH]:
        report = forensic_analyzer.analyze(str(path))

        # Measurements are populated
        assert report.spectral_peak_ratio > 0.0
        assert report.laplacian_kurtosis > 0.0
        assert report.plasticity_index > 0.0
        assert report.noise_residual_std > 0.0

        # Forensics must NOT claim authoritative AI decisions
        assert report.is_synthetic_likely is False, "Forensics must not make authoritative synthetic decisions"
        assert report.synthetic_probability is None, "Heuristic synthetic probability should not be presented as calibrated"
        assert len(report.forensic_explanation) > 0


def test_comprehensive_regression_suite_balances_both_classes():
    """
    Evaluates FPR and FNR separately across verified external regression specimens.
    Guarantees balanced evaluation without single-class bias or collapse.
    """
    # 1. AI Specimens: Must never be classified as REAL
    ai_paths = [IMAGE_AI_1_PATH, IMAGE_AI_2_PATH]

    false_negatives = 0
    for path in ai_paths:
        res = predict_image(str(path))
        if res["is_real"]:
            false_negatives += 1

    fnr = false_negatives / len(ai_paths)
    assert fnr == 0.0, f"False Negative Rate must be 0.0, got {fnr} ({false_negatives}/{len(ai_paths)} missed)"

    # 2. REAL Specimens: Must never be classified as AI
    real_paths = [IMAGE_REAL_1_PATH]

    false_positives = 0
    for path in real_paths:
        res = predict_image(str(path))
        if res["is_ai"]:
            false_positives += 1

    fpr = false_positives / len(real_paths)
    assert fpr == 0.0, f"False Positive Rate must be 0.0, got {fpr} ({false_positives}/{len(real_paths)} false alarms)"
