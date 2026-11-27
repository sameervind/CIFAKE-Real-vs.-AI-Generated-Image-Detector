"""
Unit tests for CIFAKE model, preprocessing, frequency analysis, and inference.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import preprocess_image, augment_image_numpy, train_val_test_split, TARGET_SIZE
from src.frequency_analysis import compute_fft_spectrum, azimuthal_average, extract_artifact_features
from src.predict import predict_image, get_detector
from src.evaluate import calculate_metrics


def test_preprocess_image():
    # Test random RGB uint8 image
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    processed = preprocess_image(img, target_size=(32, 32), normalize=True)
    assert processed.shape == (32, 32, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_augment_image_numpy():
    img = np.random.uniform(0.1, 0.9, (32, 32, 3)).astype(np.float32)
    aug = augment_image_numpy(img)
    assert aug.shape == (32, 32, 3)
    assert aug.min() >= 0.0
    assert aug.max() <= 1.0


def test_train_val_test_split():
    X = np.random.uniform(0, 1, (100, 32, 32, 3)).astype(np.float32)
    y = np.random.randint(0, 2, (100,))
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y, 0.7, 0.2, 0.1)
    assert len(X_tr) == 70
    assert len(X_va) == 20
    assert len(X_te) == 10


def test_compute_fft_spectrum():
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    spectrum, shift = compute_fft_spectrum(img)
    assert spectrum.shape == (32, 32)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0
    assert shift.shape == (32, 32)


def test_extract_artifact_features():
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    features = extract_artifact_features(img)
    assert "magnitude_spectrum" in features
    assert "radial_profile" in features
    assert "high_to_low_ratio" in features
    assert "spectral_peak_ratio" in features
    assert "laplacian_variance" in features
    assert isinstance(features["high_to_low_ratio"], float)


def test_predict_image():
    # Test on a real sample if exists, else random image
    real_sample = ROOT_DIR / "samples" / "real" / "01_real_airplane.jpg"
    if real_sample.exists():
        result = predict_image(str(real_sample))
    else:
        test_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = predict_image(test_img)

    assert result["label"] in ["REAL", "AI-GENERATED"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "REAL" in result["probabilities"]
    assert "AI" in result["probabilities"]
    total_prob = result["probabilities"]["REAL"] + result["probabilities"]["AI"]
    assert np.isclose(total_prob, 1.0, atol=1e-3)
    assert "artifacts" in result
    assert "explanation" in result
    assert result["processing_time_ms"] > 0


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_probs = np.array([0.1, 0.8, 0.9, 0.7, 0.4])
    metrics = calculate_metrics(y_true, y_pred_probs, threshold=0.5)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_real_photos_not_falsely_flagged():
    """Verify that real photos (daylight, bokeh, low light, compressed) are classified as REAL."""
    import cv2
    
    # 1. Daylight landscape simulation
    landscape = np.zeros((400, 600, 3), dtype=np.uint8)
    landscape[:200, :] = [235, 206, 135]
    landscape[200:, :] = [34, 139, 34]
    noise = np.random.normal(0, 3.0, landscape.shape).astype(np.int16)
    landscape = np.clip(landscape.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    res_land = predict_image(landscape)
    assert res_land["label"] == "REAL"
    assert res_land["probabilities"]["REAL"] > 0.80

    # 2. Portrait bokeh simulation
    bg = cv2.GaussianBlur(np.random.randint(50, 180, (600, 400, 3), dtype=np.uint8), (51, 51), 0)
    cv2.circle(bg, (200, 300), 80, (180, 140, 120), -1)
    res_bokeh = predict_image(bg)
    assert res_bokeh["label"] == "REAL"
    assert res_bokeh["probabilities"]["REAL"] > 0.80

    # 3. WhatsApp JPEG compression simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, enc = cv2.imencode('.jpg', landscape, encode_param)
    wa_img = cv2.imdecode(enc, 1)
    res_wa = predict_image(cv2.cvtColor(wa_img, cv2.COLOR_BGR2RGB))
    assert res_wa["label"] == "REAL"
    assert res_wa["probabilities"]["REAL"] > 0.80

