"""
Unit tests for CIFAKE ResNet-50 Transfer Learning Engine, preprocessing,
frequency diagnostics, stratified dataset splitting, and inference service.
Strictly tests genuine ResNet50 deep learning and zero silent fallback.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import preprocess_image, train_val_test_split, TARGET_SIZE
from src.model import (
    build_resnet50_model,
    configure_fine_tuning,
    ResNet50InferenceService,
    ModelUnavailableError,
    get_inference_service
)
from src.predict import predict_image
from src.frequency_analysis import compute_fft_spectrum, extract_artifact_features
from src.evaluate import calculate_metrics, detect_prediction_collapse


def test_preprocess_image():
    """Verify ImageNet BGR zero-centering preprocessing pipeline."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    processed = preprocess_image(img, target_size=(32, 32), apply_imagenet_norm=True)
    assert processed.shape == (32, 32, 3)
    assert processed.dtype == np.float32
    # ImageNet BGR mean subtraction results in [-124, 152] range
    assert processed.min() >= -130.0
    assert processed.max() <= 160.0


def test_dataset_stratified_split():
    """Verify stratified splitting maintains exact class proportion across train, val, and test."""
    X = np.random.uniform(-100, 100, (100, 32, 32, 3)).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y, 0.7, 0.2, 0.1)
    assert len(X_tr) == 70
    assert len(X_va) == 20
    assert len(X_te) == 10
    # Stratified balance check
    assert np.sum(y_tr == 1) == 35
    assert np.sum(y_va == 1) == 10
    assert np.sum(y_te == 1) == 5


def test_model_architecture():
    """Verify production ResNet50 transfer learning architecture."""
    model = build_resnet50_model(input_shape=(32, 32, 3), freeze_backbone=True)
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)


def test_fine_tuning_configuration():
    """Verify fine-tuning unfreezes conv5 blocks while preserving BatchNorm freeze."""
    model = build_resnet50_model(input_shape=(32, 32, 3), freeze_backbone=True)
    ft_model = configure_fine_tuning(model, fine_tune_from_layer="conv5_block1_out", fine_tune_lr=1e-5)
    base_model = ft_model.get_layer("resnet50")
    # Verify that conv5 convolutional layers after conv5_block1_out are trainable
    conv5_layer = base_model.get_layer("conv5_block2_1_conv")
    assert conv5_layer.trainable is True
    # Verify BatchNormalization layers remain frozen
    bn_layer = base_model.get_layer("conv5_block2_1_bn")
    assert bn_layer.trainable is False


def test_inference_service_authoritative():
    """Verify authoritative inference via ResNet50 checkpoint."""
    service = get_inference_service()
    assert service.is_model_loaded() is True
    
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
    assert np.isclose(result["probabilities"]["REAL"] + result["probabilities"]["AI"], 1.0, atol=1e-3)
    assert "ResNet50" in result["model_name"]
    assert result["processing_time_ms"] > 0


def test_model_unavailable_error():
    """Verify explicit ModelUnavailableError when checkpoint is missing (zero silent fallback)."""
    with pytest.raises(ModelUnavailableError):
        ResNet50InferenceService(model_path="nonexistent_checkpoint.keras")


def test_prediction_collapse_detection():
    """Verify collapse diagnostic accurately flags severe class bias or constant probabilities."""
    collapsed_probs = np.array([0.001] * 20)
    diag = detect_prediction_collapse(collapsed_probs, threshold=0.5)
    assert diag["collapsed"] is True
    
    healthy_probs = np.array([0.1] * 10 + [0.9] * 10)
    diag_h = detect_prediction_collapse(healthy_probs, threshold=0.5)
    assert diag_h["collapsed"] is False


def test_compute_fft_spectrum():
    """Verify 2D FFT computation."""
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    spectrum, shift = compute_fft_spectrum(img)
    assert spectrum.shape == (32, 32)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0


def test_extract_artifact_features():
    """Verify extraction of spectral and sensor shot noise artifact metrics."""
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    features = extract_artifact_features(img)
    assert "magnitude_spectrum" in features
    assert "high_to_low_ratio" in features
    assert "spectral_peak_ratio" in features
    assert "noise_residual_std" in features
