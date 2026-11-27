"""
Training pipeline for CIFAKE Real vs. AI-Generated Image Classification.
Implements Two-Stage Transfer Learning with ImageNet Pretrained ResNet50:
  - Stage A: Feature extraction with frozen backbone (learning rate = 1e-3).
  - Stage B: Fine-tuning top residual blocks (learning rate = 1e-5).
Saves model checkpoint, metadata, training history, and evaluation metrics.
Strictly requires TensorFlow — no silent fallbacks.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Ensure root workspace is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.dataset import (
    load_dataset_from_directory,
    train_val_test_split,
    discover_dataset_structure,
    create_tf_data_pipeline,
    TARGET_SIZE,
    CLASS_NAMES,
    LABEL_MAP
)
from src.model import (
    build_resnet50_model,
    configure_fine_tuning,
    DEFAULT_MODEL_PATH
)
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    plot_roc_curve,
    save_metrics_report
)


def validate_environment() -> bool:
    """
    Validates Python, TensorFlow, device availability, and model compilation.
    Invoked via `python -m src.train --check`.
    """
    print("=" * 65)
    print("CIFAKE ENGINE // ENVIRONMENT VALIDATION")
    print("=" * 65)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]} ({sys.platform})")
    
    if not TF_AVAILABLE:
        print("[FAIL] TensorFlow is NOT installed in this environment.")
        print("       Install with: pip install tensorflow")
        return False

    print(f"[OK]   TensorFlow Version: {tf.__version__}")
    
    # Check GPU / CPU devices
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[OK]   Compute Device: GPU ({len(gpus)} device(s) found: {gpus})")
    else:
        print("[INFO] Compute Device: CPU (Optimized with oneDNN)")

    # Test ResNet50 model instantiation
    try:
        test_model = build_resnet50_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), freeze_backbone=True)
        print(f"[OK]   ResNet-50 Architecture: Built successfully ({test_model.count_params():,} total parameters)")
    except Exception as e:
        print(f"[FAIL] Could not build ResNet-50 architecture: {e}")
        return False

    # Check directory permissions
    try:
        test_file = MODEL_DIR / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        print(f"[OK]   Model Storage Directory: {MODEL_DIR} (Writable)")
    except Exception as e:
        print(f"[FAIL] Cannot write to model directory {MODEL_DIR}: {e}")
        return False

    print("=" * 65)
    print("ALL ENVIRONMENT CHECKS PASSED. SYSTEM IS READY FOR TRAINING.")
    print("=" * 65)
    return True


def run_two_stage_training(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    stage_a_epochs: int = 5,
    stage_b_epochs: int = 10,
    batch_size: int = 32,
    stage_a_lr: float = 1e-3,
    stage_b_lr: float = 1e-5,
    output_path: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executes Two-Stage ResNet-50 Transfer Learning:
      Stage A: Train custom classification head with frozen ResNet50 backbone.
      Stage B: Fine-tune top residual blocks (conv5_block1_out) with low learning rate.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for training. Install with: pip install tensorflow")

    checkpoint_path = Path(output_path) if output_path else DEFAULT_MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    combined_history: Dict[str, list] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print("\n" + "=" * 65)
    print("STAGE A: FEATURE EXTRACTION (FROZEN RESNET50 BACKBONE)")
    print(f"Initial Learning Rate: {stage_a_lr} | Max Epochs: {stage_a_epochs}")
    print("=" * 65)

    model = build_resnet50_model(
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        learning_rate=stage_a_lr,
        freeze_backbone=True
    )

    stage_a_callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist_a = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=stage_a_epochs,
        batch_size=batch_size,
        callbacks=stage_a_callbacks,
        verbose=1
    )

    for k in combined_history:
        if k in hist_a.history:
            combined_history[k].extend(hist_a.history[k])

    print("\n" + "=" * 65)
    print("STAGE B: FINE-TUNING (UNFROZEN TOP RESIDUAL BLOCKS)")
    print(f"Fine-Tuning Learning Rate: {stage_b_lr} | Max Epochs: {stage_b_epochs}")
    print("=" * 65)

    model = configure_fine_tuning(
        model,
        fine_tune_from_layer="conv5_block1_out",
        fine_tune_lr=stage_b_lr
    )

    stage_b_callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist_b = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=stage_b_epochs,
        batch_size=batch_size,
        callbacks=stage_b_callbacks,
        verbose=1
    )

    for k in combined_history:
        if k in hist_b.history:
            combined_history[k].extend(hist_b.history[k])

    # Save final authoritative model
    model.save(str(checkpoint_path))
    print(f"\n[ModelCheckpoint] Authoritative model saved to: {checkpoint_path}")

    # Evaluate on held-out test partition
    print("\n" + "=" * 65)
    print("EVALUATING ON INDEPENDENT HELD-OUT TEST PARTITION")
    print("=" * 65)
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    metrics = calculate_metrics(y_test, y_pred_probs, threshold=0.5)

    # Save model artifacts & metadata
    metadata = {
        "model_name": "ResNet50_CIFAKE",
        "checkpoint_file": checkpoint_path.name,
        "trained_timestamp": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "input_resolution": list(TARGET_SIZE) + [3],
        "class_mapping": {"0": "AI", "1": "REAL"},
        "stage_a_epochs_completed": len(hist_a.history.get("loss", [])),
        "stage_b_epochs_completed": len(hist_b.history.get("loss", [])),
        "total_epochs": len(combined_history["loss"]),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "final_test_accuracy": metrics["accuracy"],
        "final_test_roc_auc": metrics["roc_auc"]
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(MODEL_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)

    with open(MODEL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    save_metrics_report(metrics)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]["matrix"]))
    plot_training_curves(combined_history)
    plot_roc_curve(y_test, y_pred_probs)

    print("\nTraining & Evaluation Complete:")
    print(f"  Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision:      {metrics['precision_macro'] * 100:.2f}%")
    print(f"  Recall:         {metrics['recall_macro'] * 100:.2f}%")
    print(f"  F1-Score:       {metrics['f1_macro'] * 100:.2f}%")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    return model, metrics


def train_pipeline(
    data_dir: str,
    stage_a_epochs: int = 5,
    stage_b_epochs: int = 10,
    batch_size: int = 32,
    max_samples_per_class: Optional[int] = None,
    output_path: Optional[str] = None
):
    """Universal training pipeline coordinator."""
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required to train the ResNet50 model. "
            "Install with: pip install tensorflow"
        )

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. "
            "Please download or supply the CIFAKE dataset (with REAL and FAKE folders) "
            "before initiating model training."
        )

    print(f"Scanning and loading dataset from: {data_dir}...")
    X, y, paths = load_dataset_from_directory(
        data_path,
        max_samples_per_class=max_samples_per_class,
        target_size=TARGET_SIZE
    )

    if len(X) == 0:
        raise ValueError(
            f"No valid images discovered in '{data_dir}'. "
            "Verify that the directory contains 'REAL' and 'FAKE' subfolders with images."
        )

    n_real = int(np.sum(y == 1))
    n_ai = int(np.sum(y == 0))
    print(f"Dataset successfully loaded: {len(X)} total images (REAL: {n_real}, AI: {n_ai})")

    # Genuine Stratified Partitioning (70% Train, 20% Val, 10% Test)
    train_data, val_data, test_data = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    print(f"Stratified Partitioning: Train={len(train_data[0])}, Val={len(val_data[0])}, Test={len(test_data[0])}")

    return run_two_stage_training(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        stage_a_epochs=stage_a_epochs,
        stage_b_epochs=stage_b_epochs,
        batch_size=batch_size,
        output_path=output_path
    )


def main():
    parser = argparse.ArgumentParser(description="Train CIFAKE AI vs Real Classifier with Two-Stage Transfer Learning.")
    parser.add_argument("--check", action="store_true", help="Validate runtime environment and dependencies.")
    parser.add_argument("--data_dir", type=str, default="data/cifake", help="Directory containing REAL and FAKE folders.")
    parser.add_argument("--epochs_stage_a", type=int, default=5, help="Epochs for Stage A feature extraction.")
    parser.add_argument("--epochs_stage_b", type=int, default=10, help="Epochs for Stage B fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max samples per class (for smoke testing).")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained model (.keras).")
    args = parser.parse_args()

    if args.check:
        success = validate_environment()
        sys.exit(0 if success else 1)

    try:
        train_pipeline(
            data_dir=args.data_dir,
            stage_a_epochs=args.epochs_stage_a,
            stage_b_epochs=args.epochs_stage_b,
            batch_size=args.batch_size,
            max_samples_per_class=args.max_samples,
            output_path=args.output
        )
    except Exception as e:
        print(f"\n[TRAINING FAILED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
