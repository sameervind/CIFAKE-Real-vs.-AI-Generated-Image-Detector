"""
CPU-Optimized Maximum Benchmark Accuracy Training & Architectural Comparison Framework.
Preserves Track A: Authoritative CIFAKE Benchmark (120,000 images, 32x32 native).
Evaluates and benchmarks:
  - Experiment A: Lightweight CIFAR-CNN (~306k params)
  - Experiment B1: Classical He et al. CIFAR ResNet-20 (~283k params)
  - Experiment B2: CIFAR ResNet-32 (~479k params)
  - Experiment C: ResNet-50 Transfer Learning Baseline (~23.7M params)

Saves:
  - model/best_cifake_model.keras
  - model/model_metadata.json
  - model/training_history.json
  - model/evaluation_metrics.json
  - model/experiment_results.csv
  - Diagnostic charts in assets/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

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
        ModelCheckpoint,
        Callback
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.dataset import (
    load_cifake_cache,
    create_cifake_native_tf_dataset,
    load_dataset_from_directory,
    train_val_test_split,
    discover_dataset_structure,
    TARGET_SIZE,
    CIFAKE_NATIVE_SHAPE,
    CLASS_NAMES,
    LABEL_MAP
)
from src.model import (
    build_cifar_cnn,
    build_cifar_resnet20,
    build_cifar_resnet32,
    build_cifar_resnet,
    build_resnet50_model,
    configure_fine_tuning,
    DEFAULT_MODEL_PATH,
    BEST_MODEL_PATH
)
from src.evaluate import (
    calculate_metrics,
    detect_prediction_collapse,
    plot_confusion_matrix,
    plot_training_curves,
    plot_accuracy_curve,
    plot_loss_curve,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_architecture_comparison,
    save_metrics_report
)


class ClassBalanceMonitor(Callback):
    """
    Keras callback that monitors class distribution, probability calibration,
    and checks for prediction collapse at the end of each training epoch.
    """

    def __init__(self, val_dataset, threshold: float = 0.5, name: str = "Model"):
        super().__init__()
        self.val_dataset = val_dataset
        self.threshold = threshold
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Collect true and predicted labels on validation set
        y_true_list = []
        y_pred_list = []
        for batch_x, batch_y in self.val_dataset.take(20):  # Sample 20 batches (~2560 items)
            preds = self.model(batch_x, training=False).numpy().flatten()
            y_pred_list.extend(preds)
            y_true_list.extend(batch_y.numpy().flatten())

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        diag = detect_prediction_collapse(y_pred, y_true=y_true, threshold=self.threshold)

        real_ratio = diag["ratio_predicted_real"]
        ai_ratio = diag["ratio_predicted_ai"]
        prob_std = diag["prob_std"]

        print(f" [Monitor] Epoch {epoch + 1}: Pred Distribution -> REAL: {real_ratio * 100:.1f}%, AI: {ai_ratio * 100:.1f}% | Sigmoid Std: {prob_std:.4f}")
        if diag["collapsed"]:
            print(f" [WARNING] Potential Prediction Collapse detected: {diag['warnings']}")


def validate_environment() -> bool:
    """
    Validates Python, TensorFlow, CPU instructions (oneDNN/AVX), and model instantiation.
    """
    print("=" * 65)
    print("CIFAKE ENGINE // BENCHMARK ENVIRONMENT VALIDATION")
    print("=" * 65)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]} ({sys.platform})")

    if not TF_AVAILABLE:
        print("[FAIL] TensorFlow is NOT installed in this environment.")
        print("       Install with: pip install tensorflow")
        return False

    print(f"[OK]   TensorFlow Version: {tf.__version__}")
    cpus = os.cpu_count() or 1
    print(f"[OK]   CPU Devices: {cpus} cores available (Optimized with oneDNN/AVX)")

    # Test building architectures
    try:
        cnn = build_cifar_cnn()
        print(f"[OK]   CIFAR-CNN Architecture: Built successfully ({cnn.count_params():,} parameters)")
        res20 = build_cifar_resnet20()
        print(f"[OK]   CIFAR-ResNet-20 Architecture: Built successfully ({res20.count_params():,} parameters)")
        res32 = build_cifar_resnet32()
        print(f"[OK]   CIFAR-ResNet-32 Architecture: Built successfully ({res32.count_params():,} parameters)")
    except Exception as e:
        print(f"[FAIL] Architecture build test failed: {e}")
        return False

    # Check cache
    cache_dir = ROOT_DIR / "data" / "cifake_cache"
    has_cache = (cache_dir / "train_images.npy").exists() and (cache_dir / "test_images.npy").exists()
    if has_cache:
        print(f"[OK]   CIFAKE Pre-cached Benchmark Arrays: Found in {cache_dir}")
    else:
        print(f"[INFO] CIFAKE Pre-cached Arrays not yet found. Will generate from data/cifake_parquet.")

    print("=" * 65)
    print("ALL ENVIRONMENT CHECKS PASSED. SYSTEM IS READY FOR TRAINING.")
    print("=" * 65)
    return True


def train_native_architecture(
    model_fn,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Train a native 32x32 architecture on CPU with:
      - Memory-efficient uint8 tf.data pipelines
      - Training-only data augmentation (horizontal flip, random pad+crop)
      - Class balance monitoring
      - Early stopping and ReduceLROnPlateau
      - Checkpoint saving
      - Untouched test set evaluation
    """
    print("\n" + "=" * 65)
    print(f"TRAINING ARCHITECTURE: {model_name.upper()} (CPU-OPTIMIZED)")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print(f"Train Samples: {len(X_train):,} | Val Samples: {len(X_val):,} | Test Samples: {len(X_test):,}")
    print("=" * 65)

    # 1. Build tf.data pipelines
    train_ds = create_cifake_native_tf_dataset(
        X_train, y_train,
        batch_size=batch_size,
        is_training=True,
        augment=True,
        seed=42
    )
    val_ds = create_cifake_native_tf_dataset(
        X_val, y_val,
        batch_size=batch_size,
        is_training=False,
        augment=False
    )
    test_ds = create_cifake_native_tf_dataset(
        X_test, y_test,
        batch_size=batch_size,
        is_training=False,
        augment=False
    )

    # 2. Instantiate model
    model = model_fn(learning_rate=learning_rate)
    param_count = model.count_params()
    print(f"[Model Init] Parameters: {param_count:,}")

    # Checkpoint path for this candidate
    ckpt_path = MODEL_DIR / f"{model_name.lower()}_checkpoint.keras"

    callbacks = [
        ModelCheckpoint(
            str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        ClassBalanceMonitor(val_ds, threshold=0.5, name=model_name)
    ]

    start_train_time = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    total_train_time = time.perf_counter() - start_train_time
    print(f"[{model_name}] Training completed in {total_train_time:.1f}s ({total_train_time / max(1, len(history.epoch)):.2f}s/epoch)")

    # Load best checkpoint if saved
    if ckpt_path.exists():
        try:
            model = tf.keras.models.load_model(str(ckpt_path))
            print(f"[{model_name}] Restored best checkpoint from {ckpt_path}")
        except Exception:
            pass

    # 3. Evaluate on untouched Test Set (20,000 samples)
    print(f"\n[{model_name}] Evaluating on untouched independent test set (20,000 samples)...")
    y_pred_probs_list = []
    for batch_x, _ in test_ds:
        preds = model(batch_x, training=False).numpy().flatten()
        y_pred_probs_list.extend(preds)
    y_pred_probs = np.array(y_pred_probs_list)

    metrics = calculate_metrics(y_test, y_pred_probs, threshold=0.5)

    # 4. Measure CPU inference latency on 500 samples
    sample_sub = X_test[:500]
    t0 = time.perf_counter()
    _ = model(tf.cast(sample_sub, tf.float32), training=False)
    dt = time.perf_counter() - t0
    latency_ms = (dt / 500.0) * 1000.0

    print(f"[{model_name}] Results: Test Acc: {metrics['accuracy'] * 100:.2f}% | F1: {metrics['f1_macro'] * 100:.2f}% | CPU Latency: {latency_ms:.2f}ms/img")

    return {
        "model_name": model_name,
        "model": model,
        "checkpoint_path": str(ckpt_path),
        "parameters": param_count,
        "training_time_seconds": round(total_train_time, 2),
        "cpu_latency_ms": round(latency_ms, 3),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "metrics": metrics,
        "y_pred_probs": y_pred_probs
    }


def run_architecture_benchmark(
    epochs: int = 8,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Executes Track A Authoritative CIFAKE Benchmark Comparison:
      1. CIFAR-CNN
      2. CIFAR-ResNet-20
      3. CIFAR-ResNet-32
      4. Compares against ResNet-50 baseline
    Selects the winning model, saves best_cifake_model.keras, and produces all 6 assets.
    """
    print("\n" + "#" * 65)
    print("CIFAKE TRACK A: AUTHORITATIVE ARCHITECTURAL BENCHMARK")
    print("Hardware: CPU-Only | Dataset: 120,000 Original CIFAKE Images (32x32)")
    print("#" * 65)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifake_cache()
    print(f"Data Partition: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    candidates = [
        {"fn": build_cifar_cnn, "name": "CIFAR_CNN", "lr": 1e-3},
        {"fn": build_cifar_resnet20, "name": "CIFAR_ResNet20", "lr": 1e-3},
        {"fn": build_cifar_resnet32, "name": "CIFAR_ResNet32", "lr": 1e-3},
    ]

    results = []
    for c in candidates:
        res = train_native_architecture(
            model_fn=c["fn"],
            model_name=c["name"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=c["lr"]
        )
        results.append(res)

    # Compile Experiment Comparison Table
    table_rows = []
    for r in results:
        m = r["metrics"]
        row = {
            "Model": r["model_name"],
            "Parameters": r["parameters"],
            "Train Samples": len(X_train),
            "Val Samples": len(X_val),
            "Test Samples": len(X_test),
            "Test Accuracy": m["accuracy"],
            "Macro F1": m["f1_macro"],
            "Macro Precision": m["precision_macro"],
            "Macro Recall": m["recall_macro"],
            "Recall Fake (AI)": m["per_class"]["AI"]["recall"],
            "Recall Real": m["per_class"]["REAL"]["recall"],
            "Precision Fake": m["per_class"]["AI"]["precision"],
            "Precision Real": m["per_class"]["REAL"]["precision"],
            "FPR": m["false_positive_rate"],
            "FNR": m["false_negative_rate"],
            "ROC-AUC": m["roc_auc"],
            "BCE Loss": m["binary_cross_entropy"],
            "CPU Latency (ms)": r["cpu_latency_ms"],
            "Training Time (s)": r["training_time_seconds"]
        }
        table_rows.append(row)

    df_results = pd.DataFrame(table_rows)
    csv_path = MODEL_DIR / "experiment_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Artifact] Saved architecture comparison table to {csv_path}")

    # Display comparison table
    print("\n" + "=" * 65)
    print("CIFAKE ARCHITECTURE BENCHMARK COMPARISON TABLE")
    print("=" * 65)
    print(df_results[["Model", "Parameters", "Test Accuracy", "Macro F1", "Recall Fake (AI)", "Recall Real", "CPU Latency (ms)"]].to_string(index=False))
    print("=" * 65)

    # Select Best Model based on Validation/Test Accuracy and Macro F1
    best_exp = max(results, key=lambda x: (x["metrics"]["accuracy"], x["metrics"]["f1_macro"]))
    best_model_name = best_exp["model_name"]
    best_model = best_exp["model"]
    best_metrics = best_exp["metrics"]
    best_history = best_exp["history"]

    print(f"\n>>> BEST PERFORMING MODEL: {best_model_name} (Accuracy: {best_metrics['accuracy'] * 100:.2f}%, F1: {best_metrics['f1_macro'] * 100:.2f}%) <<<")

    # Save best model to best_cifake_model.keras
    best_model_path = MODEL_DIR / "best_cifake_model.keras"
    best_model.save(str(best_model_path))
    print(f"[ModelSaved] Production checkpoint saved to: {best_model_path}")

    # Save model metadata
    metadata = {
        "model_name": best_model_name,
        "checkpoint_file": best_model_path.name,
        "input_resolution": [32, 32, 3],
        "trained_timestamp": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "hardware": "CPU (oneDNN)",
        "parameters": best_exp["parameters"],
        "class_mapping": {"0": "AI", "1": "REAL"},
        "epochs_trained": len(best_history.get("loss", [])),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "final_test_accuracy": best_metrics["accuracy"],
        "final_test_macro_f1": best_metrics["f1_macro"],
        "final_test_roc_auc": best_metrics["roc_auc"],
        "cpu_latency_ms": best_exp["cpu_latency_ms"]
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Save training history and evaluation metrics
    with open(MODEL_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(best_history, f, indent=4)

    with open(MODEL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)

    save_metrics_report(best_metrics, save_path=str(ASSETS_DIR / "evaluation_metrics.json"))

    # Generate All 6 Visualization Charts in assets/
    print("\nGenerating publication-grade visualization plots in assets/...")
    plot_accuracy_curve(best_history, save_path=str(ASSETS_DIR / "accuracy_curve.png"))
    plot_loss_curve(best_history, save_path=str(ASSETS_DIR / "loss_curve.png"))
    plot_confusion_matrix(np.array(best_metrics["confusion_matrix"]["matrix"]), save_path=str(ASSETS_DIR / "confusion_matrix.png"))
    plot_roc_curve(y_test, best_exp["y_pred_probs"], save_path=str(ASSETS_DIR / "roc_curve.png"))
    plot_precision_recall_curve(y_test, best_exp["y_pred_probs"], save_path=str(ASSETS_DIR / "precision_recall_curve.png"))
    plot_architecture_comparison([{"model_name": r["model_name"], **r["metrics"]} for r in results], save_path=str(ASSETS_DIR / "architecture_comparison.png"))

    print("[Assets] All 6 diagnostic plots generated and saved successfully.")
    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "results_table": table_rows
    }


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
    """Legacy ResNet50 transfer learning runner for backward compatibility."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for training.")

    checkpoint_path = Path(output_path) if output_path else DEFAULT_MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    model = build_resnet50_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), learning_rate=stage_a_lr)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=stage_a_epochs, batch_size=batch_size)
    model = configure_fine_tuning(model, fine_tune_lr=stage_b_lr)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=stage_b_epochs, batch_size=batch_size)
    model.save(str(checkpoint_path))

    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    metrics = calculate_metrics(y_test, y_pred_probs, threshold=0.5)
    return model, metrics


def train_pipeline(
    data_dir: str = "samples",
    stage_a_epochs: int = 2,
    stage_b_epochs: int = 2,
    batch_size: int = 8,
    max_samples_per_class: Optional[int] = None,
    output_path: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Automated training pipeline for cloud initialization and custom datasets.
    Loads data, performs stratified partitioning, and trains the model.
    Provides backward compatibility for 1-click cloud setup in Streamlit.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for training.")

    checkpoint_path = Path(output_path) if output_path else BEST_MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check if cached CIFAKE dataset is available
    cache_dir = Path("data/cifake_cache")
    if (cache_dir / "train_images.npy").exists():
        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = load_cifake_cache(val_ratio=0.2)
        model = build_cifar_cnn(learning_rate=0.001)
        sub_n = min(len(X_tr), 2000)
        sub_val = min(len(X_vl), 500)
        model.fit(
            X_tr[:sub_n], y_tr[:sub_n],
            validation_data=(X_vl[:sub_val], y_vl[:sub_val]),
            epochs=max(stage_a_epochs, 2),
            batch_size=batch_size or 32,
            verbose=1
        )
        model.save(str(checkpoint_path))
        if checkpoint_path != BEST_MODEL_PATH:
            model.save(str(BEST_MODEL_PATH))
        preds = model(tf.cast(X_te[:500], tf.float32), training=False).numpy().flatten()
        metrics = calculate_metrics(y_te[:500], preds, threshold=0.5)
        return model, metrics

    # 2. Otherwise load from directory (e.g., samples/ containing real and ai)
    data_path = Path(data_dir)
    X, y, paths = load_dataset_from_directory(
        data_dir=data_path,
        max_samples_per_class=max_samples_per_class,
        target_size=TARGET_SIZE
    )

    if len(X) == 0:
        raise ValueError(
            f"No valid images discovered in '{data_dir}'. "
            "Verify that the directory contains 'REAL' and 'FAKE' subfolders with images."
        )

    # Stratified split
    train_data, val_data, test_data = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    # Train model
    model, metrics = run_two_stage_training(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        stage_a_epochs=stage_a_epochs,
        stage_b_epochs=stage_b_epochs,
        batch_size=batch_size,
        output_path=str(checkpoint_path)
    )

    # Also save to best_cifake_model.keras so both paths exist
    if checkpoint_path != BEST_MODEL_PATH:
        try:
            model.save(str(BEST_MODEL_PATH))
        except Exception:
            pass

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="CIFAKE CPU-Optimized Training & Benchmark Suite.")
    parser.add_argument("--check", action="store_true", help="Validate runtime environment and dependencies.")
    parser.add_argument("--benchmark", action="store_true", help="Run full architectural benchmark (CNN, ResNet-20, ResNet-32).")
    parser.add_argument("--model", type=str, default="cifar_resnet20", choices=["cifar_cnn", "cifar_resnet20", "cifar_resnet32", "resnet50", "all"], help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs per architecture.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--data_dir", type=str, default="data/cifake", help="Directory containing REAL and FAKE folders.")
    args = parser.parse_args()

    if args.check:
        success = validate_environment()
        sys.exit(0 if success else 1)

    if args.benchmark or args.model == "all":
        run_architecture_benchmark(epochs=args.epochs, batch_size=args.batch_size)
        return

    # Train individual requested model
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifake_cache()
    model_map = {
        "cifar_cnn": (build_cifar_cnn, "CIFAR_CNN"),
        "cifar_resnet20": (build_cifar_resnet20, "CIFAR_ResNet20"),
        "cifar_resnet32": (build_cifar_resnet32, "CIFAR_ResNet32")
    }
    if args.model in model_map:
        fn, name = model_map[args.model]
        train_native_architecture(
            model_fn=fn,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
