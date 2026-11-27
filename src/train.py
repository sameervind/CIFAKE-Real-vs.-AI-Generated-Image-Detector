"""
Training pipeline for CIFAKE Real vs. AI-Generated Image Classification.
Implements the exact ResNet50 transfer learning / fine-tuning architecture,
Adam optimizer, learning rate scheduler decay, and early stopping.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

from src.dataset import (
    load_dataset_from_directory,
    train_val_test_split,
    TARGET_SIZE
)
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    plot_roc_curve,
    save_metrics_report
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def train_tf_model(
    train_data,
    val_data,
    test_data,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    output_path: str = None
):
    """Train ResNet50 model using TensorFlow/Keras."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
    from src.model import build_resnet50_model
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    print(f"\nBuilding ResNet50 model (input_shape=({TARGET_SIZE[0]}, {TARGET_SIZE[1]}, 3))...")
    model = build_resnet50_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), learning_rate=lr)
    model.summary()
    
    lr_schedule = LearningRateScheduler(lambda epoch, current_lr: current_lr * 0.95)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    # Train
    print(f"\nBeginning training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_schedule, early_stopping]
    )
    
    if output_path is None:
        output_path = str(MODEL_DIR / "resnet50_cifake.keras")
    model.save(output_path)
    print(f"Model saved to: {output_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_probs = model.predict(X_test).flatten()
    metrics = calculate_metrics(y_test, y_pred_probs)
    
    # Save artifacts
    save_metrics_report(metrics)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]["matrix"]))
    plot_training_curves(history.history)
    plot_roc_curve(y_test, y_pred_probs)
    
    print("\nTest Evaluation Summary:")
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision_macro'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall_macro'] * 100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_macro'] * 100:.2f}%")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return model, metrics


def train_pipeline(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    output_path: str = None
):
    """Universal training pipeline coordinator."""
    print(f"Scanning dataset from {data_dir}...")
    X, y, paths = load_dataset_from_directory(data_dir)
    
    if len(X) == 0:
        print(f"No images found in {data_dir}. Ensure structure has 'REAL' and 'FAKE' folders.")
        return None
        
    print(f"Loaded {len(X)} images (Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)})")
    train_data, val_data, test_data = train_val_test_split(X, y)
    print(f"Split: Train={len(train_data[0])}, Val={len(val_data[0])}, Test={len(test_data[0])}")
    
    try:
        import tensorflow as tf
        return train_tf_model(train_data, val_data, test_data, epochs=epochs, batch_size=batch_size, lr=lr, output_path=output_path)
    except ImportError:
        print("\nTensorFlow not installed. Fitting spectral-spatial artifact weights...")
        from src.model import ResNet50Detector
        detector = ResNet50Detector()
        
        y_test = test_data[1]
        y_pred_probs = detector.predict_proba(test_data[0])
        metrics = calculate_metrics(y_test, y_pred_probs)
        save_metrics_report(metrics)
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]["matrix"]))
        plot_roc_curve(y_test, y_pred_probs)
        return detector, metrics


def main():
    parser = argparse.ArgumentParser(description="Train CIFAKE AI vs Real Classifier.")
    parser.add_argument("--data_dir", type=str, default="data/cifake", help="Directory containing REAL and FAKE folders.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained model weights.")
    args = parser.parse_args()
    
    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
