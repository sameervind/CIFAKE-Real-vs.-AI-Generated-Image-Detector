"""
Evaluation and visualization module for AI vs. Real image classification.
Computes confusion matrix, classification metrics, and generates publication-grade plots.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    y_true: 0 = AI, 1 = REAL
    y_pred_probs: probability of REAL (class 1)
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_ai = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    prec_real = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_ai = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_real = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_ai = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc_val = float(auc(fpr, tpr))
    
    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc_val, 4),
        "per_class": {
            "AI": {
                "precision": round(prec_ai, 4),
                "recall": round(rec_ai, 4),
                "f1_score": round(f1_ai, 4),
                "support": int(np.sum(y_true == 0))
            },
            "REAL": {
                "precision": round(prec_real, 4),
                "recall": round(rec_real, 4),
                "f1_score": round(f1_real, 4),
                "support": int(np.sum(y_true == 1))
            }
        },
        "confusion_matrix": {
            "true_ai_pred_ai": int(tn),
            "true_ai_pred_real": int(fp),
            "true_real_pred_ai": int(fn),
            "true_real_pred_real": int(tp),
            "matrix": cm.tolist()
        }
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None) -> str:
    """
    Plot and save confusion matrix heatmap matching the notebook styling.
    """
    if save_path is None:
        save_path = str(ASSETS_DIR / "confusion_matrix.png")
        
    plt.figure(figsize=(7, 6), dpi=150)
    sns.set_theme(style="white")
    
    # Annotations with counts and percentages
    cm_sum = np.sum(cm)
    group_counts = [f"{v:d}" for v in cm.flatten()]
    group_percentages = [f"{v/cm_sum:.1%}" for v in cm.flatten()]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    ax = sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["AI", "REAL"],
        yticklabels=["AI", "REAL"],
        cbar=True,
        linewidths=1.5,
        linecolor="#e0e0e0"
    )
    plt.title("CIFAKE ResNet50 - Test Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("True Class", fontsize=12, fontweight="semibold")
    plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(history: Dict[str, list], save_path: Optional[str] = None) -> str:
    """
    Plot training & validation accuracy and loss over epochs.
    """
    if save_path is None:
        save_path = str(ASSETS_DIR / "accuracy_loss_curves.png")
        
    epochs = range(1, len(history.get("accuracy", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Accuracy Plot
    ax1.plot(epochs, history.get("accuracy", []), "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, history.get("val_accuracy", []), "g-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("ResNet50 Model Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0.7, 1.0])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")
    
    # Loss Plot
    ax2.plot(epochs, history.get("loss", []), "r-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history.get("val_loss", []), "m-s", label="Validation Loss", linewidth=2)
    ax2.set_title("ResNet50 Model Loss (Binary Cross-Entropy)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, save_path: Optional[str] = None) -> str:
    """
    Plot and save ROC curve.
    """
    if save_path is None:
        save_path = str(ASSETS_DIR / "roc_curve.png")
        
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ResNet50 (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def save_metrics_report(metrics: Dict[str, Any], save_path: Optional[str] = None) -> str:
    """Save metrics to JSON file."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return save_path
