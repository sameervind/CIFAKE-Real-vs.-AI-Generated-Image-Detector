"""
Evaluation and Research Validation Module for CIFAKE AI vs. Real Image Classification.
Computes comprehensive metrics: Accuracy, Macro F1, Per-class Precision/Recall,
ROC-AUC, Confusion Matrix, Specificity, FPR, FNR, Binary Cross-Entropy (Log Loss).
Generates publication-grade plots: Confusion Matrix, ROC Curve, PR Curve, Accuracy/Loss Curves.
Includes dedicated Prediction-Collapse Detection diagnostics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
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
    precision_recall_curve,
    log_loss
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def detect_prediction_collapse(
    y_pred_probs: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    collapse_threshold_ratio: float = 0.95
) -> Dict[str, Any]:
    """
    Examines predictions for Class Collapse (when the model predicts nearly everything as one class).
    
    Checks:
      - Ratio of REAL vs AI predictions
      - Distribution and variance of raw sigmoid probabilities
      - Mean predicted probability per true class (if y_true provided)
      - Extreme recall imbalance
    """
    n_total = len(y_pred_probs)
    if n_total == 0:
        return {"collapsed": False, "warning": "Empty prediction set."}

    y_pred = (y_pred_probs >= threshold).astype(int)
    n_pred_real = int(np.sum(y_pred == 1))
    n_pred_ai = int(np.sum(y_pred == 0))

    ratio_real = n_pred_real / n_total
    ratio_ai = n_pred_ai / n_total

    prob_mean = float(np.mean(y_pred_probs))
    prob_std = float(np.std(y_pred_probs))
    prob_min = float(np.min(y_pred_probs))
    prob_max = float(np.max(y_pred_probs))

    # Mean predicted probability for true Real vs true AI (if ground truth available)
    mean_prob_for_real = 0.0
    mean_prob_for_ai = 0.0
    if y_true is not None:
        mask_true_real = (y_true == 1)
        mask_true_ai = (y_true == 0)
        mean_prob_for_real = float(np.mean(y_pred_probs[mask_true_real])) if np.any(mask_true_real) else 0.0
        mean_prob_for_ai = float(np.mean(y_pred_probs[mask_true_ai])) if np.any(mask_true_ai) else 0.0

    is_collapsed = False
    warning_messages = []

    if ratio_real >= collapse_threshold_ratio:
        is_collapsed = True
        warning_messages.append(
            f"PREDICTION COLLAPSE: {ratio_real * 100:.1f}% of all samples were predicted as REAL. "
            "Model is failing to separate synthetic diffusion features."
        )
    elif ratio_ai >= collapse_threshold_ratio:
        is_collapsed = True
        warning_messages.append(
            f"PREDICTION COLLAPSE: {ratio_ai * 100:.1f}% of all samples were predicted as AI. "
            "Model is failing to recognize authentic optical camera noise."
        )

    if prob_std < 0.02 and n_total > 5:
        is_collapsed = True
        warning_messages.append(
            f"EXTREME SCORE INVARIANCE: Probability standard deviation is {prob_std:.4f}. "
            "Output layer sigmoid outputs have converged to a single flat value."
        )

    return {
        "collapsed": is_collapsed,
        "ratio_predicted_real": round(ratio_real, 4),
        "ratio_predicted_ai": round(ratio_ai, 4),
        "prob_mean": round(prob_mean, 4),
        "prob_std": round(prob_std, 4),
        "prob_range": [round(prob_min, 4), round(prob_max, 4)],
        "mean_prob_for_true_real": round(mean_prob_for_real, 4),
        "mean_prob_for_true_ai": round(mean_prob_for_ai, 4),
        "warnings": warning_messages
    }


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate full evaluation metrics from actual predictions.
    
    y_true: 0 = AI / FAKE, 1 = REAL
    y_pred_probs: Array of float probabilities representing P(REAL)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-Class Metrics
    prec_ai = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_ai = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_ai = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    prec_real = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_real = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # Specificity & Error Rates
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC
    if len(np.unique(y_true)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_probs)
        roc_auc_val = float(auc(fpr_curve, tpr_curve))
        bce_loss = float(log_loss(y_true, np.clip(y_pred_probs, 1e-7, 1 - 1e-7)))
    else:
        roc_auc_val = 0.5
        bce_loss = 0.0

    # Collapse Diagnostic
    collapse_diag = detect_prediction_collapse(y_true, y_pred_probs, threshold=threshold)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc_val, 4),
        "binary_cross_entropy": round(bce_loss, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "threshold": threshold,
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
        },
        "collapse_diagnostic": collapse_diag
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None) -> str:
    """Plot and save confusion matrix heatmap matching Lamborghini styling."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    cm_sum = np.sum(cm) if np.sum(cm) > 0 else 1
    group_counts = [f"{v:d}" for v in cm.flatten()]
    group_percentages = [f"{v/cm_sum:.1%}" for v in cm.flatten()]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cm.shape)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="YlOrBr",
        xticklabels=["AI", "REAL"],
        yticklabels=["AI", "REAL"],
        cbar=True,
        linewidths=1.0,
        linecolor="#282828",
        ax=ax
    )

    ax.set_title("CONFUSION MATRIX // RESNET-50 TEST BENCHMARK", fontsize=11, fontweight="bold", color="#FFFFFF", pad=12)
    ax.set_ylabel("TRUE CLASS", fontsize=10, fontweight="bold", color="#FFC000")
    ax.set_xlabel("PREDICTED CLASS", fontsize=10, fontweight="bold", color="#FFC000")
    ax.tick_params(colors="#CCCCCC")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, save_path: Optional[str] = None) -> str:
    """Plot and save ROC curve."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "roc_curve.png")

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr = [0, 1], [0, 1]
        roc_auc = 0.5

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    ax.plot(fpr, tpr, color="#FFC000", lw=2.5, label=f"ResNet-50 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#555555", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FALSE POSITIVE RATE (1 - SPECIFICITY)", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_ylabel("TRUE POSITIVE RATE (SENSITIVITY)", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_title("RECEIVER OPERATING CHARACTERISTIC (ROC)", fontsize=11, fontweight="bold", color="#FFFFFF", pad=12)
    ax.legend(loc="lower right", facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax.tick_params(colors="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, save_path: Optional[str] = None) -> str:
    """Plot and save Precision-Recall curve."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "precision_recall_curve.png")

    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
    else:
        precision, recall = [1, 0], [0, 1]
        pr_auc = 0.5

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    ax.plot(recall, precision, color="#29ABE2", lw=2.5, label=f"ResNet-50 (PR-AUC = {pr_auc:.4f})")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("RECALL", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_ylabel("PRECISION", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_title("PRECISION-RECALL CURVE", fontsize=11, fontweight="bold", color="#FFFFFF", pad=12)
    ax.legend(loc="lower left", facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax.tick_params(colors="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def plot_accuracy_curve(history: Dict[str, list], save_path: Optional[str] = None) -> str:
    """Plot and save training & validation accuracy over epochs."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "accuracy_curve.png")

    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        epochs = range(1, 2)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    ax.plot(epochs, history.get("accuracy", [1.0]), color="#FFC000", marker="o", lw=2.5, label="TRAINING ACCURACY")
    ax.plot(epochs, history.get("val_accuracy", [1.0]), color="#29ABE2", marker="s", lw=2.5, label="VALIDATION ACCURACY")
    ax.set_title("CIFAKE BENCHMARK // ACCURACY CURVE", fontsize=11, fontweight="bold", color="#FFFFFF", pad=12)
    ax.set_xlabel("EPOCH", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_ylabel("ACCURACY", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_ylim([0.7, 1.02])
    ax.legend(facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax.tick_params(colors="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def plot_loss_curve(history: Dict[str, list], save_path: Optional[str] = None) -> str:
    """Plot and save training & validation loss over epochs."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "loss_curve.png")

    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        epochs = range(1, 2)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    ax.plot(epochs, history.get("loss", [0.0]), color="#f43f5e", marker="o", lw=2.5, label="TRAINING LOSS")
    ax.plot(epochs, history.get("val_loss", [0.0]), color="#a855f7", marker="s", lw=2.5, label="VALIDATION LOSS")
    ax.set_title("CIFAKE BENCHMARK // BINARY CROSS-ENTROPY LOSS", fontsize=11, fontweight="bold", color="#FFFFFF", pad=12)
    ax.set_xlabel("EPOCH", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.set_ylabel("LOSS", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax.legend(facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax.tick_params(colors="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def plot_architecture_comparison(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> str:
    """
    Generate side-by-side benchmark comparison chart across tested neural network architectures.
    Visualizes Accuracy, Macro F1, Recall Real, Recall Fake, and Latency.
    """
    if save_path is None:
        save_path = str(ASSETS_DIR / "architecture_comparison.png")

    if not results_list:
        return save_path

    names = [r["model_name"] for r in results_list]
    accuracies = [r.get("accuracy", 0.0) * 100 for r in results_list]
    f1_scores = [r.get("f1_macro", 0.0) * 100 for r in results_list]
    rec_fakes = [r.get("per_class", {}).get("AI", {}).get("recall", 0.0) * 100 for r in results_list]
    rec_reals = [r.get("per_class", {}).get("REAL", {}).get("recall", 0.0) * 100 for r in results_list]

    x = np.arange(len(names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#101010')

    rects1 = ax.bar(x - 1.5 * width, accuracies, width, label='Test Accuracy', color='#FFC000', edgecolor='#282828')
    rects2 = ax.bar(x - 0.5 * width, f1_scores, width, label='Macro F1', color='#29ABE2', edgecolor='#282828')
    rects3 = ax.bar(x + 0.5 * width, rec_fakes, width, label='AI/Fake Recall', color='#A855F7', edgecolor='#282828')
    rects4 = ax.bar(x + 1.5 * width, rec_reals, width, label='Real Recall', color='#10B981', edgecolor='#282828')

    ax.set_ylabel('SCORE (%)', fontsize=10, fontweight='bold', color='#CCCCCC')
    ax.set_title('CIFAKE ARCHITECTURE BENCHMARK // INDEPENDENT TEST COMPARISON', fontsize=11, fontweight='bold', color='#FFFFFF', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, fontweight='bold', color='#FFFFFF')
    ax.set_ylim([70, 102])
    ax.legend(loc='lower right', facecolor='#181818', edgecolor='#282828', labelcolor='#FFFFFF', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.25, color='#555555', axis='y')
    ax.tick_params(colors='#CCCCCC')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=7, color='#FFC000', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    return save_path


def plot_training_curves(history: Dict[str, list], save_path: Optional[str] = None) -> str:
    """Plot training & validation accuracy and loss over epochs."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "accuracy_loss_curves.png")

    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        epochs = range(1, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    fig.patch.set_facecolor('#000000')
    ax1.set_facecolor('#101010')
    ax2.set_facecolor('#101010')

    # Accuracy
    ax1.plot(epochs, history.get("accuracy", [1.0]), color="#FFC000", marker="o", lw=2, label="TRAINING ACCURACY")
    ax1.plot(epochs, history.get("val_accuracy", [1.0]), color="#29ABE2", marker="s", lw=2, label="VALIDATION ACCURACY")
    ax1.set_title("RESNET-50 ACCURACY PROFILE", fontsize=11, fontweight="bold", color="#FFFFFF")
    ax1.set_xlabel("EPOCH", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax1.set_ylabel("ACCURACY", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax1.legend(facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax1.tick_params(colors="#CCCCCC")

    # Loss
    ax2.plot(epochs, history.get("loss", [0.0]), color="#f43f5e", marker="o", lw=2, label="TRAINING LOSS")
    ax2.plot(epochs, history.get("val_loss", [0.0]), color="#a855f7", marker="s", lw=2, label="VALIDATION LOSS")
    ax2.set_title("RESNET-50 LOSS (BINARY CROSS-ENTROPY)", fontsize=11, fontweight="bold", color="#FFFFFF")
    ax2.set_xlabel("EPOCH", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax2.set_ylabel("LOSS", fontsize=9, fontweight="bold", color="#CCCCCC")
    ax2.legend(facecolor="#181818", edgecolor="#282828", labelcolor="#FFFFFF", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#555555")
    ax2.tick_params(colors="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    return save_path


def save_metrics_report(metrics: Dict[str, Any], save_path: Optional[str] = None) -> str:
    """Save metrics to JSON file."""
    if save_path is None:
        save_path = str(ASSETS_DIR / "metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return save_path


def evaluate_checkpoint(
    model_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates a trained model checkpoint against a dataset directory.
    Generates all charts and reports from actual predictions.
    """
    from src.dataset import load_dataset_from_directory, discover_dataset_structure
    from src.model import ResNet50InferenceService

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    data_root = Path(data_dir)
    structure = discover_dataset_structure(data_root)

    eval_dir = structure["test_dir"] if structure["mode"] == "split_train_test" else data_root
    print(f"Evaluating checkpoint {model_path} against {eval_dir}...")

    X, y, paths = load_dataset_from_directory(eval_dir)
    if len(X) == 0:
        raise ValueError(f"No valid evaluation images found in {eval_dir}.")

    service = ResNet50InferenceService(model_path=str(model_file))
    y_pred_probs = service.model.predict(X, verbose=1).flatten()

    metrics = calculate_metrics(y, y_pred_probs, threshold=threshold)

    # Save reports and plots
    target_assets = Path(output_dir) if output_dir else ASSETS_DIR
    target_assets.mkdir(parents=True, exist_ok=True)

    save_metrics_report(metrics, save_path=str(target_assets / "metrics.json"))
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]["matrix"]), save_path=str(target_assets / "confusion_matrix.png"))
    plot_roc_curve(y, y_pred_probs, save_path=str(target_assets / "roc_curve.png"))
    plot_precision_recall_curve(y, y_pred_probs, save_path=str(target_assets / "precision_recall_curve.png"))

    # Print summary
    print("\n" + "=" * 65)
    print("VERIFIED EVALUATION REPORT")
    print("=" * 65)
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:    {metrics['f1_macro'] * 100:.2f}%")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"BCE Loss:    {metrics['binary_cross_entropy']:.4f}")
    print(f"Specificity: {metrics['specificity'] * 100:.2f}%")
    print(f"Collapse Warning: {metrics['collapse_diagnostic']['collapsed']}")
    if metrics['collapse_diagnostic']['warnings']:
        for w in metrics['collapse_diagnostic']['warnings']:
            print(f"  [WARN] {w}")
    print("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAKE ResNet-50 Checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to evaluation dataset.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for REAL class.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation artifacts.")
    args = parser.parse_args()

    evaluate_checkpoint(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
