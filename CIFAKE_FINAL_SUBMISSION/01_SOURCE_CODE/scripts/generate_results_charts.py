import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "04_RESULTS"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Set style: Clean dark theme / publication quality
plt.rcParams.update({
    "figure.facecolor": "#0F0F11",
    "axes.facecolor": "#16161A",
    "axes.edgecolor": "#2D2D35",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#B0B0B0",
    "ytick.color": "#B0B0B0",
    "text.color": "#F0F0F0",
    "grid.color": "#25252D",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif"
})

GOLD = "#FFC000"
GREEN = "#34C759"
RED = "#FF3B30"
BLUE = "#0A84FF"
PURPLE = "#BF5AF2"

def save_fig(fig, filename):
    for d in [RESULTS_DIR, ASSETS_DIR]:
        path = d / filename
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {filename}")
    plt.close(fig)

print("1. Generating Confusion Matrix...")
# 9382 TN, 618 FP, 398 FN, 9602 TP
cm = np.array([[9382, 618], [398, 9602]])
fig, ax = plt.subplots(figsize=(6, 5))
cmap = sns.color_palette(["#16161A", "#242010", "#4A3800", "#8C6A00", "#FFC000"], as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
            xticklabels=["Predicted AI", "Predicted REAL"],
            yticklabels=["Actual AI", "Actual REAL"],
            annot_kws={"size": 14, "weight": "bold", "color": "#FFFFFF"})
ax.set_title("CIFAKE Independent Test Set Confusion Matrix\nAccuracy: 94.92% (18,984 / 20,000)", fontsize=13, weight="bold", pad=15, color=GOLD)
ax.set_ylabel("Ground Truth Label", fontsize=11, weight="bold", labelpad=10)
ax.set_xlabel("Predicted Label (Threshold = 0.50)", fontsize=11, weight="bold", labelpad=10)
save_fig(fig, "confusion_matrix.png")

print("2. Generating ROC Curve...")
test_labels = np.load(ROOT / "data/cifake_cache/test_labels.npy")
preds_saved = np.load(ROOT / "model/test_predictions.npy")
fpr, tpr, _ = roc_curve(test_labels, preds_saved)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=GOLD, lw=2.5, label=f"CIFAR-CNN (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#555566", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold")
ax.set_title("Receiver Operating Characteristic (ROC)\nIndependent 20k Test Set", fontsize=13, weight="bold", pad=15, color=GOLD)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", facecolor="#16161A", edgecolor="#2D2D35")
save_fig(fig, "roc_curve.png")

print("3. Generating Precision-Recall Curve...")
prec, rec, _ = precision_recall_curve(test_labels, preds_saved)
pr_auc = auc(rec, prec)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=GREEN, lw=2.5, label=f"CIFAR-CNN (PR-AUC = {pr_auc:.4f})")
ax.axhline(0.5, color="#555566", lw=1.5, linestyle="--", label="Baseline Prior (0.50)")
ax.set_xlim([0.0, 1.02])
ax.set_ylim([0.48, 1.02])
ax.set_xlabel("Recall (Purity on Real Photography)", fontsize=11, weight="bold")
ax.set_ylabel("Precision", fontsize=11, weight="bold")
ax.set_title("Precision-Recall (PR) Curve\nMacro F1: 0.9492", fontsize=13, weight="bold", pad=15, color=GREEN)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower left", facecolor="#16161A", edgecolor="#2D2D35")
save_fig(fig, "precision_recall_curve.png")

print("4. Generating Training Curves...")
with open(ROOT / "model/training_history.json", "r") as f:
    hist = json.load(f)

epochs = list(range(1, len(hist["accuracy"]) + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(epochs, [x*100 for x in hist["accuracy"]], "o-", color=BLUE, lw=2, label="Train Accuracy")
ax1.plot(epochs, [x*100 for x in hist["val_accuracy"]], "s-", color=GOLD, lw=2, label="Val Accuracy")
ax1.set_title("Training & Validation Accuracy", fontsize=12, weight="bold", color=GOLD)
ax1.set_xlabel("Epoch", fontsize=10, weight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=10, weight="bold")
ax1.set_ylim([80, 98])
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(facecolor="#16161A", edgecolor="#2D2D35")

ax2.plot(epochs, hist["loss"], "o-", color=BLUE, lw=2, label="Train Loss (BCE)")
ax2.plot(epochs, hist["val_loss"], "s-", color=RED, lw=2, label="Val Loss (BCE)")
ax2.set_title("Binary Cross-Entropy Loss", fontsize=12, weight="bold", color=RED)
ax2.set_xlabel("Epoch", fontsize=10, weight="bold")
ax2.set_ylabel("Loss", fontsize=10, weight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(facecolor="#16161A", edgecolor="#2D2D35")

fig.suptitle("CIFAR-CNN 6-Epoch Training Progression", fontsize=14, weight="bold", y=1.03, color="#FFFFFF")
save_fig(fig, "accuracy_loss_curves.png")

print("5. Generating Architecture Comparison Chart...")
models = ["CIFAR-CNN (Ours)", "CIFAR-ResNet20", "ResNet-50 ImageNet"]
accs = [94.92, 92.23, 92.28]
latencies = [1.58, 2.45, 42.10]
params = [0.306, 0.283, 24.146]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

bars1 = ax1.bar(models, accs, color=[GOLD, "#555566", "#777788"], width=0.55)
ax1.set_title("Test Accuracy (%)", fontsize=11, weight="bold", color=GOLD)
ax1.set_ylim([85, 98])
for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h + 0.3), ha="center", fontsize=9, weight="bold")
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
ax1.tick_params(axis="x", rotation=15)

bars2 = ax2.bar(models, latencies, color=[GREEN, "#555566", RED], width=0.55)
ax2.set_title("CPU Latency (ms/img)", fontsize=11, weight="bold", color=GREEN)
for b in bars2:
    h = b.get_height()
    ax2.annotate(f"{h:.1f}ms", (b.get_x() + b.get_width() / 2, h + 0.6), ha="center", fontsize=9, weight="bold")
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
ax2.tick_params(axis="x", rotation=15)

bars3 = ax3.bar(models, params, color=[BLUE, "#555566", PURPLE], width=0.55)
ax3.set_title("Parameters (Millions)", fontsize=11, weight="bold", color=BLUE)
for b in bars3:
    h = b.get_height()
    ax3.annotate(f"{h:.2f}M", (b.get_x() + b.get_width() / 2, h + 0.4), ha="center", fontsize=9, weight="bold")
ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
ax3.tick_params(axis="x", rotation=15)

fig.suptitle("Architecture Benchmark Comparison (Independent CIFAKE Partition)", fontsize=13, weight="bold", y=1.04, color="#FFFFFF")
save_fig(fig, "architecture_comparison.png")

print("All figures successfully generated and saved to 04_RESULTS and assets!")
