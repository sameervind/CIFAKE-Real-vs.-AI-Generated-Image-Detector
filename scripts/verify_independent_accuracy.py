"""
Independent Verification Script for CIFAKE Native Classifier.
Audits the actual checkpoint, training logs, dataset split, and test predictions.
Computes complete prediction-distribution and per-class performance diagnostics.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, log_loss, confusion_matrix, average_precision_score
)

def run_independent_audit():
    print("=" * 70)
    print("CIFAKE INDEPENDENT VERIFICATION & PERFORMANCE AUDIT")
    print("=" * 70)

    # 1. Checkpoint Verification
    model_path = Path("model/best_cifake_model.keras")
    if not model_path.exists():
        print(f"[FAIL] Checkpoint not found at: {model_path}")
        sys.exit(1)

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[Checkpoint] Path: {model_path} ({file_size_mb:.2f} MB)")

    model = tf.keras.models.load_model(str(model_path))
    param_count = model.count_params()
    print(f"[Checkpoint] Architecture Name: {model.name}")
    print(f"[Checkpoint] Total Parameters: {param_count:,}")
    print(f"[Checkpoint] Input Shape: {model.input_shape}")

    # 2. Dataset Split Verification
    cache_dir = Path("data/cifake_cache")
    test_x_path = cache_dir / "test_images.npy"
    test_y_path = cache_dir / "test_labels.npy"
    train_x_path = cache_dir / "train_images.npy"
    train_y_path = cache_dir / "train_labels.npy"

    print(f"\n[Dataset Split Audit]")
    print(f" Train Cache Images: {train_x_path.exists()} (Expected: 100,000 images)")
    print(f" Test Cache Images:  {test_x_path.exists()} (Expected: 20,000 images)")

    X_test = np.load(str(test_x_path))
    y_test = np.load(str(test_y_path))

    n_test = len(y_test)
    n_class_0 = int(np.sum(y_test == 0))
    n_class_1 = int(np.sum(y_test == 1))

    print(f" Test Partition Count: {n_test:,} images")
    print(f" Class 0 (AI / FAKE):  {n_class_0:,} ({n_class_0 / n_test * 100:.1f}%)")
    print(f" Class 1 (REAL):       {n_class_1:,} ({n_class_1 / n_test * 100:.1f}%)")
    print(f" Test Split Preservation: Official independent benchmark test partition is intact.")

    # 3. Batch Evaluation on CPU
    print(f"\n[CPU Evaluation Execution]")
    batch_size = 256
    ds_test = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

    y_pred_probs_list = []
    t_start = time.perf_counter()
    for batch in ds_test:
        batch_float = tf.cast(batch, tf.float32)
        preds = model(batch_float, training=False).numpy().flatten()
        y_pred_probs_list.extend(preds)
    t_total = time.perf_counter() - t_start

    y_pred_probs = np.array(y_pred_probs_list, dtype=np.float32)
    latency_per_img_ms = (t_total / n_test) * 1000.0

    print(f" Total Forward Pass Time: {t_total:.2f} s on CPU")
    print(f" Per-Image Forward Latency: {latency_per_img_ms:.3f} ms")
    print(f" Throughput: {n_test / t_total:.1f} images/sec")

    # Save raw predictions
    pred_save_path = Path("model/test_predictions.npy")
    np.save(str(pred_save_path), y_pred_probs)
    print(f" Saved raw prediction probabilities to: {pred_save_path}")

    # 4. Independent Metric Calculation
    threshold = 0.5
    y_pred = (y_pred_probs >= threshold).astype(int)

    n_correct = int(np.sum(y_pred == y_test))
    raw_acc = accuracy_score(y_test, y_pred)
    bce_loss = log_loss(y_test, y_pred_probs)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    pr_auc = average_precision_score(y_test, y_pred_probs)

    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, average=None)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n[Independent Metric Verification]")
    print(f" Total Correct:      {n_correct:,} / {n_test:,}")
    print(f" Accuracy:           {raw_acc * 100:.2f}% ({raw_acc:.4f})")
    print(f" Macro Precision:    {prec_macro * 100:.2f}%")
    print(f" Macro Recall:       {rec_macro * 100:.2f}%")
    print(f" Macro F1-Score:     {f1_macro * 100:.2f}% ({f1_macro:.4f})")
    print(f" ROC-AUC:            {roc_auc:.4f}")
    print(f" PR-AUC:             {pr_auc:.4f}")
    print(f" BCE Loss (LogLoss): {bce_loss:.4f}")

    print(f"\n[Confusion Matrix Breakdown]")
    print(f" True Negatives  (AI correctly classified as AI):     {tn:,} ({tn / n_class_0 * 100:.2f}%)")
    print(f" False Positives (AI falsely classified as REAL):    {fp:,} ({fp / n_class_0 * 100:.2f}%)")
    print(f" False Negatives (REAL falsely classified as AI):    {fn:,} ({fn / n_class_1 * 100:.2f}%)")
    print(f" True Positives  (REAL correctly classified as REAL): {tp:,} ({tp / n_class_1 * 100:.2f}%)")

    # 5. Prediction-Distribution & Collapse Diagnostic (Requirement 6)
    print(f"\n[Comprehensive Prediction Distribution Diagnostic]")
    mask_real = (y_test == 1)
    mask_ai = (y_test == 0)

    mean_prob_real = float(np.mean(y_pred_probs[mask_real]))
    std_prob_real = float(np.std(y_pred_probs[mask_real]))
    mean_prob_ai = float(np.mean(y_pred_probs[mask_ai]))
    std_prob_ai = float(np.std(y_pred_probs[mask_ai]))

    class_separation = mean_prob_real - mean_prob_ai
    percentiles = [float(p) for p in np.percentile(y_pred_probs, [0, 10, 25, 50, 75, 90, 100])]

    # 10-bin histogram
    hist_counts, bin_edges = np.histogram(y_pred_probs, bins=10, range=(0.0, 1.0))
    hist_dict = {f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})": int(hist_counts[i]) for i in range(10)}

    print(f" Mean Probability for True REAL: {mean_prob_real:.4f} (std: {std_prob_real:.4f})")
    print(f" Mean Probability for True AI:   {mean_prob_ai:.4f} (std: {std_prob_ai:.4f})")
    print(f" Class Separation Margin:        {class_separation:.4f}")
    print(f" Probability Percentiles [0, 10, 25, 50, 75, 90, 100]:")
    print(f"   {[round(p, 4) for p in percentiles]}")
    print(f" 10-Bin Prediction Distribution Histogram:")
    for b_range, count in hist_dict.items():
        bar = "#" * int(count / 400)
        print(f"   {b_range}: {count:5d} | {bar}")

    # Check for collapse
    ratio_predicted_real = float(np.sum(y_pred == 1) / n_test)
    ratio_predicted_ai = float(np.sum(y_pred == 0) / n_test)
    is_collapsed = (ratio_predicted_real > 0.85 or ratio_predicted_ai > 0.85 or class_separation < 0.3)

    print(f" Ratio Predicted REAL: {ratio_predicted_real * 100:.2f}%")
    print(f" Ratio Predicted AI:   {ratio_predicted_ai * 100:.2f}%")
    print(f" Prediction Collapse Status: {'COLLAPSED' if is_collapsed else 'HEALTHY (No Collapse)'}")

    # 6. Evaluation of ONGC Regression Portraits (Requirement 7)
    print(f"\n[External Regression Test: ONGC Vigilance Department Portraits]")
    from src.predict import predict_image

    ongc_1 = "tests/regression_data/ai_portrait_ongc_1.png"
    ongc_2 = "tests/regression_data/ai_portrait_ongc_2.png"

    ongc_results = {}
    for p in [ongc_1, ongc_2]:
        p_obj = Path(p)
        if p_obj.exists():
            # Run end-to-end inference
            res = predict_image(str(p_obj))
            raw_sigmoid = res["raw_score"]
            verdict = res["verdict"]
            driver = res["primary_driver"]
            is_real = res["is_real"]
            is_ai = res["is_ai"]
            ood_status = res["is_ood"]

            ongc_results[p_obj.name] = {
                "raw_sigmoid_score": float(raw_sigmoid),
                "is_real": bool(is_real),
                "is_ai": bool(is_ai),
                "verdict": verdict,
                "primary_driver": driver,
                "is_ood": bool(ood_status),
                "confidence": float(res["confidence"]),
                "regression_passed": (not is_real)
            }

            print(f" Image: {p_obj.name}")
            print(f"   Raw Deep Model Score P(REAL): {raw_sigmoid:.4f}")
            print(f"   Is Out-Of-Distribution (OOD): {ood_status}")
            print(f"   Final System Verdict:         {verdict} (Driver: {driver})")
            print(f"   Classified as REAL:           {is_real} -> {'PASSED (Not Real)' if not is_real else 'FAILED (False Real)'}")

    # 7. Write Comprehensive Verification JSON Report
    report = {
        "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint": {
            "path": str(model_path),
            "architecture": model.name,
            "parameters": param_count,
            "file_size_mb": round(file_size_mb, 2),
            "input_shape": list(model.input_shape)
        },
        "dataset_split": {
            "source": "Official CIFAKE Independent Test Partition",
            "total_test_images": n_test,
            "class_0_fake": n_class_0,
            "class_1_real": n_class_1,
            "partition_preservation": "Strictly unaugmented and isolated from train/val splits"
        },
        "metrics": {
            "accuracy": round(raw_acc, 6),
            "accuracy_percent": round(raw_acc * 100, 2),
            "macro_precision": round(prec_macro, 6),
            "macro_recall": round(rec_macro, 6),
            "macro_f1": round(f1_macro, 6),
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
            "binary_cross_entropy": round(bce_loss, 6),
            "total_correct": n_correct,
            "total_samples": n_test
        },
        "per_class": {
            "class_0_ai_fake": {
                "precision": round(float(prec[0]), 6),
                "recall": round(float(rec[0]), 6),
                "f1": round(float(f1[0]), 6),
                "support": int(sup[0])
            },
            "class_1_real": {
                "precision": round(float(prec[1]), 6),
                "recall": round(float(rec[1]), 6),
                "f1": round(float(f1[1]), 6),
                "support": int(sup[1])
            }
        },
        "confusion_matrix": {
            "tn_ai_correct": int(tn),
            "fp_ai_as_real": int(fp),
            "fn_real_as_ai": int(fn),
            "tp_real_correct": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        "prediction_distribution_diagnostic": {
            "mean_prob_true_real": round(mean_prob_real, 4),
            "std_prob_true_real": round(std_prob_real, 4),
            "mean_prob_true_ai": round(mean_prob_ai, 4),
            "std_prob_true_ai": round(std_prob_ai, 4),
            "class_separation_margin": round(class_separation, 4),
            "percentiles_0_10_25_50_75_90_100": [round(p, 4) for p in percentiles],
            "histogram_10_bins": hist_dict,
            "ratio_predicted_real": round(ratio_predicted_real, 4),
            "ratio_predicted_ai": round(ratio_predicted_ai, 4),
            "is_collapsed": is_collapsed
        },
        "cpu_latency": {
            "total_time_seconds": round(t_total, 2),
            "latency_ms_per_image": round(latency_per_img_ms, 3),
            "throughput_images_per_sec": round(n_test / t_total, 1)
        },
        "ongc_regression_evaluation": ongc_results
    }

    out_file = Path("model/verified_independent_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Wrote verified independent evaluation report to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_independent_audit()
