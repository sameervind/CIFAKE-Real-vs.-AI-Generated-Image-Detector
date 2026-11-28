# CIFAKE Model Evaluation & Experimental Verification Results

**Verification Date:** November 28, 2026  
**Hardware Platform:** CPU (oneDNN Optimized, AVX2/FMA instructions)  
**Model Name:** `CIFAR_CNN`  
**Checkpoint File:** `model/best_cifake_model.keras`  
**Framework:** TensorFlow 2.22.0-rc1 / Keras 3.x  

---

## 1. Executive Metrics (Independent 20,000-Image CIFAKE Test Partition)

The active production model was independently verified against the official, unaugmented CIFAKE test set (10,000 genuine CIFAR-10 photographs and 10,000 Stable Diffusion v1.4 synthetic images).

| Metric | Measured Value | Percentage / Standard Error |
| :--- | :--- | :--- |
| **Test Accuracy** | **0.9492** | **94.92%** (18,984 / 20,000 correct) |
| **Macro Precision** | **0.9494** | 94.94% |
| **Macro Recall** | **0.9492** | 94.92% |
| **Macro F1-Score** | **0.9492** | 0.9492 |
| **ROC-AUC Score** | **0.9903** | Area Under Receiver Operating Characteristic |
| **PR-AUC Score** | **0.9907** | Area Under Precision-Recall Curve |
| **Binary Cross-Entropy Loss** | **0.1298** | Test Partition BCE |
| **Average CPU Latency** | **1.58 ms** | Standard Intel/AMD CPU (Batch Size = 1) |
| **Model Parameter Count** | **305,953** | 1.17 MB Model File Size |

---

## 2. Confusion Matrix

Decision Boundary: $\text{Threshold} = 0.50$, where $\hat{y} = P(\text{REAL})$.

```
                    Predicted AI (0)    Predicted REAL (1)
Actual AI (0)             9,382                 618
Actual REAL (1)             398               9,602
```

- **True AI (TN):** 9,382 images correctly classified as AI-generated.
- **False REAL (FP):** 618 AI images misclassified as REAL (False Positive Rate = 6.18%).
- **False AI (FN):** 398 REAL photographs misclassified as AI (False Negative Rate = 3.98%).
- **True REAL (TP):** 9,602 genuine photographs correctly classified as REAL.
- **Total Correct:** $9,382 + 9,602 = 18,984$ out of 20,000 ($94.92\%$).

---

## 3. Per-Class Granular Performance

| Class Label | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Class 0: AI / Synthetic** | 0.9593 (95.93%) | 0.9382 (93.82%) | 0.9486 (94.86%) | 10,000 |
| **Class 1: Real Photography** | 0.9395 (93.95%) | 0.9602 (96.02%) | 0.9498 (94.98%) | 10,000 |
| **Macro Average** | **0.9494** | **0.9492** | **0.9492** | **20,000** |
| **Weighted Average** | **0.9494** | **0.9492** | **0.9492** | **20,000** |

---

## 4. Architecture Benchmark Comparison

Identical evaluation conditions on the official 20,000 CIFAKE test partition:

| Model Architecture | Input Resolution | Parameter Count | Test Accuracy | Macro F1 | ROC-AUC | CPU Latency (ms) | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAR_CNN (Production)** | **32 × 32 × 3** | **305,953** | **94.92%** | **0.9492** | **0.9903** | **1.58 ms** | **Production Winner** |
| **CIFAR_ResNet20** | 32 × 32 × 3 | 283,137 | 92.23% | 0.9221 | 0.9740 | 2.45 ms | Benchmark Baseline |
| **ResNet-50 (ImageNet)** | 224 × 224 × 3 | 24,146,305 | 92.28% | 0.9224 | 0.9742 | 42.10 ms | Legacy Architecture |

### Key Architectural Findings:
1. **The Native Resolution Advantage:** Native $32 \times 32$ convolutional kernels preserve subtle high-frequency diffusion artifacts without destructive bilinear smoothing or $49\times$ spatial memory inflation.
2. **Speed & Efficiency:** `CIFAR_CNN` delivers a **$26.6\times$ latency reduction** compared to ResNet-50 (1.58 ms vs. 42.10 ms) on standard CPU hardware.
3. **Parameter Optimization:** `CIFAR_CNN` requires only **305k parameters**, representing a **$79\times$ reduction** over the 24.1M-parameter ResNet-50 while achieving **+2.64% higher accuracy**.

---

## 5. Live Specimen Regression Verification

Evaluation across verified in-domain and external real-world test collections:

| Specimen Name | Ground Truth | Model Verdict | Raw P(REAL) | Raw P(AI) | Calibrated Confidence | OOD Score | Primary Driver | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAKE REAL Sample (Airplane)** | REAL | LIKELY REAL | 0.7233 | 0.2767 | 72.3% | 0.4803 (In-Dist) | CIFAR_CNN | **PASSED** |
| **CIFAKE FAKE Sample (Mythical Bird)** | AI | LIKELY AI-GENERATED | 0.1002 | 0.8998 | 85.2% | 0.5300 (OOD) | CIFAR_CNN | **PASSED** |
| **Known Genuine Portrait (Real)** | REAL | LIKELY REAL | 0.9996 | 0.0004 | 91.2% | 0.8717 (OOD) | CIFAR_CNN | **PASSED** |
| **Known AI-Generated Portrait 1 (ONGC)** | AI | LIKELY AI-GENERATED | 0.0582 | 0.9418 | 86.7% | 0.7925 (OOD) | CIFAR_CNN | **PASSED** |
| **Known AI-Generated Portrait 2 (ONGC)** | AI | LIKELY AI-GENERATED | 0.0818 | 0.9182 | 84.6% | 0.7857 (OOD) | CIFAR_CNN | **PASSED** |

- **False Positive Rate on External Portraits:** $0.0\%$ (0/1 missed).
- **False Negative Rate on External Portraits:** $0.0\%$ (0/2 missed).
- **Consolidated Specimen Accuracy:** **100% (5 / 5)**.
