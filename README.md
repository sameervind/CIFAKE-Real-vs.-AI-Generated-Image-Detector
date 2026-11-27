# CIFAKE: Real Photography vs. AI-Generated Synthetic Media Detection

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![ResNet50](https://img.shields.io/badge/Model-ResNet50_Transfer_Learning-orange.svg)](https://keras.io/api/applications/resnet/)
[![Backend](https://img.shields.io/badge/Engine-TensorFlow_Keras_3-red.svg)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_Backend-009688.svg)](http://localhost:8000/docs)
[![Streamlit](https://img.shields.io/badge/Streamlit-Lamborghini_UI-FFC000.svg)](http://localhost:8501)
[![Tests](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](tests/)

An end-to-end computer vision and deep learning platform engineered to distinguish **authentic camera photographs** from **AI-generated synthetic media** (Diffusion Models like Stable Diffusion and Generative Adversarial Networks).

Powered by an authoritative **two-stage transfer learning ResNet-50** deep residual network, calibrated ImageNet zero-centered preprocessing, and non-blocking **2D Fourier spectral diagnostics**.

---

## 📑 Table of Contents
1. [Core Architectural Upgrades](#-core-architectural-upgrades)
2. [Scientific Limitations & Out-of-Domain Scope](#-scientific-limitations--out-of-domain-scope)
3. [Deep Learning Architecture & Two-Stage Training](#-deep-learning-architecture--two-stage-training)
4. [Benchmark & Verification Metrics](#-benchmark--verification-metrics)
5. [Repository Directory Structure](#-repository-directory-structure)
6. [Quickstart & Installation](#-quickstart--installation)
7. [Command-Line Interface (CLI)](#-command-line-interface-cli)
8. [Model Training & Dataset Pipeline](#-model-training--dataset-pipeline)
9. [FastAPI REST Backend](#-fastapi-rest-backend)
10. [Streamlit Web Application](#-streamlit-web-application)
11. [Automated Unit Testing](#-automated-unit-testing)
12. [Citations & Acknowledgments](#-citations--acknowledgments)

---

## 🚀 Core Architectural Upgrades

### 1. Authoritative ResNet-50 Inference (Zero Silent Fallback)
- **Eliminated Silent Substitution**: The platform never secretly substitutes heuristic engines or spectral analyzers when the deep learning checkpoint is missing.
- **Explicit Failure Mode**: Missing or corrupted weights immediately raise a descriptive `ModelUnavailableError` (mapped to HTTP 503 `Service Unavailable` in the REST API).
- **Active Backend Telemetry**: All predictions, `/health` endpoints, and UI views report the exact running engine (`tensorflow_keras (ResNet50)`).
- **Fast Eager Tensor Inference**: Uses direct `model(batch, training=False)` eager execution to bypass `tf.data` graph execution latency on single-image requests.

### 2. Verified Preprocessing Pipeline
- **Zero-Centered ImageNet BGR Subtraction**: Exactly subtracts ImageNet channel means:
  $$\mu = [103.939, 116.779, 123.68] \text{ (BGR order)}$$
- Standardized image dimensions to $32 \times 32 \times 3$ with OpenCV area downsampling.

### 3. Non-Blocking 2D Fourier & Sensor Noise Diagnostics
- **Supplementary Diagnostic Signals**: 2D Discrete Fourier Transform (2D FFT) and physical sensor noise residual analyses run strictly as explainable supplementary signals and never override ResNet-50 class probabilities.
- Eliminates spatial aliasing false-positives on real photographs containing sharp repetitive textures (e.g., foliage, university campus brickwork, architecture).

---

## ⚠️ Scientific Limitations & Out-of-Domain Scope

> [!IMPORTANT]
> The **CIFAKE** benchmark comprises **120,000 images** (60,000 real CIFAR-10 photographic cutouts and 60,000 synthetic images generated using **Stable Diffusion v1.4**) at a native resolution of **$32 \times 32$ pixels**.

- **Dataset Representation**: Real images derive from CIFAR-10 categories (*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*). Synthetic images represent Stable Diffusion v1.4 outputs matching these classes.
- **High-Resolution & Modern Generators**: Models trained exclusively on CIFAKE are specialized to $32 \times 32$ latent diffusion signatures. Modern state-of-the-art generators (Midjourney v6, DALL-E 3, Flux.1, SDXL) operate at resolutions $\ge 1024 \times 1024$ with advanced noise schedules and upsamplers.
- **Domain Calibration Notice**: When arbitrary high-resolution user photographs or modern synthetic images are downsampled to $32 \times 32$, out-of-domain distribution shifts can occur. The application explicitly flags non-native input resolutions ($> 32 \times 32$) with a domain calibration advisory.

---

## 🧠 Deep Learning Architecture & Two-Stage Training

### Network Architecture

```
Input Image (H × W × 3, RGB)
       │
       ▼
 [Preprocessing: RGB ──► BGR ──► Resize(32, 32) ──► Mean Subtraction [103.939, 116.779, 123.68]]
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Pre-trained ResNet-50 Feature Backbone                      │
 │   - conv1_pad, conv1_conv, conv1_bn, conv1_relu             │
 │   - conv2_block1 .. conv2_block3 (Residual Skips)           │
 │   - conv3_block1 .. conv3_block4 (Residual Skips)           │
 │   - conv4_block1 .. conv4_block6 (Residual Skips)           │
 │   - conv5_block1 .. conv5_block3 (Deep Semantic Manifold)   │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
               GlobalAveragePooling2D (2048-dim)
                               │
                               ▼
                      Dense (128, ReLU)
                               │
                               ▼
                      BatchNormalization
                               │
                               ▼
                        Dropout (0.40)
                               │
                               ▼
               Dense (1, Sigmoid) ──► P(REAL PHOTOGRAPH)
```

### Two-Stage Transfer Learning Methodology

1. **Stage A — Frozen Backbone Warm-Up**:
   - ResNet-50 convolutional base layers are frozen (`layer.trainable = False`).
   - Custom classification head is trained for initial epochs with `Adam(lr=1e-3)` and binary cross-entropy loss.
   - Prevents catastrophic disruption of ImageNet feature extractors.
2. **Stage B — Selective Fine-Tuning**:
   - Deepest residual stages (`conv5_block1_out` through `conv5_block3_out`) are unfrozen.
   - All `BatchNormalization` layers remain strictly frozen (`bn.trainable = False`) to prevent destructive recalculation of running mean and variance on small batch sizes.
   - Trained with reduced learning rate `Adam(lr=1e-5)` to delicately adapt high-level convolutional kernels to low-resolution generative diffusion fingerprints.

---

## 📊 Benchmark & Verification Metrics

The platform transparently distinguishes between **Published CIFAKE Benchmark Reference** and the **Active Model Checkpoint**:

| Metric | Published CIFAKE Benchmark (120k) | Active Checkpoint (`model/resnet50_cifake.keras`) |
| :--- | :---: | :---: |
| **Accuracy** | **92.28%** | **100.0%** (Sample Verification Set) |
| **ROC-AUC** | **0.9742** | **1.0000** |
| **Precision (Macro Avg)** | **92.42%** | **1.0000** |
| **Recall (Macro Avg)** | **92.30%** | **1.0000** |
| **Binary Cross-Entropy Loss** | **0.1983** | **0.0084** |
| **Prediction Collapse** | False | False (Balanced class distribution) |

---

## 📂 Repository Directory Structure

```
sv/
├── app/
│   └── streamlit_app.py        # Streamlit dashboard (app folder mirror)
├── api/
│   └── main.py                 # FastAPI REST API with sanitized Pydantic schemas
├── src/
│   ├── __init__.py             # Package marker
│   ├── dataset.py              # Stratified data loader & ImageNet BGR preprocessing
│   ├── model.py                # ResNet50 model builder & ResNet50InferenceService
│   ├── predict.py              # CLI & programmatic inference orchestrator
│   ├── train.py                # Two-stage transfer learning training pipeline
│   ├── evaluate.py             # Metrics calculator & prediction collapse detector
│   ├── frequency_analysis.py   # 2D FFT & sensor noise supplementary diagnostics
│   └── download_cifake.py      # Dataset acquisition helper
├── model/
│   ├── resnet50_cifake.keras   # Genuine trained ResNet50 Keras 3 model checkpoint
│   ├── model_metadata.json     # Architecture specification & training metadata
│   ├── evaluation_metrics.json # Active evaluation metrics & collapse status
│   └── training_history.json   # Two-stage epoch loss/accuracy logs
├── samples/
│   ├── real/                   # Validated CIFAR-10 photograph samples
│   └── ai/                     # Validated Stable Diffusion synthetic samples
├── assets/
│   ├── confusion_matrix.png    # Evaluation heatmap
│   ├── accuracy_loss_curves.png# Training/validation curves
│   └── roc_curve.png           # Receiver Operating Characteristic
├── tests/
│   ├── test_model.py           # Unit tests for preprocessing, ResNet50, and service
│   └── test_api.py             # Integration tests for FastAPI endpoints
├── Dockerfile                  # Containerized deployment spec
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & tool configuration
├── run_app.bat                 # Windows interactive launcher
├── streamlit_app.py            # Root Streamlit dashboard entrypoint
└── README.md                   # Technical documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 (64-bit)
- Windows, macOS, or Linux

### 2. Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd sv

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Windows 1-Click Interactive Menu
Run the batch launcher:
```cmd
run_app.bat
```
Provides an interactive 7-option menu:
1. Launch Streamlit Web Dashboard
2. Start FastAPI REST Backend Server
3. Run Automated Pytest Test Suite
4. Check Environment & Dataset Health
5. Run ResNet50 Transfer Learning (Full Training)
6. Run Single Image Prediction (CLI)
7. Exit

---

## 💻 Command-Line Interface (CLI)

Run inference on any image directly from the terminal using the authoritative ResNet-50 engine:

```bash
# Predict an authentic photograph
python -m src.predict samples/real/01_real_airplane.jpg

# Predict a synthetic AI-generated image
python -m src.predict samples/ai/01_ai_hyperrealistic_cyberpunk_neon_car.jpg

# Custom decision threshold (default: 0.50)
python -m src.predict path/to/image.jpg --threshold 0.60
```

### Example Terminal Output
```
[ResNet50Service] Loaded trained checkpoint from: model\resnet50_cifake.keras
======================================================================
  Image:       samples\real\01_real_airplane.jpg
  Prediction:  REAL (99.9% confidence)
  P(REAL):     0.9995
  P(AI):       0.0005
  Backend:     tensorflow_keras (ResNet50)
  Latency:     31.42 ms
  Explanation: ResNet-50 deep residual manifold matches photographic distribution (P(REAL)=99.9%). Physical sensor noise std=1.84 with natural gradient distribution.
======================================================================
```

---

## 🏋️ Model Training & Dataset Pipeline

### Verify Environment & Dataset
```bash
python -m src.train --check
```

### Execute Two-Stage Transfer Learning
```bash
python -m src.train --data_dir data/cifake \
                    --epochs_stage_a 5 \
                    --epochs_stage_b 5 \
                    --batch_size 32 \
                    --fine_tune_from conv5_block1_out \
                    --output_model model/resnet50_cifake.keras
```

The training process automatically:
1. Validates directory structure (`REAL/` and `FAKE/` or `train/` and `test/`).
2. Applies stratified dataset splitting (70% Train, 20% Validation, 10% Test).
3. Trains Stage A (head warm-up with frozen ResNet-50 backbone).
4. Trains Stage B (fine-tuning from `conv5_block1_out` with frozen BatchNorm).
5. Runs post-training evaluation on held-out test data.
6. Assesses predictions for class collapse.
7. Saves the model checkpoint, metadata, history, and metrics to `model/`.

---

## ⚡ FastAPI REST Backend

Launch the production REST backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation is live at `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API status, version, and documentation link | 200 |
| `GET` | `/health` | System health, model availability, and active backend | 200, 503 |
| `GET` | `/metrics` | Active model evaluation metrics and benchmark figures | 200 |
| `GET` | `/samples` | Available CIFAR-10 and AI sample image library | 200 |
| `POST` | `/predict` | Single image classification via multipart file upload | 200, 400, 503 |
| `POST` | `/batch-predict` | Batch classification for up to 50 images | 200, 400, 503 |

### Example `POST /predict` Request & Response
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/real/01_real_airplane.jpg"
```

```json
{
  "filename": "01_real_airplane.jpg",
  "label": "REAL",
  "is_ai": false,
  "is_real": true,
  "confidence": 0.9995,
  "confidence_percent": "99.9%",
  "probabilities": {
    "REAL": 0.9995,
    "AI": 0.0005
  },
  "backend": "tensorflow_keras (ResNet50)",
  "artifacts": {
    "spectral_peak_ratio": 1.25,
    "high_to_low_ratio": 0.000042,
    "laplacian_variance": 412.35,
    "noise_residual_std": 1.84
  },
  "explanation": "ResNet-50 deep residual manifold matches photographic distribution (P(REAL)=99.9%). Physical sensor noise std=1.84 with natural gradient distribution.",
  "processing_time_ms": 31.42
}
```

---

## 🏎️ Streamlit Web Application

Launch the interactive dashboard:
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501` in your web browser.

### Lamborghini Automotive Design System
The UI implements an aggressive, precision-engineered automotive theme:
- **Color Palette**: Pure Carbon Black (`#000000`), Dark Titanium (`#0B0E14`), and Giallo Auge Gold (`#FFC000`).
- **Geometry**: Razor-sharp $0\text{px}$ corner radii with high-contrast structural borders (`#222222`).
- **Dynamic Telemetry**: Live status badge reflecting the active model backend and health dot in the navigation bar.
- **Domain Calibration Alert**: Automatically displays a scientific advisory banner when user uploads exceed $32 \times 32$ resolution.
- **Dual Benchmark Transparency**: Displays both active checkpoint metrics and published 120,000 CIFAKE benchmark statistics side-by-side.

---

## 🧪 Automated Unit Testing

Run the automated test suite with pytest:
```bash
pytest tests/ -v
```

### Coverage
- `tests/test_model.py`:
  - ImageNet BGR zero-centering normalization (`test_preprocess_image`)
  - Stratified 70/20/10 dataset splitting (`test_dataset_stratified_split`)
  - ResNet-50 production architecture construction (`test_model_architecture`)
  - Two-stage fine-tuning layer freeze verification (`test_fine_tuning_configuration`)
  - Authoritative inference and probability calibration (`test_inference_service_authoritative`)
  - Zero-fallback `ModelUnavailableError` enforcement (`test_model_unavailable_error`)
  - Prediction collapse detection diagnostics (`test_prediction_collapse_detection`)
  - 2D Fourier spectrum computation (`test_compute_fft_spectrum`)
  - Sensor shot noise and spectral artifact extraction (`test_extract_artifact_features`)
- `tests/test_api.py`:
  - Root `/` and metadata endpoints
  - Health `/health` endpoint and backend verification
  - Performance metrics `/metrics` endpoint
  - Sample library `/samples` endpoint
  - Image prediction `/predict` endpoint
  - Invalid file rejection handling
  - Multi-image batch inference `/batch-predict` endpoint

---

## 📜 Citations & Acknowledgments
- **CIFAKE Benchmark**: Bird, J.J. and Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. arXiv:2303.14126.
- **CIFAR-10**: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
- **ResNet Architecture**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- **Stable Diffusion**: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
