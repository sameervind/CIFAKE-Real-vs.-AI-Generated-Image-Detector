# CIFAKE FINAL SUBMISSION PACKAGE

**Project Name:** CIFAKE — AI vs. Real Image Detection Platform  
**Academic Submission Date:** November 29, 2026  
**Primary Architecture:** Native CIFAR-CNN (305,953 Parameters) + 2D FFT Frequency Analysis + OOD Arbiter  
**Verified Performance:** 94.92% Test Accuracy, 0.9492 Macro F1, 0.9903 ROC-AUC, 1.58 ms CPU Latency  

---

## Submission Directory Structure

```
CIFAKE_FINAL_SUBMISSION/
├── 01_SOURCE_CODE/          Complete standalone, fully-functional codebase
│   ├── api/                 FastAPI REST backend with OpenAPI/Swagger documentation
│   ├── app/                 Streamlit web application (synced entry point)
│   ├── assets/              Visual plots, architectural diagrams, metric JSONs
│   ├── data/                Official CIFAKE independent 20k test cache (unaugmented)
│   ├── model/               Trained checkpoint (best_cifake_model.keras) & evaluation JSONs
│   ├── samples/             Real & AI test images across 10 semantic classes
│   ├── scripts/             Evaluation, verification, and launcher scripts
│   ├── src/                 Core inference engine, OOD detector, decision arbiter, forensics
│   ├── tests/               Pytest test suite (23/23 tests passing)
│   ├── requirements.txt     Pinned dependencies
│   ├── pyproject.toml       Package metadata and CLI entry points
│   ├── README.md            Comprehensive architecture and quickstart guide
│   ├── run_app.bat          One-click Windows Batch launcher
│   ├── run_app.ps1          One-click PowerShell launcher
│   ├── run_app.sh           One-click Unix/Linux bash launcher
│   └── streamlit_app.py     Streamlit Lamborghini Cockpit main entry point
│
├── 02_PROJECT_REPORT/       Complete Technical Academic Project Report
│   ├── CIFAKE_PROJECT_REPORT.md    Full markdown report (17 formal sections)
│   ├── CIFAKE_PROJECT_REPORT.docx  Formal Microsoft Word document with tables & figures
│   └── CIFAKE_PROJECT_REPORT.pdf   Formatted academic publication PDF
│
├── 03_PRESENTATION/         Professional Academic Defense Presentation
│   ├── CIFAKE_FINAL_PRESENTATION.pptx  12 widescreen slides in Lamborghini dark/gold theme
│   └── CIFAKE_FINAL_PRESENTATION.pdf   Printable/presentable PDF slide deck
│
├── 04_RESULTS/              Empirical Verification & Benchmark Results
│   ├── confusion_matrix.png            Verified confusion matrix (18,984 / 20,000 correct)
│   ├── roc_curve.png                   ROC curve (AUC = 0.9903)
│   ├── precision_recall_curve.png      PR curve (Macro F1 = 0.9492, PR-AUC = 0.9907)
│   ├── accuracy_loss_curves.png        Training & validation progression curves
│   ├── architecture_comparison.png     CIFAR-CNN vs. ResNet-50 benchmark chart
│   ├── evaluation_metrics.json         Granular test metrics and per-class scores
│   ├── experiment_results.csv          Architecture comparison table
│   ├── verified_prediction_samples.json Live specimen test telemetry
│   └── RESULTS_SUMMARY.md              Executive results summary
│
└── 05_DEMO/                 5-Minute Demonstration Walkthrough & Staged Assets
    ├── DEMO_SCRIPT.md                  Verbatim 5-minute spoken script and action guide
    ├── DEMO_WALKTHROUGH.pdf            Printable 1-page presentation reference card
    └── demo_images/                    Staged test images (real photos, AI portraits)
```

---

## Quickstart & Launch Instructions

### 1. Interactive Streamlit Cockpit (Port 8501)
```bash
cd 01_SOURCE_CODE
run_app.bat         # On Windows (Select Option 1)
# or
streamlit run streamlit_app.py
```

### 2. FastAPI REST Server (Port 8000)
```bash
cd 01_SOURCE_CODE
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Interactive documentation available at: http://localhost:8000/docs
```

### 3. Automated Test Suite (pytest)
```bash
cd 01_SOURCE_CODE
pytest tests/ -v
```
