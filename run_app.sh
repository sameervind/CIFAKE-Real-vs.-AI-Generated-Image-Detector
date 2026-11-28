#!/usr/bin/env bash
# CIFAKE Real vs. AI-Generated Image Detector - Bash Launcher
set -e

echo "========================================================"
echo "  CIFAKE: Real vs. AI-Generated Image Detector"
echo "  CIFAR-CNN Propulsion Engine + Multi-Scale Diagnostics"
echo "  Design System: Lamborghini Automotive // Aceternity UI"
echo "========================================================"
echo ""
echo "Select an option:"
echo "[1] Launch Streamlit Interactive Web App (Port 8501)"
echo "[2] Launch FastAPI REST Backend (Port 8000)"
echo "[3] Run Automated Test Suite (pytest)"
echo "[4] Run Sample CLI Predictions (Real + AI)"
echo "[5] Run Live Specimen Regression Verification"
echo "[6] Exit"
echo ""

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Launching Streamlit App..."
        streamlit run streamlit_app.py
        ;;
    2)
        echo "Launching FastAPI REST Server at http://localhost:8000 ..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    3)
        echo "Running pytest test suite..."
        pytest tests/ -v
        ;;
    4)
        echo "Running CLI prediction on sample real airplane..."
        python -m src.predict samples/real/01_real_airplane.jpg
        echo ""
        echo "Running CLI prediction on sample AI image..."
        python -m src.predict samples/ai/02_ai_diffused_fantasy_mythical_bird.jpg
        ;;
    5)
        echo "Running live specimen regression verification..."
        python scripts/verify_live_predictions.py
        ;;
    *)
        echo "Exiting."
        ;;
esac
