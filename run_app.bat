@echo off
REM Launcher script for CIFAKE Real vs. AI Image Detector // Lamborghini Forensics Suite

echo ========================================================
echo   CIFAKE: Real vs. AI-Generated Image Detector
echo   CIFAR-CNN Propulsion Engine + Multi-Scale Diagnostics
echo   Design System: Lamborghini Automotive // Aceternity UI
echo ========================================================
echo.
echo Select an option:
echo [1] Launch Streamlit Interactive Web App (Port 8501)
echo [2] Launch FastAPI REST Backend (Port 8000)
echo [3] Run Automated Test Suite (pytest)
echo [4] Run Sample CLI Predictions (Real + AI)
echo [5] Run Two-Stage Model Training Pipeline
echo [6] Verify Environment & Compute Device
echo [7] Exit
echo.

set /p choice="Enter choice [1-7]: "

if "%choice%"=="1" (
    echo Launching Streamlit App...
    streamlit run app/streamlit_app.py
) else if "%choice%"=="2" (
    echo Launching FastAPI Server at http://localhost:8000 ...
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
) else if "%choice%"=="3" (
    echo Running unit tests...
    pytest tests/ -v
    pause
) else if "%choice%"=="4" (
    echo Running CLI prediction on sample real airplane...
    python -m src.predict samples/real/01_real_airplane.jpg
    echo.
    echo Running CLI prediction on sample AI car...
    python -m src.predict samples/ai/01_ai_hyperrealistic_cyberpunk_neon_car.jpg
    pause
) else if "%choice%"=="5" (
    echo Executing Two-Stage ResNet-50 Training Pipeline...
    python -m src.train --data_dir samples
    pause
) else if "%choice%"=="6" (
    echo Running Environment Validation...
    python -m src.train --check
    pause
) else (
    echo Exiting.
)
