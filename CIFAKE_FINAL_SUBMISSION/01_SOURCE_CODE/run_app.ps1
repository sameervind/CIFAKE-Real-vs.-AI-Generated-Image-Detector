# CIFAKE Real vs. AI-Generated Image Detector - PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "  CIFAKE: Real vs. AI-Generated Image Detector" -ForegroundColor Cyan
Write-Host "  CIFAR-CNN Propulsion Engine + Multi-Scale Diagnostics" -ForegroundColor White
Write-Host "  Design System: Lamborghini Automotive // Aceternity UI" -ForegroundColor DarkGray
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Select an option:"
Write-Host "[1] Launch Streamlit Interactive Web App (Port 8501)"
Write-Host "[2] Launch FastAPI REST Backend (Port 8000)"
Write-Host "[3] Run Automated Test Suite (pytest)"
Write-Host "[4] Run Sample CLI Predictions (Real + AI)"
Write-Host "[5] Run Live Specimen Regression Verification"
Write-Host "[6] Exit"
Write-Host ""

$choice = Read-Host "Enter choice [1-6]"

switch ($choice) {
    "1" {
        Write-Host "Launching Streamlit App..." -ForegroundColor Green
        streamlit run streamlit_app.py
    }
    "2" {
        Write-Host "Launching FastAPI REST Server at http://localhost:8000 ..." -ForegroundColor Green
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "3" {
        Write-Host "Running pytest test suite..." -ForegroundColor Green
        pytest tests/ -v
    }
    "4" {
        Write-Host "Running CLI prediction on sample real airplane..." -ForegroundColor Green
        python -m src.predict samples/real/01_real_airplane.jpg
        Write-Host ""
        Write-Host "Running CLI prediction on sample AI image..." -ForegroundColor Green
        python -m src.predict samples/ai/02_ai_diffused_fantasy_mythical_bird.jpg
    }
    "5" {
        Write-Host "Running live specimen regression verification..." -ForegroundColor Green
        python scripts/verify_live_predictions.py
    }
    Default {
        Write-Host "Exiting." -ForegroundColor Gray
    }
}
