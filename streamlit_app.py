"""
Root entry point for Streamlit Community Cloud, Hugging Face Spaces, and local execution.
Automatically launches the CIFAKE Real vs. AI Image Detector Dashboard.
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
