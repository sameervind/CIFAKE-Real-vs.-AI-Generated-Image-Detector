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

try:
    import app.streamlit_app as dashboard
    dashboard.main()
except Exception as e:
    import streamlit as st
    import traceback
    st.error(f"Application error during execution: {e}")
    st.code(traceback.format_exc())
