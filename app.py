"""
Alternative root entry point for hosting platforms looking for app.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import app.streamlit_app as dashboard
    dashboard.main()
except Exception as e:
    import streamlit as st
    import traceback
    st.error(f"Application error: {e}")
    st.code(traceback.format_exc())
