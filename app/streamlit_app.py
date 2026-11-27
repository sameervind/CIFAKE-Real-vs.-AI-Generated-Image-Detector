"""
Streamlit Web Dashboard for Real vs. AI-Generated Image Classification.
Designed with Clean, Minimalist Production UI: Crisp Typography, Subtle Glassmorphism,
Refined Telemetry, and High-Precision ResNet50 / Spectral Forensics.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root workspace is on path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import preprocess_image, TARGET_SIZE
from src.predict import predict_image
from src.frequency_analysis import compute_fft_spectrum, extract_artifact_features

# Streamlit Page Config
st.set_page_config(
    page_title="AI vs. Real Image Classifier | CIFAKE ResNet-50",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional Dark UI Styling (No AI slop, no gaudy gradients, clean hierarchy)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Minimalist Dark Graphite Canvas */
    .stApp {
        background-color: #0b0f17 !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
            radial-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px) !important;
        background-size: 100% 100%, 32px 32px !important;
        color: #f1f5f9 !important;
    }

    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background: #0d121d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Header & Badge */
    .app-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .app-badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
    }

    .main-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* Clean Container Cards */
    .panel-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Clean Verdict Card - Real */
    .result-card-real {
        background: rgba(6, 78, 59, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 10px;
        padding: 18px 20px;
        margin: 12px 0 16px 0;
    }

    /* Clean Verdict Card - AI */
    .result-card-ai {
        background: rgba(136, 19, 55, 0.2);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 10px;
        padding: 18px 20px;
        margin: 12px 0 16px 0;
    }

    .verdict-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }

    .verdict-pill-real {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .verdict-pill-ai {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #fb7185;
        background: rgba(244, 63, 94, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .verdict-main-text {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 4px 0 6px 0;
    }

    .verdict-text-real { color: #f0fdf4; }
    .verdict-text-ai { color: #fff1f2; }

    .verdict-sub-text {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    .verdict-sub-text b {
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metric Tiles */
    .metric-tile {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px 10px;
        text-align: center;
    }

    .metric-tile-val {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-tile-lbl {
        font-size: 0.7rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 3px;
    }

    /* Diagnostic Note */
    .diagnostic-box {
        background: rgba(99, 102, 241, 0.05);
        border-left: 2px solid #6366f1;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin: 12px 0;
        line-height: 1.45;
    }

    /* Streamlit Widget Polishing */
    .stSlider > div > div > div > div {
        background-color: #6366f1 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #6366f1 !important;
    }
    .stDownloadButton > button {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    .stDownloadButton > button:hover {
        background-color: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
ASSETS_DIR = ROOT_DIR / "assets"
SAMPLES_REAL = ROOT_DIR / "samples" / "real"
SAMPLES_AI = ROOT_DIR / "samples" / "ai"


def load_metrics_data():
    metrics_file = ASSETS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {
        "accuracy": 0.9228,
        "precision_macro": 0.9242,
        "recall_macro": 0.9230,
        "f1_macro": 0.9224,
        "roc_auc": 0.9742
    }


def render_sidebar():
    st.sidebar.markdown("""
    <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25); padding: 8px 12px; border-radius: 6px; text-align: center; font-weight: 600; font-size: 0.8rem; color: #c7d2fe; margin-bottom: 14px;">
        ResNet-50 · 92.28% Accuracy
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Module Selection:",
        ["Detector", "Frequency Studio", "Model & Benchmarks", "Batch Evaluation", "About CIFAKE"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">SETTINGS</p>', unsafe_allow_html=True)
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold for REAL classification (default: 0.50). Images with P(Real) >= threshold are classified as Real."
    )
    show_fft = st.sidebar.checkbox("Show 2D Fourier Spectrum", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.5;">
        <span style="color: #94a3b8; font-weight: 600;">System Specs:</span><br>
        Architecture: ResNet-50 Transfer Learning<br>
        Dataset: CIFAKE (120,000 images)<br>
        Analysis: 2D FFT & Multi-Scale Forensics<br>
        Backend: NumPy / SciPy / OpenCV
    </div>
    """, unsafe_allow_html=True)
    return page, threshold, show_fft


def page_detector(threshold: float, show_fft: bool):
    st.markdown("""
    <div class="app-badge">
        <span class="app-badge-dot"></span> CIFAKE Detector · ResNet-50
    </div>
    <div class="main-title">AI vs. Real Image Classifier</div>
    <div class="main-subtitle">Upload any image to verify whether it is an authentic camera photograph or generated by AI.</div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 1.2], gap="large")

    selected_image = None
    image_source_name = "Uploaded Image"

    with col_input:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">1. Input Image</div>', unsafe_allow_html=True)
        input_mode = st.radio("Input Source:", ["Upload Image", "Sample Gallery"], horizontal=True, label_visibility="collapsed")

        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_name = uploaded_file.name
        else:
            sample_category = st.selectbox("Category:", ["Real Photos (CIFAR-10)", "AI-Generated (Stable Diffusion)"])
            sample_folder = SAMPLES_REAL if "Real" in sample_category else SAMPLES_AI

            if sample_folder.exists():
                sample_files = sorted(list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png")))
                if sample_files:
                    sample_options = [f.name for f in sample_files]
                    selected_sample_name = st.selectbox("Select Image:", sample_options)
                    selected_file_path = sample_folder / selected_sample_name
                    selected_image = Image.open(selected_file_path).convert("RGB")
                    image_source_name = selected_sample_name
            else:
                st.warning("Samples directory not found.")

        if selected_image is not None:
            st.image(selected_image, caption=f"{image_source_name} ({selected_image.size[0]}×{selected_image.size[1]} px)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">2. Classification Results</div>', unsafe_allow_html=True)

        if selected_image is None:
            st.info("Upload an image or select a sample from the gallery on the left to begin classification.")
        else:
            with st.spinner("Analyzing image features and frequency spectrum..."):
                img_array = np.array(selected_image)
                result = predict_image(img_array, threshold=threshold)

            is_real = result["is_real"]
            conf = result["confidence"]
            p_real = result["probabilities"]["REAL"]
            p_ai = result["probabilities"]["AI"]
            latency = result["processing_time_ms"]

            # Clean, refined verdict display
            if is_real:
                st.markdown(f"""
                <div class="result-card-real">
                    <div class="verdict-header-row">
                        <span class="verdict-pill-real">Authentic Media</span>
                        <span style="font-size: 0.8rem; color: #34d399; font-family: 'JetBrains Mono', monospace;">P(Real) = {p_real * 100:.1f}%</span>
                    </div>
                    <div class="verdict-main-text verdict-text-real">Real Photograph</div>
                    <div class="verdict-sub-text">Confidence: <b>{conf * 100:.1f}%</b> · Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ai">
                    <div class="verdict-header-row">
                        <span class="verdict-pill-ai">Synthetic Media</span>
                        <span style="font-size: 0.8rem; color: #fb7185; font-family: 'JetBrains Mono', monospace;">P(AI) = {p_ai * 100:.1f}%</span>
                    </div>
                    <div class="verdict-main-text verdict-text-ai">AI-Generated Image</div>
                    <div class="verdict-sub-text">Confidence: <b>{conf * 100:.1f}%</b> · Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Clean Probability Distribution
            st.markdown('<p style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 6px 0;">PROBABILITY DISTRIBUTION</p>', unsafe_allow_html=True)
            c_bar1, c_bar2 = st.columns(2)
            with c_bar1:
                st.metric("P(Real Image)", f"{p_real * 100:.2f}%")
                st.progress(p_real)
            with c_bar2:
                st.metric("P(AI-Generated)", f"{p_ai * 100:.2f}%")
                st.progress(p_ai)

            # Diagnostic Explanation
            st.markdown(f"""
            <div class="diagnostic-box">
                <span style="color: #a5b4fc; font-weight: 600;">Analysis Note:</span> {result["explanation"]}
            </div>
            """, unsafe_allow_html=True)

            # Clean Metric Tiles
            st.markdown('<p style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 14px 0 6px 0;">ARTIFACT METRICS</p>', unsafe_allow_html=True)
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-tile-val">{result['artifacts'].get('spectral_peak_ratio', 1.0):.1f}x</div>
                    <div class="metric-tile-lbl">Peak Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with t2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-tile-val">{result['artifacts'].get('laplacian_kurtosis', 3.0):.1f}</div>
                    <div class="metric-tile-lbl">Kurtosis</div>
                </div>
                """, unsafe_allow_html=True)
            with t3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-tile-val">{result['artifacts'].get('noise_residual_std', 5.0):.2f}</div>
                    <div class="metric-tile-lbl">Sensor Noise</div>
                </div>
                """, unsafe_allow_html=True)
            with t4:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-tile-val">{result['artifacts'].get('high_to_low_ratio', 0.0):.3f}</div>
                    <div class="metric-tile-lbl">High/Low Ratio</div>
                </div>
                """, unsafe_allow_html=True)

            # Optional Fourier Magnitude Plot
            if show_fft:
                st.markdown("---")
                st.markdown('<p style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">2D FOURIER SPECTRUM</p>', unsafe_allow_html=True)
                spec = result["spectrum_2d"]

                f1, f2 = st.columns(2)
                with f1:
                    st.image(selected_image.resize((256, 256)), caption="Resized Image", use_container_width=True)
                with f2:
                    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
                    fig.patch.set_facecolor('#0d121d')
                    ax.set_facecolor('#0d121d')
                    im = ax.imshow(spec, cmap="inferno")
                    ax.set_title("2D Magnitude Spectrum", fontsize=8, color="#cbd5e1")
                    ax.axis("off")
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.yaxis.set_tick_params(color='#64748b')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#64748b', fontsize=7)
                    st.pyplot(fig)
                    plt.close()

        st.markdown('</div>', unsafe_allow_html=True)


def page_frequency_studio():
    st.markdown("""
    <div class="app-badge">
        <span class="app-badge-dot"></span> Frequency Domain Analysis
    </div>
    <div class="main-title">2D Fourier Spectrum Studio</div>
    <div class="main-subtitle">Inspect frequency domain artifacts, deconvolution checkerboard spikes, and radial power decay.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card" style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
        <span style="color: #f8fafc; font-weight: 600;">How Frequency Forensics Works:</span><br>
        Generative models (GANs and Latent Diffusion models like Stable Diffusion) upscale feature maps using transposed convolutions 
        or interpolation. This creates subtle periodic grid artifacts in the spatial domain that manifest as distinct <b>off-axis peaks in 2D Fourier space</b>. 
        Natural camera images, in contrast, adhere to optical diffraction laws where spectral power decays smoothly according to a <b>1/f² power law</b>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Authentic CIFAR-10 Sample</div>', unsafe_allow_html=True)
        real_sample = SAMPLES_REAL / "01_real_airplane.jpg"
        if real_sample.exists():
            img_real = Image.open(real_sample).convert("RGB")
            st.image(img_real.resize((256, 256)), caption="Real Airplane (CIFAR-10)", use_container_width=True)
            res_real = predict_image(np.array(img_real))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#0d121d')
            ax.set_facecolor('#0d121d')
            ax.imshow(res_real["spectrum_2d"], cmap="viridis")
            ax.set_title("Natural 1/f Smooth Spectrum", fontsize=9, color="#cbd5e1")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Synthetic AI Sample (Stable Diffusion)</div>', unsafe_allow_html=True)
        ai_sample = SAMPLES_AI / "01_ai_hyperrealistic_cyberpunk_neon_car.jpg"
        if ai_sample.exists():
            img_ai = Image.open(ai_sample).convert("RGB")
            st.image(img_ai.resize((256, 256)), caption="AI Generated Car (Synthetic)", use_container_width=True)
            res_ai = predict_image(np.array(img_ai))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#0d121d')
            ax.set_facecolor('#0d121d')
            ax.imshow(res_ai["spectrum_2d"], cmap="inferno")
            ax.set_title("Periodic Harmonic Spikes", fontsize=9, color="#cbd5e1")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def page_architecture():
    st.markdown("""
    <div class="app-badge">
        <span class="app-badge-dot"></span> Performance & Evaluation
    </div>
    <div class="main-title">ResNet-50 Model & Benchmarks</div>
    <div class="main-subtitle">Validation metrics on 12,000 CIFAKE test set images matching the project notebook report.</div>
    """, unsafe_allow_html=True)

    metrics = load_metrics_data()

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="panel-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{metrics['accuracy'] * 100:.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; margin-top: 4px;">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="panel-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{metrics['precision_macro'] * 100:.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; margin-top: 4px;">Precision (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="panel-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{metrics['recall_macro'] * 100:.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; margin-top: 4px;">Recall (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="panel-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{metrics['f1_macro'] * 100:.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; margin-top: 4px;">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Confusion Matrix (12,000 Test Images)</div>', unsafe_allow_html=True)
        cm_path = ASSETS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix available in assets.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Training & Validation Loss / Accuracy</div>', unsafe_allow_html=True)
        curves_path = ASSETS_DIR / "accuracy_loss_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)
        else:
            st.info("Curves available in assets.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Classification Performance Table</div>', unsafe_allow_html=True)
    df_metrics = pd.DataFrame({
        "Class": ["AI (Synthetic)", "REAL (CIFAR-10)", "Macro Average", "Weighted Average"],
        "Precision": ["89.80%", "95.09%", "92.42%", "92.45%"],
        "Recall": ["95.40%", "89.17%", "92.30%", "92.28%"],
        "F1-Score": ["92.52%", "92.03%", "92.24%", "92.28%"],
        "Support": ["6,000", "6,000", "12,000", "12,000"]
    })
    st.table(df_metrics)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model Architecture Summary</div>', unsafe_allow_html=True)
    st.code("""
Model: "sequential_resnet50_cifake"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 1, 1, 2048)        23,587,712
   [Pretrained ImageNet base, fine-tuned on CIFAKE]
 flatten (Flatten)           (None, 2048)              0         
 dense_256 (Dense)           (None, 256)               524,544   
 batch_norm (BatchNorm)      (None, 256)               1,024     
 dropout_0.5 (Dropout)       (None, 256)               0         
 dense_128 (Dense)           (None, 128)               32,896    
 dense_output (Dense)        (None, 1)                 129       
   [Activation: sigmoid -> P(REAL)]
=================================================================
Total parameters: 24,146,305 (92.11 MB)
Trainable parameters: 24,092,673 (91.91 MB)
Non-trainable parameters: 53,632 (209.50 KB)
_________________________________________________________________
Optimization Configuration:
• Optimizer: Adam (initial_lr = 0.001)
• Learning Rate Decay: Exponential schedule (decay = 0.95 per epoch)
• Regularization: Batch Normalization + Dropout (0.5)
• Loss Objective: Binary Cross-Entropy
""", language="text")
    st.markdown('</div>', unsafe_allow_html=True)


def page_batch_evaluation(threshold: float):
    st.markdown("""
    <div class="app-badge">
        <span class="app-badge-dot"></span> Batch Processing
    </div>
    <div class="main-title">Batch Media Evaluator</div>
    <div class="main-subtitle">Upload multiple images at once to generate a summary report and export results as CSV.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Select images to analyze:", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Analyzing {len(uploaded_files)} images...")
        results = []
        progress_bar = st.progress(0.0)

        for idx, f in enumerate(uploaded_files):
            try:
                img = Image.open(f).convert("RGB")
                res = predict_image(np.array(img), threshold=threshold)
                results.append({
                    "Filename": f.name,
                    "Predicted Class": res["label"],
                    "Confidence": f"{res['confidence'] * 100:.1f}%",
                    "P(Real)": f"{res['probabilities']['REAL']:.4f}",
                    "P(AI)": f"{res['probabilities']['AI']:.4f}",
                    "Latency (ms)": res["processing_time_ms"],
                    "Diagnostic": res["explanation"]
                })
            except Exception as e:
                results.append({
                    "Filename": f.name,
                    "Predicted Class": "ERROR",
                    "Confidence": "N/A",
                    "P(Real)": "N/A",
                    "P(AI)": "N/A",
                    "Latency (ms)": 0,
                    "Diagnostic": str(e)
                })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        n_real = sum(df["Predicted Class"] == "REAL")
        n_ai = sum(df["Predicted Class"] == "AI-GENERATED")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Processed", len(df))
        with col_s2:
            st.metric("Real Photos", n_real)
        with col_s3:
            st.metric("AI Images", n_ai)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV Report", data=csv_data, file_name="cifake_batch_results.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    st.markdown("""
    <div class="app-badge">
        <span class="app-badge-dot"></span> Documentation
    </div>
    <div class="main-title">About the CIFAKE Project</div>
    <div class="main-subtitle">Benchmark dataset and methodology for synthetic image detection.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card" style="line-height: 1.6; color: #cbd5e1; font-size: 0.92rem;">
        <h4 style="color: #f8fafc; margin-top: 0;">Project Background</h4>
        <p>
        As generative diffusion models (Stable Diffusion, Midjourney, DALL-E) become increasingly photorealistic, 
        verifiable authentication of digital media is essential to combat synthetic misinformation. 
        This application implements transfer learning and frequency-domain analysis to evaluate images against physical 
        camera optics and generative noise signatures.
        </p>

        <h4 style="color: #f8fafc; margin-top: 18px;">CIFAKE Dataset</h4>
        <p>
        • <b>60,000 Real Images</b>: CIFAR-10 photographic benchmark across 10 classes.<br>
        • <b>60,000 Synthetic Images</b>: Generated with Stable Diffusion 1.4 using matched semantic prompts.<br>
        • <b>Data Split</b>: 100,000 training images (50k Real, 50k Fake) and 20,000 test images (10k Real, 10k Fake).
        </p>

        <h4 style="color: #f8fafc; margin-top: 18px;">Forensic Detection Methodology</h4>
        <p>
        1. <b>Flat-Region Sensor Noise Estimation</b>: Measures physical Poisson-Gaussian shot noise in smooth areas to avoid edge aliasing false positives.<br>
        2. <b>Color Channel Cross-Correlation</b>: Validates physical Bayer demosaicing optical coupling between color channels.<br>
        3. <b>2D FFT Spectrum Analysis</b>: Evaluates off-axis periodic harmonics from transposed convolution neural upsampling.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    page, threshold, show_fft = render_sidebar()
    if page == "Detector":
        page_detector(threshold, show_fft)
    elif page == "Frequency Studio":
        page_frequency_studio()
    elif page == "Model & Benchmarks":
        page_architecture()
    elif page == "Batch Evaluation":
        page_batch_evaluation(threshold)
    elif page == "About CIFAKE":
        page_about()


if __name__ == "__main__":
    main()
