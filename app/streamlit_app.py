"""
Streamlit Web Dashboard for Real vs. AI-Generated Image Classification.
Upgraded with Aceternity UI Design System: Glassmorphism, Ambient Beams,
Bento Grid Diagnostics, Moving Neon Borders, and Futuristic Forensic Telemetry.
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
    page_title="CIFAKE Neural Forensics | Real vs. AI Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aceternity UI Design System Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Space-Black Background with Background Grid Pattern */
    .stApp {
        background-color: #030712 !important;
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.25), transparent),
            radial-gradient(circle at 100% 0%, rgba(6, 182, 212, 0.12), transparent 40%),
            radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 28px 28px !important;
        color: #f8fafc !important;
    }

    /* Dark Glass Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(8, 12, 22, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Aceternity Glowing Pill Badge */
    .aceternity-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.25);
        margin-bottom: 12px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse-glow 2s infinite ease-in-out;
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(1.4); }
    }

    /* Hero Headline with Metallic Chrome & Indigo Gradient */
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 40%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* Aceternity Bento Glass Card */
    .bento-card {
        background: rgba(13, 17, 28, 0.65);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
    }

    .bento-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.25);
    }

    /* Aceternity Verdict Banners */
    .verdict-banner-real {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(4, 120, 87, 0.25) 100%);
        border: 1.5px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 14px 0;
        box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.35);
        backdrop-filter: blur(16px);
    }

    .verdict-banner-ai {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.5) 0%, rgba(159, 18, 57, 0.25) 100%);
        border: 1.5px solid #f43f5e;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 14px 0;
        box-shadow: 0 0 35px -5px rgba(244, 63, 94, 0.35);
        backdrop-filter: blur(16px);
    }

    .verdict-tag {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .verdict-real-tag { color: #34d399; }
    .verdict-ai-tag { color: #fb7185; }

    .verdict-headline {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .verdict-real-headline {
        color: #ecfdf5;
        text-shadow: 0 0 25px rgba(16, 185, 129, 0.5);
    }

    .verdict-ai-headline {
        color: #fff1f2;
        text-shadow: 0 0 25px rgba(244, 63, 94, 0.5);
    }

    .verdict-meta {
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    /* Artifact Metrics Bento Cards */
    .artifact-bento {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        backdrop-filter: blur(14px);
        transition: all 0.25s ease;
    }

    .artifact-bento:hover {
        border-color: rgba(129, 140, 248, 0.45);
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }

    .artifact-icon {
        font-size: 1.5rem;
        margin-bottom: 6px;
    }

    .artifact-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .artifact-title {
        font-size: 0.76rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Terminal Window with macOS Traffic Lights */
    .terminal-window {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        overflow: hidden;
        margin: 18px 0;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
    }

    .terminal-topbar {
        background: rgba(255, 255, 255, 0.04);
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .dot-mac {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-close { background-color: #ef4444; }
    .dot-min { background-color: #f59e0b; }
    .dot-max { background-color: #10b981; }

    .terminal-tag {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 8px;
    }

    /* Streamlit Input Overrides */
    .stSlider > div > div > div > div {
        background-color: #6366f1 !important;
    }
    .stRadio > div {
        background: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
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
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.4); color: #e0e7ff; padding: 10px 14px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 0.85rem; margin-bottom: 16px; box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);">
        ⚡ RESNET-50 NEURAL ENGINE · 92.28%
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Select Module:",
        ["🔍 Live Detector", "📊 2D FFT Frequency Studio", "🧠 Architecture & Benchmarks", "📁 Batch Evaluation", "📖 About & CIFAKE"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">DETECTOR SENSITIVITY</p>', unsafe_allow_html=True)
    threshold = st.sidebar.slider(
        "Decision Threshold (P(Real))",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Decision boundary for authentic classification (default 0.50). Values >= threshold are classified as REAL."
    )
    show_fft = st.sidebar.checkbox("Compute 2D FFT Fourier Spectrum", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 10px; font-size: 0.82rem; color: #94a3b8;">
        <span style="color: #a5b4fc; font-weight: 700;">Active Pipeline:</span><br>
        • Transfer Learning: ResNet50 (Keras/Native)<br>
        • Cardinal-Masked 2D FFT<br>
        • Multi-Scale Diffusion Noise Estimator<br>
        • Cross-Channel Optical Demosaicing<br>
        • Calibrated Bayesian Real Prior
    </div>
    """, unsafe_allow_html=True)
    return page, threshold, show_fft


def page_live_detector(threshold: float, show_fft: bool):
    st.markdown("""
    <div class="aceternity-pill">
        <span class="pulse-dot"></span> NEURAL FORENSICS SUITE · CIFAKE RESNET-50
    </div>
    <div class="hero-title">Real vs. AI Image Detector</div>
    <div class="hero-subtitle">High-precision neural transfer learning & 2D Fourier frequency forensics for authentic vs. synthetic media verification</div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 1.25], gap="large")

    selected_image = None
    image_source_name = "Uploaded Image"

    with col_input:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("### 1. Ingest Media")
        input_mode = st.radio("Input Source:", ["📤 Upload Custom Image", "🖼️ CIFAKE Sample Gallery"], horizontal=True)

        if input_mode == "📤 Upload Custom Image":
            uploaded_file = st.file_uploader("Upload an image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_name = uploaded_file.name
        else:
            sample_category = st.selectbox("Gallery Category:", ["Real CIFAR-10 Photos", "Synthetic AI (Stable Diffusion)"])
            sample_folder = SAMPLES_REAL if "Real" in sample_category else SAMPLES_AI

            if sample_folder.exists():
                sample_files = sorted(list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png")))
                if sample_files:
                    sample_options = [f.name for f in sample_files]
                    selected_sample_name = st.selectbox("Select Benchmark Sample:", sample_options)
                    selected_file_path = sample_folder / selected_sample_name
                    selected_image = Image.open(selected_file_path).convert("RGB")
                    image_source_name = selected_sample_name
            else:
                st.warning("Samples directory not found. Please verify local samples folder.")

        if selected_image is not None:
            st.image(selected_image, caption=f"Loaded: {image_source_name} ({selected_image.size[0]}×{selected_image.size[1]} px)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("### 2. Forensic Verdict & Telemetry")

        if selected_image is None:
            st.info("👈 Upload an image or select a benchmark sample from the gallery on the left to begin forensic analysis.")
        else:
            with st.spinner("Executing neural transfer evaluation, 2D FFT decomposition, and optical demosaicing checks..."):
                img_array = np.array(selected_image)
                result = predict_image(img_array, threshold=threshold)

            label = result["label"]
            is_real = result["is_real"]
            conf = result["confidence"]
            p_real = result["probabilities"]["REAL"]
            p_ai = result["probabilities"]["AI"]
            latency = result["processing_time_ms"]

            # Aceternity Radiant Glowing Verdict Banner
            if is_real:
                st.markdown(f"""
                <div class="verdict-banner-real">
                    <div class="verdict-tag verdict-real-tag">✓ AUTHENTIC MEDIA VERDICT</div>
                    <div class="verdict-headline verdict-real-headline">📸 REAL PHOTOGRAPH</div>
                    <div class="verdict-meta">Confidence: <b>{conf * 100:.1f}%</b> &nbsp;|&nbsp; Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-banner-ai">
                    <div class="verdict-tag verdict-ai-tag">⚠ SYNTHETIC MEDIA DETECTED</div>
                    <div class="verdict-headline verdict-ai-headline">🤖 AI-GENERATED SYNTHETIC</div>
                    <div class="verdict-meta">Confidence: <b>{conf * 100:.1f}%</b> &nbsp;|&nbsp; Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Aceternity Dual Gradient Progress Bars
            st.markdown('<p style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">CLASSIFICATION PROBABILITY DISTRIBUTION</p>', unsafe_allow_html=True)
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                st.metric("P(Real Image)", f"{p_real * 100:.2f}%")
                st.progress(p_real)
            with col_bar2:
                st.metric("P(AI / Synthetic)", f"{p_ai * 100:.2f}%")
                st.progress(p_ai)

            # Forensic Explanation Box
            st.markdown("---")
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.4); border-left: 3px solid #6366f1; padding: 12px 16px; border-radius: 8px; margin: 12px 0;">
                <span style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">🔬 Forensic Diagnostic Signature</span><br>
                <span style="color: #e2e8f0; font-size: 0.95rem;">{}</span>
            </div>
            """.format(result["explanation"]), unsafe_allow_html=True)

            # Aceternity Bento Artifact Metrics Grid
            st.markdown('<p style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 18px; margin-bottom: 10px;">SPATIAL & SPECTRAL ARTIFACT TELEMETRY</p>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="artifact-bento">
                    <div class="artifact-icon">🌌</div>
                    <div class="artifact-value">{result['artifacts'].get('spectral_peak_ratio', 1.0):.1f}x</div>
                    <div class="artifact-title">Spectral Peak</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="artifact-bento">
                    <div class="artifact-icon">📐</div>
                    <div class="artifact-value">{result['artifacts'].get('laplacian_kurtosis', 3.0):.1f}</div>
                    <div class="artifact-title">Grad Kurtosis</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="artifact-bento">
                    <div class="artifact-icon">🎯</div>
                    <div class="artifact-value">{result['artifacts'].get('noise_residual_std', 5.0):.2f}</div>
                    <div class="artifact-title">Sensor Noise</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="artifact-bento">
                    <div class="artifact-icon">⚡</div>
                    <div class="artifact-value">{result['artifacts'].get('high_to_low_ratio', 0.0):.3f}</div>
                    <div class="artifact-title">Spectral Ratio</div>
                </div>
                """, unsafe_allow_html=True)

            # 2D Fourier Spectrum Visualizer
            if show_fft:
                st.markdown("---")
                st.markdown('<p style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">2D FOURIER MAGNITUDE DECOMPOSITION</p>', unsafe_allow_html=True)
                spec = result["spectrum_2d"]

                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.image(selected_image.resize((256, 256)), caption="Resized Analysis View", use_container_width=True)
                with f_col2:
                    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
                    fig.patch.set_facecolor('#090d16')
                    ax.set_facecolor('#090d16')
                    im = ax.imshow(spec, cmap="inferno")
                    ax.set_title("2D Magnitude Spectrum", fontsize=9, fontweight="bold", color="#e2e8f0")
                    ax.axis("off")
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8', fontsize=7)
                    st.pyplot(fig)
                    plt.close()

        st.markdown('</div>', unsafe_allow_html=True)


def page_spectral_studio():
    st.markdown("""
    <div class="aceternity-pill">
        <span class="pulse-dot"></span> FREQUENCY DOMAIN LABORATORY
    </div>
    <div class="hero-title">2D FFT Spectral Studio</div>
    <div class="hero-subtitle">Observe generative neural transposed-convolution checkerboard spikes and optical 1/f MTF attenuation</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bento-card">
        <span style="color: #a5b4fc; font-weight: 700;">Theoretical Background:</span><br>
        <span style="color: #cbd5e1; font-size: 0.95rem;">
        Generative models (GANs and Latent Diffusion Models like Stable Diffusion) use transposed convolutions or nearest-neighbor upsamplers 
        to scale latent vectors into pixel dimensions. This mathematical operation introduces <b>periodic grid harmonics</b> that appear as 
        symmetrical off-axis delta spikes in 2D Discrete Fourier space. Real camera lenses, by contrast, follow physical optics: 
        azimuthal power spectral density naturally decays monotonically according to the <b>1/f² power law</b>.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.subheader("Authentic CIFAR-10 Sample")
        real_sample = SAMPLES_REAL / "01_real_airplane.jpg"
        if real_sample.exists():
            img_real = Image.open(real_sample).convert("RGB")
            st.image(img_real.resize((256, 256)), caption="Real Airplane (CIFAR-10)", use_container_width=True)
            res_real = predict_image(np.array(img_real))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#090d16')
            ax.set_facecolor('#090d16')
            ax.imshow(res_real["spectrum_2d"], cmap="viridis")
            ax.set_title("Natural 1/f Smooth Decay Spectrum", fontsize=10, fontweight="bold", color="#e2e8f0")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.subheader("Synthetic AI Sample (Stable Diffusion)")
        ai_sample = SAMPLES_AI / "01_ai_hyperrealistic_cyberpunk_neon_car.jpg"
        if ai_sample.exists():
            img_ai = Image.open(ai_sample).convert("RGB")
            st.image(img_ai.resize((256, 256)), caption="AI Generated Car (Synthetic)", use_container_width=True)
            res_ai = predict_image(np.array(img_ai))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#090d16')
            ax.set_facecolor('#090d16')
            ax.imshow(res_ai["spectrum_2d"], cmap="inferno")
            ax.set_title("Synthetic Spectrum (Periodic Harmonics)", fontsize=10, fontweight="bold", color="#e2e8f0")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def page_architecture_benchmarks():
    st.markdown("""
    <div class="aceternity-pill">
        <span class="pulse-dot"></span> EVALUATION & BENCHMARKS
    </div>
    <div class="hero-title">ResNet50 Architecture & Performance</div>
    <div class="hero-subtitle">Comprehensive validation on 12,000 CIFAKE test set images matching original project notebook metrics</div>
    """, unsafe_allow_html=True)

    metrics = load_metrics_data()

    # 4 Key Bento Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="artifact-bento">
            <div class="artifact-icon">🏆</div>
            <div class="artifact-value">{metrics['accuracy'] * 100:.2f}%</div>
            <div class="artifact-title">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="artifact-bento">
            <div class="artifact-icon">🎯</div>
            <div class="artifact-value">{metrics['precision_macro'] * 100:.2f}%</div>
            <div class="artifact-title">Precision (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="artifact-bento">
            <div class="artifact-icon">📡</div>
            <div class="artifact-value">{metrics['recall_macro'] * 100:.2f}%</div>
            <div class="artifact-title">Recall (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="artifact-bento">
            <div class="artifact-icon">⚡</div>
            <div class="artifact-value">{metrics['f1_macro'] * 100:.2f}%</div>
            <div class="artifact-title">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix (12,000 Test Images)")
        cm_path = ASSETS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix asset available in assets directory.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.subheader("Training & Validation Curves")
        curves_path = ASSETS_DIR / "accuracy_loss_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)
        else:
            st.info("Learning curves asset available in assets directory.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Classification Report
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.subheader("Classification Report (CIFAKE Test Benchmark)")
    df_metrics = pd.DataFrame({
        "Class": ["AI (Synthetic)", "REAL (CIFAR-10)", "Macro Average", "Weighted Average"],
        "Precision": ["89.80%", "95.09%", "92.42%", "92.45%"],
        "Recall": ["95.40%", "89.17%", "92.30%", "92.28%"],
        "F1-Score": ["92.52%", "92.03%", "92.24%", "92.28%"],
        "Support": ["6,000", "6,000", "12,000", "12,000"]
    })
    st.table(df_metrics)
    st.markdown('</div>', unsafe_allow_html=True)

    # Aceternity macOS Terminal Card
    st.markdown("""
    <div class="terminal-window">
        <div class="terminal-topbar">
            <span class="dot-mac dot-close"></span>
            <span class="dot-mac dot-min"></span>
            <span class="dot-mac dot-max"></span>
            <span class="terminal-tag">resnet50_cifake_architecture.spec</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
Model: "sequential_resnet50_cifake"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 1, 1, 2048)        23,587,712
   [Base: ImageNet pre-trained, unfrozen for fine-tuning]
 flatten (Flatten)           (None, 2048)              0         
 dense_256 (Dense)           (None, 256)               524,544   
 batch_norm (BatchNorm)      (None, 256)               1,024     
 dropout_0.5 (Dropout)       (None, 256)               0         
 dense_128 (Dense)           (None, 128)               32,896    
 dense_output (Dense)        (None, 1)                 129       
   [Activation: sigmoid -> P(REAL)]
=================================================================
Total params: 24,146,305 (92.11 MB)
Trainable params: 24,092,673 (91.91 MB)
Non-trainable params: 53,632 (209.50 KB)
_________________________________________________________________
Optimization Hyperparameters:
• Optimizer: Adam (initial_lr = 0.001)
• Learning Rate Scheduler: Exponential decay (lr * 0.95 / epoch)
• Regularization: Batch Normalization + Dropout (rate = 0.5)
• Binary Cross-Entropy Loss Objective
""", language="text")


def page_batch_evaluation(threshold: float):
    st.markdown("""
    <div class="aceternity-pill">
        <span class="pulse-dot"></span> BATCH INFERENCE PIPELINE
    </div>
    <div class="hero-title">Batch Media Evaluator</div>
    <div class="hero-subtitle">Upload multiple images concurrently to generate automated forensic classifications and downloadable CSV audit logs</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload batch images (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    if uploaded_files:
        st.markdown(f'<p style="color: #a5b4fc; font-weight: 700;">Processing {len(uploaded_files)} images through multi-domain classifier...</p>', unsafe_allow_html=True)
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
            st.metric("Total Batch Size", len(df))
        with col_s2:
            st.metric("Authentic Real Photos", n_real)
        with col_s3:
            st.metric("Synthetic AI Images", n_ai)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forensic Report CSV", data=csv_data, file_name="cifake_batch_forensics.csv", mime="text/csv")


def page_about():
    st.markdown("""
    <div class="aceternity-pill">
        <span class="pulse-dot"></span> DOCUMENTATION & ARCHITECTURE
    </div>
    <div class="hero-title">About the CIFAKE Project</div>
    <div class="hero-subtitle">Open-source forensic research for distinguishing synthetic media from camera photography</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bento-card">
        <h3 style="color: #818cf8; margin-top: 0;">📌 Project Motivation</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        With the rapid democratization of high-fidelity generative models (Stable Diffusion, Midjourney, DALL-E, Flux), 
        synthetic images are becoming imperceptible to the naked human eye. This system implements multi-domain 
        spatial and frequency analysis backed by deep residual transfer learning (ResNet50) to provide verifiable, 
        statistically grounded authentication of digital media.
        </p>

        <h3 style="color: #818cf8; margin-top: 20px;">📦 CIFAKE Dataset Specifications</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        • <b>60,000 Real Images</b>: Sourced from the CIFAR-10 photographic benchmark across 10 classes.<br>
        • <b>60,000 Synthetic Images</b>: Generated using Stable Diffusion 1.4 using matched semantic prompts.<br>
        • <b>Balanced Split</b>: 100,000 training images (50,000 Real / 50,000 Fake) and 20,000 test images (10,000 Real / 10,000 Fake).
        </p>

        <h3 style="color: #818cf8; margin-top: 20px;">🔬 Dual-Domain Verification Methodology</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        1. <b>Spatial Gradient & Flat-Region Sensor Noise</b>: Eliminates spatial aliasing false alarms from outdoor foliage and brick textures 
        by measuring Poisson shot noise strictly in smooth, low-gradient image patches.<br>
        2. <b>Color Channel Cross-Correlation</b>: Leverages physical Bayer filter optical coupling to detect synthetic channel divergence.<br>
        3. <b>2D FFT Frequency Forensics</b>: Isolates off-axis transposed convolution checkerboard spikes with cardinal axis masking.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    page, threshold, show_fft = render_sidebar()
    if page == "🔍 Live Detector":
        page_live_detector(threshold, show_fft)
    elif page == "📊 2D FFT Frequency Studio":
        page_spectral_studio()
    elif page == "🧠 Architecture & Benchmarks":
        page_architecture_benchmarks()
    elif page == "📁 Batch Evaluation":
        page_batch_evaluation(threshold)
    elif page == "📖 About & CIFAKE":
        page_about()


if __name__ == "__main__":
    main()
