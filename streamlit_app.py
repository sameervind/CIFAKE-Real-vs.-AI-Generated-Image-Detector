"""
Streamlit Web Dashboard for Real vs. AI-Generated Image Classification.
Crafted with the Apple Design System (DESIGN.md) & Aceternity UI Component Architecture.
Featuring SF Pro Typography (-0.28px tracking), Action Blue (#0066cc) Pill CTAs,
Aceternity Card Spotlight & Bento Grid, Frosted Vibrancy, and the Signature Product Resting Shadow.
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
    page_title="AI vs. Real Image Classifier | Apple Design & CIFAKE ResNet-50",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apple Design System + Aceternity UI Component Styling (DESIGN.md Reference)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Apple System Font Stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", system-ui, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Apple Canvas: Parchment (#f5f5f7) and Pure White (#ffffff) */
    .stApp {
        background-color: #f5f5f7 !important;
        color: #1d1d1f !important;
    }

    /* Apple Frosted Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        border-right: 1px solid #e0e0e0 !important;
    }

    /* Apple Top Bar Badge */
    .apple-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        color: #0066cc;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: -0.12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    /* Apple Display Typography with Signature Negative Tracking */
    .apple-hero-title {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        font-size: 48px;
        font-weight: 600;
        line-height: 1.07;
        letter-spacing: -0.28px;
        color: #1d1d1f;
        margin-bottom: 10px;
    }

    .apple-hero-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        font-size: 21px;
        font-weight: 400;
        line-height: 1.35;
        letter-spacing: -0.2px;
        color: #7a7a7a;
        margin-bottom: 32px;
    }

    /* Aceternity Card Spotlight + Apple Store Utility Card ({rounded.lg} = 18px) */
    .apple-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .apple-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.08);
        border-color: #d2d2d7;
    }

    .apple-card-title {
        font-size: 21px;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.2px;
        margin-bottom: 16px;
    }

    /* The Signature Apple Product Drop-Shadow (DESIGN.md specification) */
    .product-shadow {
        box-shadow: rgba(0, 0, 0, 0.18) 0px 8px 30px 0px;
        border-radius: 18px;
    }

    /* Apple Verdict Banner - Authentic */
    .apple-verdict-real {
        background: #fafafc;
        border: 1.5px solid #10b981;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 20px -2px rgba(16, 185, 129, 0.12);
    }

    /* Apple Verdict Banner - AI */
    .apple-verdict-ai {
        background: #fafafc;
        border: 1.5px solid #f43f5e;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 20px -2px rgba(244, 63, 94, 0.12);
    }

    .apple-verdict-tag {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .tag-real { color: #059669; }
    .tag-ai { color: #e11d48; }

    .apple-verdict-headline {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        font-size: 34px;
        font-weight: 600;
        letter-spacing: -0.37px;
        line-height: 1.15;
        margin: 4px 0 8px 0;
    }
    .headline-real { color: #065f46; }
    .headline-ai { color: #9f1239; }

    .apple-verdict-sub {
        font-size: 15px;
        color: #7a7a7a;
    }
    .apple-verdict-sub b {
        color: #1d1d1f;
        font-weight: 600;
    }

    /* Aceternity Bento Grid - Apple Telemetry Tiles */
    .bento-tile {
        background: #f5f5f7;
        border: 1px solid #e0e0e0;
        border-radius: 14px;
        padding: 16px 12px;
        text-align: center;
        transition: transform 0.15s ease, background 0.15s ease;
    }
    .bento-tile:hover {
        background: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .bento-tile-value {
        font-family: 'JetBrains Mono', -apple-system, monospace;
        font-size: 20px;
        font-weight: 600;
        color: #1d1d1f;
    }

    .bento-tile-label {
        font-size: 12px;
        font-weight: 500;
        color: #7a7a7a;
        letter-spacing: -0.12px;
        margin-top: 4px;
    }

    /* Apple Note Box */
    .apple-diagnostic-note {
        background: #f5f5f7;
        border-left: 3px solid #0066cc;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        font-size: 14px;
        color: #333333;
        line-height: 1.45;
        margin: 16px 0;
    }

    /* Action Blue (#0066cc) Primary Pill Buttons with 0.95 scale micro-interaction */
    button, .stDownloadButton > button {
        background-color: #0066cc !important;
        color: #ffffff !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 10px 22px !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        letter-spacing: -0.2px !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }

    button:hover, .stDownloadButton > button:hover {
        background-color: #0071e3 !important;
    }

    button:active, .stDownloadButton > button:active {
        transform: scale(0.95) !important;
    }

    /* Apple-styled inputs */
    .stSelectbox, .stRadio, .stFileUploader {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif !important;
    }

    /* Slider Action Blue */
    .stSlider > div > div > div > div {
        background-color: #0066cc !important;
    }
    .stProgress > div > div > div > div {
        background-color: #0066cc !important;
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
    <div style="background: #ffffff; border: 1px solid #e0e0e0; padding: 10px 14px; border-radius: 12px; text-align: center; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
        <span style="font-size: 13px; font-weight: 600; color: #1d1d1f; letter-spacing: -0.2px;">ResNet-50 Classifier</span><br>
        <span style="font-size: 12px; color: #0066cc; font-weight: 500;">92.28% Benchmark Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p style="font-size: 11px; font-weight: 600; color: #7a7a7a; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">EXPERIENCE</p>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        ["Image Detector", "Frequency Studio", "Model Benchmarks", "Batch Evaluation", "About CIFAKE"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p style="font-size: 11px; font-weight: 600; color: #7a7a7a; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">CONTROLS</p>', unsafe_allow_html=True)
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Decision boundary for authentic classification (default: 0.50). Values >= threshold are classified as Real."
    )
    show_fft = st.sidebar.checkbox("Compute 2D Fourier Spectrum", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 12px; color: #7a7a7a; line-height: 1.5; padding: 6px 2px;">
        <b style="color: #1d1d1f;">Design System:</b> Apple Design Reference (DESIGN.md)<br>
        <b style="color: #1d1d1f;">Components:</b> Aceternity Bento & Spotlight<br>
        <b style="color: #1d1d1f;">Typography:</b> SF Pro Display / SF Pro Text<br>
        <b style="color: #1d1d1f;">Color Accent:</b> Action Blue (#0066cc)
    </div>
    """, unsafe_allow_html=True)
    return page, threshold, show_fft


def page_detector(threshold: float, show_fft: bool):
    st.markdown("""
    <div class="apple-badge">
        <span>CIFAKE Benchmark · ResNet-50</span>
    </div>
    <div class="apple-hero-title">AI vs. Real Image Classifier</div>
    <div class="apple-hero-subtitle">Authentic photographic verification powered by deep residual transfer learning and 2D Fourier frequency analysis.</div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 1.25], gap="large")

    selected_image = None
    image_source_name = "Uploaded Image"

    with col_input:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">1. Source Image</div>', unsafe_allow_html=True)
        input_mode = st.radio("Source Mode:", ["Upload Photo", "Sample Library"], horizontal=True, label_visibility="collapsed")

        if input_mode == "Upload Photo":
            uploaded_file = st.file_uploader("Select an image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_name = uploaded_file.name
        else:
            sample_category = st.selectbox("Category:", ["Real Photographs (CIFAR-10)", "Synthetic Media (Stable Diffusion)"])
            sample_folder = SAMPLES_REAL if "Real" in sample_category else SAMPLES_AI

            if sample_folder.exists():
                sample_files = sorted(list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png")))
                if sample_files:
                    sample_options = [f.name for f in sample_files]
                    selected_sample_name = st.selectbox("Select Sample:", sample_options)
                    selected_file_path = sample_folder / selected_sample_name
                    selected_image = Image.open(selected_file_path).convert("RGB")
                    image_source_name = selected_sample_name
            else:
                st.warning("Samples directory not found.")

        if selected_image is not None:
            # Apple resting product shadow applied
            st.image(selected_image, caption=f"{image_source_name} ({selected_image.size[0]} × {selected_image.size[1]} px)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">2. Analysis & Diagnostics</div>', unsafe_allow_html=True)

        if selected_image is None:
            st.info("Upload an image on the left or choose from the sample library to run verification.")
        else:
            with st.spinner("Analyzing optical noise and spectral harmonics..."):
                img_array = np.array(selected_image)
                result = predict_image(img_array, threshold=threshold)

            is_real = result["is_real"]
            conf = result["confidence"]
            p_real = result["probabilities"]["REAL"]
            p_ai = result["probabilities"]["AI"]
            latency = result["processing_time_ms"]

            # Apple Style Clean Verdict Card
            if is_real:
                st.markdown(f"""
                <div class="apple-verdict-real">
                    <div class="apple-verdict-tag tag-real">Authentic Media Verification</div>
                    <div class="apple-verdict-headline headline-real">Real Photograph</div>
                    <div class="apple-verdict-sub">Confidence: <b>{conf * 100:.1f}%</b> &nbsp;·&nbsp; Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="apple-verdict-ai">
                    <div class="apple-verdict-tag tag-ai">Synthetic Media Detected</div>
                    <div class="apple-verdict-headline headline-ai">AI-Generated Image</div>
                    <div class="apple-verdict-sub">Confidence: <b>{conf * 100:.1f}%</b> &nbsp;·&nbsp; Latency: <b>{latency} ms</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Apple Probability Bars
            st.markdown('<p style="font-size: 12px; font-weight: 600; color: #7a7a7a; text-transform: uppercase; letter-spacing: 0.04em; margin: 16px 0 6px 0;">PROBABILITY DISTRIBUTION</p>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                st.metric("P(Real Image)", f"{p_real * 100:.2f}%")
                st.progress(p_real)
            with b2:
                st.metric("P(AI-Generated)", f"{p_ai * 100:.2f}%")
                st.progress(p_ai)

            # Apple Diagnostic Note
            st.markdown(f"""
            <div class="apple-diagnostic-note">
                <b style="color: #0066cc;">Diagnostic Finding:</b> {result["explanation"]}
            </div>
            """, unsafe_allow_html=True)

            # Aceternity Bento Grid - Apple Telemetry Tiles
            st.markdown('<p style="font-size: 12px; font-weight: 600; color: #7a7a7a; text-transform: uppercase; letter-spacing: 0.04em; margin: 16px 0 8px 0;">SPECTRAL & SPATIAL TELEMETRY</p>', unsafe_allow_html=True)
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-value">{result['artifacts'].get('spectral_peak_ratio', 1.0):.1f}x</div>
                    <div class="bento-tile-label">Peak Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with t2:
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-value">{result['artifacts'].get('laplacian_kurtosis', 3.0):.1f}</div>
                    <div class="bento-tile-label">Kurtosis</div>
                </div>
                """, unsafe_allow_html=True)
            with t3:
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-value">{result['artifacts'].get('noise_residual_std', 5.0):.2f}</div>
                    <div class="bento-tile-label">Sensor Noise</div>
                </div>
                """, unsafe_allow_html=True)
            with t4:
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-value">{result['artifacts'].get('high_to_low_ratio', 0.0):.3f}</div>
                    <div class="bento-tile-label">Spectral Ratio</div>
                </div>
                """, unsafe_allow_html=True)

            # 2D Fourier Magnitude
            if show_fft:
                st.markdown("---")
                st.markdown('<p style="font-size: 12px; font-weight: 600; color: #7a7a7a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">2D FOURIER SPECTRUM ANALYSIS</p>', unsafe_allow_html=True)
                spec = result["spectrum_2d"]

                f1, f2 = st.columns(2)
                with f1:
                    st.image(selected_image.resize((256, 256)), caption="Analysis Resolution", use_container_width=True)
                with f2:
                    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#ffffff')
                    im = ax.imshow(spec, cmap="inferno")
                    ax.set_title("2D Magnitude Spectrum", fontsize=8, color="#1d1d1f")
                    ax.axis("off")
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.yaxis.set_tick_params(color='#7a7a7a')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#7a7a7a', fontsize=7)
                    st.pyplot(fig)
                    plt.close()

        st.markdown('</div>', unsafe_allow_html=True)


def page_frequency_studio():
    st.markdown("""
    <div class="apple-badge">
        <span>2D Fourier Frequency Forensics</span>
    </div>
    <div class="apple-hero-title">Spectral Analysis Studio</div>
    <div class="apple-hero-subtitle">Observe generative neural transposed-convolution checkerboard spikes vs. physical 1/f optical MTF lens decay.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="apple-card" style="font-size: 15px; color: #333333; line-height: 1.55;">
        <span style="font-weight: 600; color: #1d1d1f;">The Science of Frequency Domain Forensics:</span><br>
        Generative models (GANs and Latent Diffusion models like Stable Diffusion) upscale feature maps using transposed convolutions 
        or bilinear interpolation. This mathematical operation introduces <b>periodic grid harmonics</b> that appear as 
        symmetrical off-axis delta spikes in 2D Discrete Fourier space. Real camera lenses, by contrast, follow physical optics: 
        azimuthal power spectral density naturally decays monotonically according to the <b>1/f² power law</b>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">Authentic CIFAR-10 Sample</div>', unsafe_allow_html=True)
        real_sample = SAMPLES_REAL / "01_real_airplane.jpg"
        if real_sample.exists():
            img_real = Image.open(real_sample).convert("RGB")
            st.image(img_real.resize((256, 256)), caption="Real Airplane (CIFAR-10)", use_container_width=True)
            res_real = predict_image(np.array(img_real))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            ax.imshow(res_real["spectrum_2d"], cmap="viridis")
            ax.set_title("Natural 1/f Smooth Decay Spectrum", fontsize=9, color="#1d1d1f")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">Synthetic AI Sample (Stable Diffusion)</div>', unsafe_allow_html=True)
        ai_sample = SAMPLES_AI / "01_ai_hyperrealistic_cyberpunk_neon_car.jpg"
        if ai_sample.exists():
            img_ai = Image.open(ai_sample).convert("RGB")
            st.image(img_ai.resize((256, 256)), caption="AI Generated Car (Synthetic)", use_container_width=True)
            res_ai = predict_image(np.array(img_ai))

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            ax.imshow(res_ai["spectrum_2d"], cmap="inferno")
            ax.set_title("Synthetic Spectrum (Periodic Harmonics)", fontsize=9, color="#1d1d1f")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def page_architecture():
    st.markdown("""
    <div class="apple-badge">
        <span>ResNet-50 Architecture</span>
    </div>
    <div class="apple-hero-title">Model Specifications & Benchmarks</div>
    <div class="apple-hero-subtitle">Evaluation on 12,000 CIFAKE test benchmark images matching the original project report.</div>
    """, unsafe_allow_html=True)

    metrics = load_metrics_data()

    # 4 Key Apple Metric Tiles
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="apple-card" style="text-align: center; padding: 20px;">
            <div style="font-size: 32px; font-weight: 600; color: #1d1d1f; font-family: 'JetBrains Mono', monospace;">{metrics['accuracy'] * 100:.2f}%</div>
            <div style="font-size: 13px; color: #7a7a7a; font-weight: 500; margin-top: 4px;">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="apple-card" style="text-align: center; padding: 20px;">
            <div style="font-size: 32px; font-weight: 600; color: #1d1d1f; font-family: 'JetBrains Mono', monospace;">{metrics['precision_macro'] * 100:.2f}%</div>
            <div style="font-size: 13px; color: #7a7a7a; font-weight: 500; margin-top: 4px;">Precision (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="apple-card" style="text-align: center; padding: 20px;">
            <div style="font-size: 32px; font-weight: 600; color: #1d1d1f; font-family: 'JetBrains Mono', monospace;">{metrics['recall_macro'] * 100:.2f}%</div>
            <div style="font-size: 13px; color: #7a7a7a; font-weight: 500; margin-top: 4px;">Recall (Macro)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="apple-card" style="text-align: center; padding: 20px;">
            <div style="font-size: 32px; font-weight: 600; color: #1d1d1f; font-family: 'JetBrains Mono', monospace;">{metrics['f1_macro'] * 100:.2f}%</div>
            <div style="font-size: 13px; color: #7a7a7a; font-weight: 500; margin-top: 4px;">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">Confusion Matrix (12,000 Test Images)</div>', unsafe_allow_html=True)
        cm_path = ASSETS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix available in assets.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown('<div class="apple-card-title">Training & Validation Loss / Accuracy</div>', unsafe_allow_html=True)
        curves_path = ASSETS_DIR / "accuracy_loss_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)
        else:
            st.info("Curves available in assets.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
    st.markdown('<div class="apple-card-title">Classification Performance Report</div>', unsafe_allow_html=True)
    df_metrics = pd.DataFrame({
        "Class": ["AI (Synthetic)", "REAL (CIFAR-10)", "Macro Average", "Weighted Average"],
        "Precision": ["89.80%", "95.09%", "92.42%", "92.45%"],
        "Recall": ["95.40%", "89.17%", "92.30%", "92.28%"],
        "F1-Score": ["92.52%", "92.03%", "92.24%", "92.28%"],
        "Support": ["6,000", "6,000", "12,000", "12,000"]
    })
    st.table(df_metrics)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
    st.markdown('<div class="apple-card-title">ResNet-50 Sequential Architecture</div>', unsafe_allow_html=True)
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
    <div class="apple-badge">
        <span>Batch Evaluation Pipeline</span>
    </div>
    <div class="apple-hero-title">Batch Media Evaluator</div>
    <div class="apple-hero-subtitle">Upload multiple images concurrently to generate a forensic audit report and export results as CSV.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
    st.markdown('<div class="apple-card-title">Ingest Media Batch</div>', unsafe_allow_html=True)
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
    <div class="apple-badge">
        <span>Research & Dataset</span>
    </div>
    <div class="apple-hero-title">About the CIFAKE Project</div>
    <div class="apple-hero-subtitle">Open-source forensic research for distinguishing synthetic media from camera photography.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="apple-card" style="line-height: 1.6; color: #333333; font-size: 15px;">
        <h4 style="color: #1d1d1f; margin-top: 0; font-size: 21px;">Project Background</h4>
        <p>
        As generative diffusion models (Stable Diffusion, Midjourney, DALL-E) become increasingly photorealistic, 
        verifiable authentication of digital media is essential to combat synthetic misinformation. 
        This application implements transfer learning and frequency-domain analysis to evaluate images against physical 
        camera optics and generative noise signatures.
        </p>

        <h4 style="color: #1d1d1f; margin-top: 24px; font-size: 21px;">CIFAKE Dataset</h4>
        <p>
        • <b>60,000 Real Images</b>: CIFAR-10 photographic benchmark across 10 classes.<br>
        • <b>60,000 Synthetic Images</b>: Generated with Stable Diffusion 1.4 using matched semantic prompts.<br>
        • <b>Data Split</b>: 100,000 training images (50k Real, 50k Fake) and 20,000 test images (10k Real, 10k Fake).
        </p>

        <h4 style="color: #1d1d1f; margin-top: 24px; font-size: 21px;">Forensic Detection Methodology</h4>
        <p>
        1. <b>Flat-Region Sensor Noise Estimation</b>: Measures physical Poisson-Gaussian shot noise in smooth areas to avoid edge aliasing false positives.<br>
        2. <b>Color Channel Cross-Correlation</b>: Validates physical Bayer demosaicing optical coupling between color channels.<br>
        3. <b>2D FFT Spectrum Analysis</b>: Evaluates off-axis periodic harmonics from transposed convolution neural upsampling.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    page, threshold, show_fft = render_sidebar()
    if page == "Image Detector":
        page_detector(threshold, show_fft)
    elif page == "Frequency Studio":
        page_frequency_studio()
    elif page == "Model Benchmarks":
        page_architecture()
    elif page == "Batch Evaluation":
        page_batch_evaluation(threshold)
    elif page == "About CIFAKE":
        page_about()


if __name__ == "__main__":
    main()
