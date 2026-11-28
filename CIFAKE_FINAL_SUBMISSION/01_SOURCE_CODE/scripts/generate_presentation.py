import sys
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
PRES_DIR = ROOT / "03_PRESENTATION"
PRES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT / "04_RESULTS"

# Color Palette: Lamborghini Dark Cockpit
BG_COLOR = RGBColor(15, 15, 17)        # #0F0F11
CARD_BG = RGBColor(24, 24, 28)         # #18181C
CARD_BORDER = RGBColor(45, 45, 53)     # #2D2D35
GOLD = RGBColor(255, 192, 0)           # #FFC000
WHITE = RGBColor(255, 255, 255)       # #FFFFFF
GRAY = RGBColor(180, 180, 190)         # #B4B4BE
MUTED = RGBColor(120, 120, 130)        # #787882
GREEN = RGBColor(52, 199, 89)          # #34C759
BLUE = RGBColor(10, 132, 255)          # #0A84FF

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

def apply_background(slide):
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_COLOR
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, title_text, category_text="CIFAKE // ACADEMIC DEFENSE"):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = GOLD
    p0.font.name = "Segoe UI"
    
    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.font.name = "Segoe UI"

def add_card(slide, left, top, width, height, title=None, border_color=CARD_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    
    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = GOLD
        p.font.name = "Segoe UI"
    return card

# ==============================================================================
# SLIDE 1: Title Slide
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
apply_background(s1)

tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.5))
tf = tb.text_frame
tf.word_wrap = True

p0 = tf.paragraphs[0]
p0.text = "CORSE CLIENTE // MACHINE LEARNING RESEARCH"
p0.font.size = Pt(12)
p0.font.bold = True
p0.font.color.rgb = GOLD
p0.font.name = "Segoe UI"

p1 = tf.add_paragraph()
p1.text = "CIFAKE: AI vs. Real Image Detection Platform"
p1.font.size = Pt(34)
p1.font.bold = True
p1.font.color.rgb = WHITE
p1.font.name = "Segoe UI"
p1.space_before = Pt(12)

p2 = tf.add_paragraph()
p2.text = "Native 32×32 CIFAR-CNN Architecture, Out-of-Distribution Gating & Frequency-Domain Diagnostics"
p2.font.size = Pt(16)
p2.font.color.rgb = GRAY
p2.font.name = "Segoe UI"
p2.space_before = Pt(8)

p3 = tf.add_paragraph()
p3.text = "Master of Science in Machine Learning // Final Academic Engineering Defense\nCPU-Optimized Inference • 94.92% Accuracy • 0.9903 ROC-AUC • 1.58 ms Latency"
p3.font.size = Pt(12)
p3.font.color.rgb = MUTED
p3.font.name = "Segoe UI"
p3.space_before = Pt(36)

# Badge Card on Slide 1
add_card(s1, Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.2), border_color=GOLD)
tb_badge = s1.shapes.add_textbox(Inches(1.2), Inches(5.5), Inches(10.9), Inches(1.0))
tf_b = tb_badge.text_frame
p_b1 = tf_b.paragraphs[0]
p_b1.text = "SYSTEM ARCHITECTURE: NATIVE CIFAR-CNN (305,953 PARAMS) + 2D FFT MTF + OOD ARBITER"
p_b1.font.size = Pt(11)
p_b1.font.bold = True
p_b1.font.color.rgb = GOLD
p_b2 = tf_b.add_paragraph()
p_b2.text = "Evaluated on Official CIFAKE Independent Benchmark (120,000 Images: 60k CIFAR-10 Photographic / 60k Stable Diffusion v1.4)"
p_b2.font.size = Pt(10)
p_b2.font.color.rgb = WHITE

# ==============================================================================
# SLIDE 2: Problem Statement & Motivation
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
apply_background(s2)
add_header(s2, "Problem Statement & Scientific Motivation", "01 // CONTEXT")

add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "The Generative Synthesis Crisis")
tb2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.3))
tf2_1 = tb2_1.text_frame
tf2_1.word_wrap = True
tf2_1.text = (
    "• Proliferation of Latent Diffusion: Modern text-to-image models (Stable Diffusion, Midjourney, DALL-E) "
    "produce hyperrealistic synthetic media that easily deceives the naked human eye.\n\n"
    "• Societal & Forensic Urgency: Synthetic imagery weaponized for digital disinformation, biometric spoofing, "
    "identity theft, and evidentiary contamination requires automated, mathematically defensible authentication.\n\n"
    "• The Benchmark Bottleneck: Standard benchmarks (e.g. CIFAKE) capture 32×32 generative artifacts, but real-world "
    "deployment encounters high-resolution, unconstrained imagery from modern camera lenses."
)
for p in tf2_1.paragraphs:
    p.font.size = Pt(12)
    p.font.color.rgb = GRAY

add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Core Project Objectives")
tb2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
tf2_2 = tb2_2.text_frame
tf2_2.word_wrap = True
tf2_2.text = (
    "1. High Benchmark Accuracy: Achieve reproducible >94% classification accuracy on the official 120,000 CIFAKE dataset.\n\n"
    "2. Extreme CPU Efficiency: Eliminate massive multi-million parameter models; target <5 ms inference latency on commodity CPU hardware.\n\n"
    "3. Real-World Domain Generalization: Prevent false-positive real classifications on modern diffusion models while preventing false-fake classifications on genuine high-res photographs.\n\n"
    "4. Transparent Multi-Signal Forensics: Integrate deep neural representations with 2D Fourier power spectra and out-of-distribution (OOD) telemetry."
)
for p in tf2_2.paragraphs:
    p.font.size = Pt(12)
    p.font.color.rgb = GRAY

# ==============================================================================
# SLIDE 3: Forensic Diagnostic Audit of Legacy Systems
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
apply_background(s3)
add_header(s3, "Root-Cause Failure Audit: Why Legacy Models Failed", "02 // DIAGNOSTICS")

card_w = Inches(3.7)
card_h = Inches(5.2)

add_card(s3, Inches(0.8), Inches(1.6), card_w, card_h, "Failure 1: Receptive Field Mismatch")
tb3_1 = s3.shapes.add_textbox(Inches(1.0), Inches(2.2), card_w - Inches(0.4), card_h - Inches(0.8))
tf3_1 = tb3_1.text_frame
tf3_1.word_wrap = True
tf3_1.text = (
    "• Destructive Upsampling:\n"
    "Legacy systems upscaled 32×32 images to 224×224 (49× spatial bloat), introducing severe interpolation artifacts and blurring microscopic generative cues.\n\n"
    "• Aggressive Stem Striding:\n"
    "Standard ResNet-50 uses 7×7 conv (stride 2) + 3×3 pool (stride 2), collapsing 32×32 inputs to 8×8 before stage 1, destroying delicate high-frequency diffusion fingerprints."
)
for p in tf3_1.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s3, Inches(4.8), Inches(1.6), card_w, card_h, "Failure 2: Uncalibrated Heuristics")
tb3_2 = s3.shapes.add_textbox(Inches(5.0), Inches(2.2), card_w - Inches(0.4), card_h - Inches(0.8))
tf3_2 = tb3_2.text_frame
tf3_2.word_wrap = True
tf3_2.text = (
    "• Arbitrary Formula Overrides:\n"
    "Heuristic formulas combined raw FFT peaks and Laplacian kurtosis with arbitrary constants, overriding the trained deep network.\n\n"
    "• False-Positive AI Predictions:\n"
    "Sharp authentic photographs with high natural texture triggered spectral peak overrides, causing genuine human portraits to be classified as AI with 98% false confidence."
)
for p in tf3_2.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s3, Inches(8.8), Inches(1.6), card_w, card_h, "Failure 3: OOD Conflation")
tb3_3 = s3.shapes.add_textbox(Inches(9.0), Inches(2.2), card_w - Inches(0.4), card_h - Inches(0.8))
tf3_3 = tb3_3.text_frame
tf3_3.word_wrap = True
tf3_3.text = (
    "• Domain Shift as Fake Signal:\n"
    "High-resolution external images naturally have larger scales than CIFAR-10. Legacy logic treated any high OOD score as proof of AI generation.\n\n"
    "• Mathematical Inversion:\n"
    "OOD measures epistemic uncertainty, not class likelihood. It must discount decision confidence rather than dictate class assignment."
)
for p in tf3_3.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

# ==============================================================================
# SLIDE 4: Proposed System Architecture
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
apply_background(s4)
add_header(s4, "Proposed Multi-Signal Decision Architecture", "03 // ARCHITECTURE")

add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Three-Tiered Defense Pipeline")
tb4 = s4.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(4.3))
tf4 = tb4.text_frame
tf4.word_wrap = True
tf4.text = (
    "TIER 1 — NATIVE CONVOLUTIONAL FEATURE EXTRACTION (CIFAR-CNN)\n"
    "• Processes images at native 32×32 spatial resolution using small 3×3 receptive fields with stride 1.\n"
    "• 6 convolutional stages with Batch Normalization, Dropout (0.2–0.4), and Global Average Pooling.\n"
    "• Extracts 128-dimensional dense manifold embedding and outputs calibrated P(REAL) in [0.0, 1.0].\n\n"
    "TIER 2 — OUT-OF-DISTRIBUTION (OOD) DOMAIN SHIFT ENGINE\n"
    "• Analyzes input resolution, aspect ratio, and latent feature distance relative to CIFAKE manifold.\n"
    "• Decouples domain novelty from class likelihood: high OOD score triggers margin discounting.\n"
    "• When OOD ambiguity is detected, requires strong deep model confidence (≥85%) or marks INCONCLUSIVE.\n\n"
    "TIER 3 — MULTI-SCALE PHYSICAL FORENSICS & FREQUENCY ANALYSIS\n"
    "• Computes 2D Discrete Fourier Transform (2D FFT) and radial power spectrum profile.\n"
    "• Evaluates sensor PRNU residual variance, Laplacian kurtosis, and texture plasticity index.\n"
    "• Serves strictly as supplementary diagnostic evidence for human analysts; zero hard override authority."
)
for p in tf4.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = WHITE if p.text.startswith("TIER") else GRAY
    if p.text.startswith("TIER"):
        p.font.bold = True
        p.font.color.rgb = GOLD

# ==============================================================================
# SLIDE 5: Dataset & Preprocessing Pipeline
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
apply_background(s5)
add_header(s5, "CIFAKE Dataset Specifications & Preprocessing", "04 // DATA PIPELINE")

add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Dataset Composition & Semantic Classes")
tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.3))
tf5_1 = tb5_1.text_frame
tf5_1.word_wrap = True
tf5_1.text = (
    "• Total Images: 120,000 balanced binary specimens\n"
    "• Class 0 (AI / Synthetic): 60,000 images generated via Stable Diffusion v1.4\n"
    "• Class 1 (Real Photography): 60,000 authentic images from CIFAR-10\n"
    "• Resolution: Native 32 × 32 pixels, 3 color channels (RGB)\n\n"
    "• 10 Matched Semantic Categories:\n"
    "  - Vehicles: Airplane, Automobile, Ship, Truck\n"
    "  - Animals: Bird, Cat, Deer, Dog, Frog, Horse\n\n"
    "• Strict Split Isolation:\n"
    "  - Train Partition: 100,000 images (50,000 Real / 50,000 AI)\n"
    "  - Test Partition: 20,000 images (10,000 Real / 10,000 AI)\n"
    "  - Test split strictly preserved and unaugmented."
)
for p in tf5_1.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Preprocessing & Anti-Leakage Controls")
tb5_2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
tf5_2 = tb5_2.text_frame
tf5_2.word_wrap = True
tf5_2.text = (
    "• Internalized Model Rescaling:\n"
    "Rescaling layer (1/255.0) is baked directly into the Keras model computation graph, eliminating training/inference preprocessing mismatch.\n\n"
    "• Physics-Preserving Augmentations:\n"
    "Restricted to horizontal flips and 4-pixel reflection padding with random cropping. Vertical flips were removed to preserve natural gravitational lighting.\n\n"
    "• Uniform Class Label Contract:\n"
    "Standardized Label 0 = AI / FAKE, Label 1 = REAL. Model sigmoid output represents P(REAL); P(AI) = 1.0 - P(REAL).\n\n"
    "• Zero Data Leakage:\n"
    "Augmentations applied dynamically on training batches only after dataset partitions were established."
)
for p in tf5_2.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

# ==============================================================================
# SLIDE 6: Model Implementation (CIFAR-CNN)
# ==============================================================================
s6 = prs.slides.add_slide(blank_layout)
apply_background(s6)
add_header(s6, "Production Model Architecture: CIFAR-CNN", "05 // DEEP LEARNING")

add_card(s6, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2), "Layer-by-Layer Architectural Specification")
tb6 = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.6), Inches(4.3))
tf6 = tb6.text_frame
tf6.word_wrap = True
tf6.text = (
    "• Input Layer: (None, 32, 32, 3) -> Rescaling(1/255.0)\n\n"
    "• Stage 1 (32x32 -> 16x16):\n"
    "  Conv2D(32, 3x3) -> BatchNorm -> ReLU\n"
    "  Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)\n\n"
    "• Stage 2 (16x16 -> 8x8):\n"
    "  Conv2D(64, 3x3) -> BatchNorm -> ReLU\n"
    "  Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.3)\n\n"
    "• Stage 3 (8x8 -> 4x4):\n"
    "  Conv2D(128, 3x3) -> BatchNorm -> ReLU\n"
    "  Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.4)\n\n"
    "• Classification Head:\n"
    "  GlobalAveragePooling2D -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.4) -> Dense(1, Sigmoid)"
)
for p in tf6.paragraphs:
    p.font.size = Pt(10.5)
    p.font.color.rgb = GRAY

add_card(s6, Inches(7.2), Inches(1.6), Inches(5.3), Inches(5.2), "Key Architectural Metrics")
tb6_r = s6.shapes.add_textbox(Inches(7.4), Inches(2.2), Inches(4.9), Inches(4.3))
tf6_r = tb6_r.text_frame
tf6_r.word_wrap = True
tf6_r.text = (
    "• Total Parameters: 305,953 (1.17 MB)\n"
    "• Trainable Parameters: 304,673\n"
    "• Non-trainable Parameters: 1,280 (BatchNorm)\n\n"
    "• Optimizer: Adam (lr = 1e-3, ReduceLROnPlateau)\n"
    "• Loss Function: Binary Cross-Entropy (BCE)\n"
    "• Regularization: L2 Weight Decay (1e-4) + Progressive Spatial Dropout\n\n"
    "• CPU Inference Latency: 1.58 ms / image\n"
    "• Throughput: >630 images / second on single CPU core\n\n"
    "• Hardware Optimization:\n"
    "AVX2, AVX-512, and oneDNN vectorization."
)
for p in tf6_r.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = WHITE if "• Total" in p.text or "• CPU" in p.text else GRAY

# ==============================================================================
# SLIDE 7: Technology Stack & Full-Stack Implementation
# ==============================================================================
s7 = prs.slides.add_slide(blank_layout)
apply_background(s7)
add_header(s7, "Technology Stack & Production Engineering", "06 // IMPLEMENTATION")

col_w = Inches(3.7)
col_h = Inches(5.2)

add_card(s7, Inches(0.8), Inches(1.6), col_w, col_h, "Deep Learning Core")
tb7_1 = s7.shapes.add_textbox(Inches(1.0), Inches(2.2), col_w - Inches(0.4), col_h - Inches(0.8))
tf7_1 = tb7_1.text_frame
tf7_1.word_wrap = True
tf7_1.text = (
    "• TensorFlow 2.x & Keras 3.x\n"
    "High-performance numerical graph execution and training.\n\n"
    "• NumPy & SciPy\n"
    "2D Discrete Fourier Transforms, radial frequency integration, and statistical kurtosis.\n\n"
    "• OpenCV & PIL\n"
    "Color space conversions, spatial resizing, and PRNU noise residual extraction."
)
for p in tf7_1.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s7, Inches(4.8), Inches(1.6), col_w, col_h, "Backend & API Architecture")
tb7_2 = s7.shapes.add_textbox(Inches(5.0), Inches(2.2), col_w - Inches(0.4), col_h - Inches(0.8))
tf7_2 = tb7_2.text_frame
tf7_2.word_wrap = True
tf7_2.text = (
    "• FastAPI REST Server\n"
    "Asynchronous, OpenAPI/Swagger-documented endpoints for single and batch predictions.\n\n"
    "• Uvicorn ASGI Worker\n"
    "Sub-millisecond routing and multipart streaming.\n\n"
    "• Pydantic V2 Models\n"
    "Strict type validation and dual probability response contracts."
)
for p in tf7_2.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s7, Inches(8.8), Inches(1.6), col_w, col_h, "Frontend & Testing Suite")
tb7_3 = s7.shapes.add_textbox(Inches(9.0), Inches(2.2), col_w - Inches(0.4), col_h - Inches(0.8))
tf7_3 = tb7_3.text_frame
tf7_3.word_wrap = True
tf7_3.text = (
    "• Streamlit Cockpit\n"
    "Lamborghini-inspired dark aesthetic (#0F0F11, #FFC000) with Aceternity Bento Grid.\n\n"
    "• Pytest QA Automation\n"
    "23/23 unit and regression tests passing with 100% pass rate in 32s.\n\n"
    "• Multi-Platform Launchers\n"
    "Native scripts: run_app.bat, run_app.ps1, run_app.sh."
)
for p in tf7_3.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

# ==============================================================================
# SLIDE 8: Evaluation Results & Confusion Matrix
# ==============================================================================
s8 = prs.slides.add_slide(blank_layout)
apply_background(s8)
add_header(s8, "Empirical Evaluation Results: Independent 20k Test Set", "07 // RESULTS")

# Embed Confusion Matrix
cm_img_path = ASSETS_DIR / "confusion_matrix.png"
if cm_img_path.exists():
    s8.shapes.add_picture(str(cm_img_path), Inches(0.8), Inches(1.6), width=Inches(5.6))

# Right card with metrics table
add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Independent Partition Verification Metrics")
tb8 = s8.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
tf8 = tb8.text_frame
tf8.word_wrap = True
tf8.text = (
    "• Overall Accuracy: 94.92% (18,984 / 20,000 correct)\n"
    "• Macro F1-Score: 0.9492 | Macro Precision: 0.9494\n"
    "• ROC-AUC Score: 0.9903 | PR-AUC: 0.9907\n"
    "• Test Binary Cross-Entropy Loss: 0.1298\n\n"
    "CONFUSION MATRIX BREAKDOWN (Threshold = 0.50):\n"
    "  - True AI (TN): 9,382 / 10,000 (Recall = 93.82%)\n"
    "  - False REAL (FP): 618 / 10,000 (FPR = 6.18%)\n"
    "  - False AI (FN): 398 / 10,000 (FNR = 3.98%)\n"
    "  - True REAL (TP): 9,602 / 10,000 (Recall = 96.02%)\n\n"
    "CLASS SEPARATION & CALIBRATION:\n"
    "  - Mean P(REAL) on True Real: 0.9437 (σ = 0.1638)\n"
    "  - Mean P(REAL) on True AI: 0.0839 (σ = 0.1965)\n"
    "  - Class Separation Margin: 0.8598"
)
for p in tf8.paragraphs:
    p.font.size = Pt(10.5)
    p.font.color.rgb = GOLD if "Overall Accuracy" in p.text or "CONFUSION" in p.text else GRAY

# ==============================================================================
# SLIDE 9: ROC and Precision-Recall Curves
# ==============================================================================
s9 = prs.slides.add_slide(blank_layout)
apply_background(s9)
add_header(s9, "Statistical Verification: ROC & PR Analysis", "08 // VALIDATION")

roc_img = ASSETS_DIR / "roc_curve.png"
if roc_img.exists():
    s9.shapes.add_picture(str(roc_img), Inches(0.8), Inches(1.6), width=Inches(5.6))

pr_img = ASSETS_DIR / "precision_recall_curve.png"
if pr_img.exists():
    s9.shapes.add_picture(str(pr_img), Inches(6.8), Inches(1.6), width=Inches(5.6))

# ==============================================================================
# SLIDE 10: Architecture Benchmark Comparison
# ==============================================================================
s10 = prs.slides.add_slide(blank_layout)
apply_background(s10)
add_header(s10, "Architectural Benchmarking & Efficiency Comparison", "09 // BENCHMARK")

arch_img = ASSETS_DIR / "architecture_comparison.png"
if arch_img.exists():
    s10.shapes.add_picture(str(arch_img), Inches(0.8), Inches(1.6), width=Inches(11.7))

# Summary text card below
add_card(s10, Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.4), "Benchmark Conclusion", border_color=GOLD)
tb10_b = s10.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.1))
tf10_b = tb10_b.text_frame
tf10_b.word_wrap = True
tf10_b.text = (
    "• Parameter Reduction: CIFAR-CNN achieves 79× fewer parameters than ResNet-50 (305k vs 24.1M).\n"
    "• Latency Acceleration: Inference speed increased by 26.6× (1.58 ms vs 42.1 ms per image on standard CPU).\n"
    "• Superior Accuracy: Native spatial convolutions beat upsampled ResNet-50 by +2.64% (94.92% vs 92.28%)."
)
for p in tf10_b.paragraphs:
    p.font.size = Pt(10.5)
    p.font.color.rgb = WHITE

# ==============================================================================
# SLIDE 11: Real-World Generalization & Regression Suite
# ==============================================================================
s11 = prs.slides.add_slide(blank_layout)
apply_background(s11)
add_header(s11, "Real-World Generalization & Regression Testing", "10 // REGRESSION")

add_card(s11, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Mandatory Real-World Regression Test Results")
tb11 = s11.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.3))
tf11 = tb11.text_frame
tf11.word_wrap = True
tf11.text = (
    "SPECIMEN 1: VERIFIED GENUINE PHOTOGRAPHIC PORTRAIT (Ground Truth: REAL)\n"
    "• Raw Neural Network Score: 99.96% REAL | Calibrated Decision Confidence: 91.2% REAL\n"
    "• OOD Telemetry: Score = 0.8717 (High resolution domain shift recognized, but NOT penalized as AI).\n"
    "• Status: PASSED (Eliminated critical false-positive AI classification bug).\n\n"
    "SPECIMEN 2: ONGC VIGILANCE AI PORTRAIT 1 (Ground Truth: AI-Generated)\n"
    "• Raw Neural Network Score: 94.18% AI | Calibrated Decision Confidence: 86.7% AI\n"
    "• OOD Telemetry: Score = 0.7925 | Verdict: LIKELY AI-GENERATED\n"
    "• Status: PASSED (Eliminated false-real prediction on photorealistic modern diffusion).\n\n"
    "SPECIMEN 3: ONGC VIGILANCE AI PORTRAIT 2 (Ground Truth: AI-Generated)\n"
    "• Raw Neural Network Score: 91.82% AI | Calibrated Decision Confidence: 84.6% AI\n"
    "• OOD Telemetry: Score = 0.7857 | Verdict: LIKELY AI-GENERATED\n"
    "• Status: PASSED (100% pass rate across all external regression collections)."
)
for p in tf11.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GOLD if p.text.startswith("SPECIMEN") else GRAY
    if p.text.startswith("SPECIMEN"):
        p.font.bold = True

# ==============================================================================
# SLIDE 12: Limitations, Future Scope & Conclusion
# ==============================================================================
s12 = prs.slides.add_slide(blank_layout)
apply_background(s12)
add_header(s12, "Limitations, Future Scope & Academic Conclusion", "11 // CONCLUSION")

add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Limitations & Real-World Boundaries")
tb12_1 = s12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.3))
tf12_1 = tb12_1.text_frame
tf12_1.word_wrap = True
tf12_1.text = (
    "• Resolution Constraints: CIFAKE benchmark training is centered on 32×32 resolution. Modern 4K diffusion media requires multi-scale patch sampling.\n\n"
    "• Compression Artifacts: Heavy social media recompression (JPEG quality < 50) attenuates subtle high-frequency spectral fingerprints.\n\n"
    "• Novel Generative Paradigms: Continuous domain shifts from novel architectures (FLUX, Sora) necessitate continual learning."
)
for p in tf12_1.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Future Scope & Final Remarks")
tb12_2 = s12.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
tf12_2 = tb12_2.text_frame
tf12_2.word_wrap = True
tf12_2.text = (
    "• Future Directions:\n"
    "  - Multi-scale CIFAKE-1024 benchmark creation.\n"
    "  - Cross-attention frequency-conditioned vision transformers.\n"
    "  - Edge-device quantization (INT8 TFLite/ONNX).\n\n"
    "• Academic Conclusion:\n"
    "CIFAKE delivers a verified, CPU-optimized, reproducible AI detection platform achieving 94.92% benchmark accuracy at 1.58 ms latency, backed by rigorous OOD gating.\n\n"
    "THANK YOU // QUESTIONS & DISCUSSION"
)
for p in tf12_2.paragraphs:
    p.font.size = Pt(11)
    p.font.color.rgb = GOLD if "THANK YOU" in p.text else GRAY
    if "THANK YOU" in p.text:
        p.font.bold = True

output_pptx = PRES_DIR / "CIFAKE_FINAL_PRESENTATION.pptx"
prs.save(str(output_pptx))
print(f"Successfully generated 12-slide presentation: {output_pptx}")
