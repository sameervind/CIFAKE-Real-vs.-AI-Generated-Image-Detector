import os
from pathlib import Path
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent.parent
PRES_DIR = ROOT / "03_PRESENTATION"
OUTPUT_PDF = PRES_DIR / "CIFAKE_FINAL_PRESENTATION.pdf"
ASSETS_DIR = ROOT / "04_RESULTS"

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)  # 11 x 8.5 inches (792 x 612 pt)

doc = SimpleDocTemplate(
    str(OUTPUT_PDF),
    pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom styles matching dark Lamborghini aesthetic
c_bg = colors.HexColor("#0F0F11")
c_card = colors.HexColor("#18181C")
c_gold = colors.HexColor("#FFC000")
c_white = colors.HexColor("#FFFFFF")
c_gray = colors.HexColor("#B4B4BE")
c_muted = colors.HexColor("#787882")
c_green = colors.HexColor("#34C759")
c_blue = colors.HexColor("#0A84FF")
c_red = colors.HexColor("#FF3B30")

title_style = ParagraphStyle(
    "SlideTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=22,
    leading=26,
    textColor=c_white
)

category_style = ParagraphStyle(
    "SlideCategory",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9,
    leading=12,
    textColor=c_gold
)

body_style = ParagraphStyle(
    "SlideBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    leading=16,
    textColor=c_gray
)

bold_gold_style = ParagraphStyle(
    "SlideBoldGold",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=c_gold
)

def make_slide_card(title, content_flowables, width=720, height=None, border_color=c_gold):
    card_elements = []
    if title:
        card_elements.append(Paragraph(title.upper(), bold_gold_style))
        card_elements.append(Spacer(1, 8))
    card_elements.extend(content_flowables)
    
    t = Table([[card_elements]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
    ]))
    return t

def draw_background(canvas, document):
    canvas.saveState()
    canvas.setFillColor(c_bg)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=True, stroke=False)
    # Bottom brand accent
    canvas.setFillColor(c_gold)
    canvas.rect(36, 18, 120, 2, fill=True, stroke=False)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.setFillColor(c_muted)
    canvas.drawString(165, 16, "CIFAKE // REAL VS. AI IMAGE DETECTION // ACADEMIC ENGINEERING DEFENSE")
    canvas.drawRightString(PAGE_WIDTH - 36, 16, f"Slide {canvas.getPageNumber()} / 12")
    canvas.restoreState()

slides = []

# ==================== SLIDE 1 ====================
s1 = [
    Spacer(1, 40),
    Paragraph("CORSE CLIENTE // MACHINE LEARNING RESEARCH", category_style),
    Spacer(1, 8),
    Paragraph("CIFAKE: High-Performance Real vs. AI Image Detection", title_style),
    Spacer(1, 6),
    Paragraph("Native 32×32 CIFAR-CNN Architecture, Out-of-Distribution Gating & Frequency-Domain Diagnostics", ParagraphStyle("Sub", parent=body_style, fontSize=13, textColor=c_gray)),
    Spacer(1, 30),
    make_slide_card(
        "Production System Specifications",
        [
            Paragraph("• <b>Primary Architecture:</b> Native CIFAR-CNN (305,953 Parameters, 1.17 MB)", body_style),
            Paragraph("• <b>Verified Benchmark Performance:</b> 94.92% Accuracy | 0.9492 Macro F1 | 0.9903 ROC-AUC", body_style),
            Paragraph("• <b>CPU Inference Latency:</b> 1.58 ms / image on standard CPU (oneDNN Optimized)", body_style),
            Paragraph("• <b>Decision Engine:</b> Out-of-Distribution Manifold Arbiter + 2D FFT Radial Spectral Analysis", body_style),
        ],
        width=720,
        border_color=c_gold
    ),
    Spacer(1, 20),
    Paragraph("Master of Science in Machine Learning // Final Academic Engineering Defense", ParagraphStyle("F", parent=body_style, fontSize=10, textColor=c_muted))
]
slides.append(s1)

# ==================== SLIDE 2 ====================
s2 = [
    Paragraph("01 // CONTEXT & MOTIVATION", category_style),
    Spacer(1, 4),
    Paragraph("Problem Statement & Scientific Motivation", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("The Generative Media Crisis", [
            Paragraph("• <b>Proliferation of Latent Diffusion:</b> Stable Diffusion, Midjourney, and DALL-E produce photorealistic synthetic imagery indistinguishable by human perception.", body_style),
            Paragraph("• <b>Threat to Information Integrity:</b> Synthetic media weaponized for biometric spoofing, digital disinformation, and judicial evidence contamination requires automated, verifiable forensic authentication.", body_style),
            Paragraph("• <b>The Domain Gap:</b> Models trained purely on low-resolution benchmarks often fail catastrophically when presented with high-resolution real-world portraits.", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35")),
        make_slide_card("Core Engineering Objectives", [
            Paragraph("• <b>1. High Benchmark Accuracy:</b> Achieve >94% reproducible accuracy on the official 120,000 CIFAKE benchmark.", body_style),
            Paragraph("• <b>2. Extreme CPU Efficiency:</b> Deliver sub-2 ms latency without requiring GPU accelerators.", body_style),
            Paragraph("• <b>3. Domain Generalization:</b> Prevent false-real predictions on modern AI diffusion and false-fake errors on genuine portraits.", body_style),
            Paragraph("• <b>4. Transparent Forensics:</b> Expose 2D FFT radial power fall-off and OOD telemetry as supplementary human-verifiable diagnostics.", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35"))
    ]], colWidths=[360, 360])
]
slides.append(s2)

# ==================== SLIDE 3 ====================
s3 = [
    Paragraph("02 // DIAGNOSTIC AUDIT", category_style),
    Spacer(1, 4),
    Paragraph("Root-Cause Failure Audit: Why Legacy Systems Failed", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("Receptive Field Mismatch", [
            Paragraph("• <b>Destructive Upsampling:</b> Scaling 32×32 inputs to 224×224 created 49× spatial inflation and blurred microscopic diffusion signatures.", body_style),
            Paragraph("• <b>Aggressive Downsampling:</b> Standard ResNet-50's 7×7 conv (stride 2) collapsed 32×32 inputs to 8×8 before stage 1.", body_style)
        ], width=230, border_color=c_red),
        make_slide_card("Uncalibrated Heuristics", [
            Paragraph("• <b>Arbitrary Formula Overrides:</b> Heuristic FFT peaks and kurtosis equations overrode deep neural predictions.", body_style),
            Paragraph("• <b>False-Positive AI Verdicts:</b> Sharp authentic photographs triggered spectral overrides, causing genuine portraits to be labeled AI with 98% confidence.", body_style)
        ], width=230, border_color=c_red),
        make_slide_card("OOD Conflation", [
            Paragraph("• <b>Scale Conflated with Fake:</b> High resolution was mistakenly treated as proof of synthetic generation.", body_style),
            Paragraph("• <b>Epistemic Uncertainty:</b> OOD measures domain distance and must discount confidence, not force classification.", body_style)
        ], width=230, border_color=c_red)
    ]], colWidths=[240, 240, 240])
]
slides.append(s3)

# ==================== SLIDE 4 ====================
s4 = [
    Paragraph("03 // SYSTEM ARCHITECTURE", category_style),
    Spacer(1, 4),
    Paragraph("Proposed Three-Tiered Decision Architecture", title_style),
    Spacer(1, 16),
    make_slide_card("Integrated Decision Pipeline", [
        Paragraph("<b>TIER 1 — NATIVE CONVOLUTIONAL FEATURE EXTRACTION (CIFAR-CNN)</b>", bold_gold_style),
        Paragraph("Processes native 32×32 inputs using 3×3 convolutions with stride 1. Extracts 128-dim dense manifold representations and outputs calibrated P(REAL) in [0.0, 1.0]. Eliminates destructive upsampling.", body_style),
        Spacer(1, 6),
        Paragraph("<b>TIER 2 — OUT-OF-DISTRIBUTION (OOD) DOMAIN SHIFT ENGINE</b>", bold_gold_style),
        Paragraph("Monitors spatial resolution and manifold distance. High OOD scores discount decision confidence without flipping class labels. Demands decisive neural confidence (≥85%) or outputs INCONCLUSIVE.", body_style),
        Spacer(1, 6),
        Paragraph("<b>TIER 3 — MULTI-SCALE PHYSICAL FORENSICS & FREQUENCY ANALYSIS</b>", bold_gold_style),
        Paragraph("Computes 2D Discrete Fourier Transforms (2D FFT), sensor PRNU noise variance, and Laplacian kurtosis. Strictly non-authoritative: serves as supplementary explainability telemetry.", body_style),
    ], width=720, border_color=c_gold)
]
slides.append(s4)

# ==================== SLIDE 5 ====================
s5 = [
    Paragraph("04 // DATA PIPELINE", category_style),
    Spacer(1, 4),
    Paragraph("CIFAKE Dataset & Anti-Leakage Preprocessing", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("Dataset Composition (120,000 Images)", [
            Paragraph("• <b>Class 0 (AI / Synthetic):</b> 60,000 Stable Diffusion v1.4 images.", body_style),
            Paragraph("• <b>Class 1 (Real Photography):</b> 60,000 CIFAR-10 photographs.", body_style),
            Paragraph("• <b>Resolution:</b> 32 × 32 pixels, 3 color channels (RGB).", body_style),
            Paragraph("• <b>10 Semantic Categories:</b> Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.", body_style),
            Paragraph("• <b>Strict Split:</b> 100,000 Train / 20,000 Test (strictly isolated and unaugmented).", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35")),
        make_slide_card("Strict Preprocessing Controls", [
            Paragraph("• <b>Internalized Model Rescaling:</b> <i>layers.Rescaling(1/255.0)</i> is embedded in the Keras graph, eliminating training/inference mismatch.", body_style),
            Paragraph("• <b>Physics-Preserving Augmentations:</b> Restricted to horizontal flips and 4px reflection padding. Vertical flips removed to preserve lighting physics.", body_style),
            Paragraph("• <b>Standardized Contract:</b> Label 0 = AI, Label 1 = REAL. Model sigmoid output represents P(REAL).", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35"))
    ]], colWidths=[360, 360])
]
slides.append(s5)

# ==================== SLIDE 6 ====================
s6 = [
    Paragraph("05 // MODEL IMPLEMENTATION", category_style),
    Spacer(1, 4),
    Paragraph("CIFAR-CNN: Native Deep Architecture", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("Layer Architecture (305,953 Parameters)", [
            Paragraph("• <b>Input:</b> (None, 32, 32, 3) -> Rescaling(1/255.0)", body_style),
            Paragraph("• <b>Stage 1:</b> 2x Conv2D(32, 3x3, same) + BN + ReLU -> MaxPool(2x2) -> Dropout(0.2)", body_style),
            Paragraph("• <b>Stage 2:</b> 2x Conv2D(64, 3x3, same) + BN + ReLU -> MaxPool(2x2) -> Dropout(0.3)", body_style),
            Paragraph("• <b>Stage 3:</b> 2x Conv2D(128, 3x3, same) + BN + ReLU -> MaxPool(2x2) -> Dropout(0.4)", body_style),
            Paragraph("• <b>Head:</b> GlobalAveragePooling2D -> Dense(128) + BN + Dropout(0.4) -> Dense(1, Sigmoid)", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35")),
        make_slide_card("Training Hyperparameters & Latency", [
            Paragraph("• <b>Loss:</b> Binary Cross-Entropy (BCE)", body_style),
            Paragraph("• <b>Optimizer:</b> Adam (lr = 1e-3, ReduceLROnPlateau)", body_style),
            Paragraph("• <b>Regularization:</b> L2 Weight Decay (1e-4) + Spatial Dropout", body_style),
            Paragraph("• <b>CPU Latency:</b> 1.58 ms / image (oneDNN optimized)", body_style),
            Paragraph("• <b>Model File Size:</b> 1.17 MB (305,953 parameters)", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35"))
    ]], colWidths=[360, 360])
]
slides.append(s6)

# ==================== SLIDE 7 ====================
s7 = [
    Paragraph("06 // TECHNOLOGY STACK", category_style),
    Spacer(1, 4),
    Paragraph("Technology Stack & Engineering Architecture", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("Deep Learning Core", [
            Paragraph("• TensorFlow 2.x & Keras 3.x", body_style),
            Paragraph("• NumPy & SciPy (2D FFT / MTF)", body_style),
            Paragraph("• OpenCV & PIL (Color & PRNU)", body_style)
        ], width=230, border_color=c_blue),
        make_slide_card("API Backend", [
            Paragraph("• FastAPI Asynchronous Server", body_style),
            Paragraph("• Uvicorn ASGI Worker", body_style),
            Paragraph("• Pydantic V2 Response Schemas", body_style)
        ], width=230, border_color=c_green),
        make_slide_card("Cockpit UI & QA", [
            Paragraph("• Streamlit Lamborghini UI", body_style),
            Paragraph("• Aceternity Bento Grid Layout", body_style),
            Paragraph("• Pytest (23/23 tests passing)", body_style)
        ], width=230, border_color=c_gold)
    ]], colWidths=[240, 240, 240])
]
slides.append(s7)

# ==================== SLIDE 8 ====================
s8 = [
    Paragraph("07 // BENCHMARK RESULTS", category_style),
    Spacer(1, 4),
    Paragraph("Independent 20,000 CIFAKE Test Set Evaluation", title_style),
    Spacer(1, 10),
    Table([[
        RLImage(str(ASSETS_DIR / "confusion_matrix.png"), width=340, height=280) if (ASSETS_DIR / "confusion_matrix.png").exists() else Paragraph("Confusion Matrix", body_style),
        make_slide_card("Verified Metric Breakdown", [
            Paragraph("• <b>Accuracy:</b> <b>94.92%</b> (18,984 / 20,000 correct)", bold_gold_style),
            Paragraph("• <b>Macro F1-Score:</b> 0.9492 | <b>Macro Precision:</b> 0.9494", body_style),
            Paragraph("• <b>ROC-AUC Score:</b> 0.9903 | <b>PR-AUC:</b> 0.9907", body_style),
            Paragraph("• <b>Binary Cross-Entropy:</b> 0.1298", body_style),
            Spacer(1, 4),
            Paragraph("<b>Confusion Matrix (Threshold = 0.50):</b>", bold_gold_style),
            Paragraph("  - True AI (TN): 9,382 / 10,000 (Recall = 93.82%)", body_style),
            Paragraph("  - False REAL (FP): 618 / 10,000 (FPR = 6.18%)", body_style),
            Paragraph("  - False AI (FN): 398 / 10,000 (FNR = 3.98%)", body_style),
            Paragraph("  - True REAL (TP): 9,602 / 10,000 (Recall = 96.02%)", body_style)
        ], width=360, border_color=c_gold)
    ]], colWidths=[355, 365])
]
slides.append(s8)

# ==================== SLIDE 9 ====================
s9 = [
    Paragraph("08 // STATISTICAL CURVES", category_style),
    Spacer(1, 4),
    Paragraph("Receiver Operating Characteristic & PR Curves", title_style),
    Spacer(1, 10),
    Table([[
        RLImage(str(ASSETS_DIR / "roc_curve.png"), width=350, height=280) if (ASSETS_DIR / "roc_curve.png").exists() else Paragraph("ROC Curve", body_style),
        RLImage(str(ASSETS_DIR / "precision_recall_curve.png"), width=350, height=280) if (ASSETS_DIR / "precision_recall_curve.png").exists() else Paragraph("PR Curve", body_style),
    ]], colWidths=[360, 360])
]
slides.append(s9)

# ==================== SLIDE 10 ====================
s10 = [
    Paragraph("09 // ARCHITECTURAL BENCHMARK", category_style),
    Spacer(1, 4),
    Paragraph("Architectural Benchmarking & Efficiency Comparison", title_style),
    Spacer(1, 10),
    Table([[
        RLImage(str(ASSETS_DIR / "architecture_comparison.png"), width=710, height=210) if (ASSETS_DIR / "architecture_comparison.png").exists() else Paragraph("Arch Comparison", body_style)
    ]], colWidths=[720]),
    Spacer(1, 10),
    make_slide_card("Benchmark Conclusions", [
        Paragraph("• <b>79x Parameter Reduction:</b> CIFAR-CNN requires only 305k parameters compared to 24.1M for ResNet-50.", body_style),
        Paragraph("• <b>26.6x Latency Acceleration:</b> CPU inference dropped from 42.1 ms to 1.58 ms per image.", body_style),
        Paragraph("• <b>+2.64% Higher Accuracy:</b> Native 32×32 convolutions beat upsampled ResNet-50 by avoiding interpolation blur.", body_style)
    ], width=720, border_color=c_gold)
]
slides.append(s10)

# ==================== SLIDE 11 ====================
s11 = [
    Paragraph("10 // REGRESSION AUDIT", category_style),
    Spacer(1, 4),
    Paragraph("Real-World Generalization & Regression Suite", title_style),
    Spacer(1, 16),
    make_slide_card("Mandatory Specimen Verification (100% Pass Rate)", [
        Paragraph("<b>Specimen 1: Verified Genuine Human Portrait (Ground Truth: REAL)</b>", bold_gold_style),
        Paragraph("Raw Score: 99.96% REAL | Decision Confidence: 91.2% REAL | Primary Driver: CIFAR_CNN | OOD: 0.8717 (Non-penalizing). False positive eliminated.", body_style),
        Spacer(1, 6),
        Paragraph("<b>Specimen 2: ONGC Vigilance AI Portrait 1 (Ground Truth: AI)</b>", bold_gold_style),
        Paragraph("Raw Score: 94.18% AI | Decision Confidence: 86.7% AI | Primary Driver: CIFAR_CNN | Verdict: LIKELY AI-GENERATED. Correctly detected.", body_style),
        Spacer(1, 6),
        Paragraph("<b>Specimen 3: ONGC Vigilance AI Portrait 2 (Ground Truth: AI)</b>", bold_gold_style),
        Paragraph("Raw Score: 91.82% AI | Decision Confidence: 84.6% AI | Primary Driver: CIFAR_CNN | Verdict: LIKELY AI-GENERATED. Correctly detected.", body_style)
    ], width=720, border_color=c_green)
]
slides.append(s11)

# ==================== SLIDE 12 ====================
s12 = [
    Paragraph("11 // CONCLUSION", category_style),
    Spacer(1, 4),
    Paragraph("Limitations, Future Scope & Academic Conclusion", title_style),
    Spacer(1, 16),
    Table([[
        make_slide_card("Limitations & Boundary Conditions", [
            Paragraph("• <b>Resolution Discrepancy:</b> 32×32 benchmark training necessitates multi-scale patch sampling for 4K inputs.", body_style),
            Paragraph("• <b>Compression Degradation:</b> Aggressive social media JPEG compression (<50 quality) attenuates high-frequency forensic artifacts.", body_style),
            Paragraph("• <b>Continual Architecture Shift:</b> Novel generative models (e.g. FLUX, Sora) introduce new latent signatures.", body_style)
        ], width=350, border_color=colors.HexColor("#2D2D35")),
        make_slide_card("Future Directions & Conclusion", [
            Paragraph("• <b>Future Scope:</b> Development of CIFAKE-1024 high-resolution benchmark and frequency-conditioned Vision Transformers.", body_style),
            Paragraph("• <b>Conclusion:</b> Successfully transformed an unstable legacy codebase into a verified, reproducible, CPU-optimized platform achieving 94.92% accuracy at 1.58 ms latency.", body_style),
            Spacer(1, 6),
            Paragraph("<b>THANK YOU // QUESTIONS & DISCUSSION</b>", bold_gold_style)
        ], width=350, border_color=c_gold)
    ]], colWidths=[360, 360])
]
slides.append(s12)

# Build flowables with page breaks
flowables = []
for idx, s in enumerate(slides):
    flowables.extend(s)
    if idx < len(slides) - 1:
        from reportlab.platypus import PageBreak
        flowables.append(PageBreak())

doc.build(flowables, onFirstPage=draw_background, onLaterPages=draw_background)
print(f"Successfully generated PDF Presentation: {OUTPUT_PDF}")
