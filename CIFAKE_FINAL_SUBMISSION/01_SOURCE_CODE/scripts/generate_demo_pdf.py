import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "05_DEMO"
OUTPUT_PDF = DEMO_DIR / "DEMO_WALKTHROUGH.pdf"

doc = SimpleDocTemplate(
    str(OUTPUT_PDF),
    pagesize=letter,
    leftMargin=45,
    rightMargin=45,
    topMargin=45,
    bottomMargin=45
)

styles = getSampleStyleSheet()

c_gold = colors.HexColor("#A07800")
c_dark = colors.HexColor("#1A1A22")
c_card = colors.HexColor("#F8F8FA")
c_border = colors.HexColor("#D8D8E0")

title_style = ParagraphStyle(
    "DemoTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=c_dark,
    alignment=1
)

sub_style = ParagraphStyle(
    "DemoSub",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#505058"),
    alignment=1
)

h1_style = ParagraphStyle(
    "DemoH1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=c_dark,
    spaceBefore=10,
    spaceAfter=4
)

body_style = ParagraphStyle(
    "DemoBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=colors.HexColor("#303038")
)

bold_gold = ParagraphStyle(
    "DemoGold",
    parent=body_style,
    fontName="Helvetica-Bold",
    textColor=c_gold
)

elements = [
    Paragraph("CIFAKE: 5-Minute Live Academic Demonstration Walkthrough", title_style),
    Spacer(1, 4),
    Paragraph("Step-by-Step Practical Guide for Academic Engineering Committee Presentation", sub_style),
    Spacer(1, 8),
    HRFlowable(width="100%", thickness=1, color=c_gold, spaceAfter=12),
    Paragraph("<b>Demonstration Overview & URLs:</b>", body_style),
    Paragraph("• <b>Streamlit Cockpit:</b> <code>http://localhost:8501</code> (Launch via <code>run_app.bat [1]</code> or <code>streamlit run streamlit_app.py</code>)", body_style),
    Paragraph("• <b>FastAPI REST Documentation:</b> <code>http://localhost:8000/docs</code> (Launch via <code>run_app.bat [2]</code>)", body_style),
    Paragraph("• <b>Demonstration Images:</b> Staged in <code>05_DEMO/demo_images/</code>", body_style),
    Spacer(1, 10),
]

steps = [
    ("Step 1: Introduction (00:00 - 00:45)", [
        "<b>Goal:</b> Establish problem context and show active CIFAR-CNN engine online.",
        "<b>Action:</b> Open Streamlit app. Point to the top telemetry sub-nav bar showing <code>CIFAR-CNN ENGINE ONLINE</code>.",
        "<b>Talking Point:</b> High-performance AI detection platform powered by native 32x32 CNN running on CPU at 1.58 ms latency."
    ]),
    ("Step 2: Authentic Real Photograph (00:45 - 01:30)", [
        "<b>Goal:</b> Demonstrate zero false-positives on genuine photography.",
        "<b>Action:</b> Drag & drop <code>05_DEMO/demo_images/03_genuine_human_portrait.jpg</code>. Click 'ANALYZE SPECIMEN'.",
        "<b>Output:</b> <b>LIKELY REAL</b> (91.2% Calibrated Confidence). Raw P(REAL) = 99.96%.",
        "<b>Talking Point:</b> OOD telemetry detects domain shift (1080p), but arbiter correctly avoids false synthetic classification."
    ]),
    ("Step 3: AI-Generated Synthetic Image (01:30 - 02:15)", [
        "<b>Goal:</b> Demonstrate reliable detection of modern photorealistic diffusion imagery.",
        "<b>Action:</b> Drag & drop <code>05_DEMO/demo_images/04_ongc_ai_portrait_1.png</code>. Click 'ANALYZE SPECIMEN'.",
        "<b>Output:</b> <b>LIKELY AI-GENERATED</b> (86.7% Confidence). Raw P(AI) = 94.18%.",
        "<b>Talking Point:</b> Eliminates regression failure; successfully identifies latent generative artifacts in modern portraits."
    ]),
    ("Step 4: Multi-Signal Decision Engine & Frequency Analysis (02:15 - 03:15)", [
        "<b>Goal:</b> Explain transparent forensic diagnostics beneath the hood.",
        "<b>Action:</b> Scroll down to the Bento Forensic Grid. Expand '2D FFT Radial Frequency Spectrum'.",
        "<b>Talking Point:</b> Natural photos follow continuous 1/f power law. Diffusion generates subtle high-frequency peaks. Metrics serve as supplementary diagnostics with zero hard overrides."
    ]),
    ("Step 5: Benchmark Evaluation & Architecture Comparison (03:15 - 04:00)", [
        "<b>Goal:</b> Present empirical verification on 20,000 CIFAKE test images.",
        "<b>Action:</b> Switch to 'MODEL BENCHMARKS & TELEMETRY' tab. Show Confusion Matrix and Comparison Chart.",
        "<b>Talking Point:</b> 94.92% accuracy, 0.9492 F1, 0.9903 ROC-AUC. 79x fewer parameters and 26.6x faster than ResNet-50."
    ]),
    ("Step 6: Limitations & Academic Conclusion (04:00 - 05:00)", [
        "<b>Goal:</b> Address real-world boundaries with scientific rigor.",
        "<b>Action:</b> Show regression audit table and OOD uncertainty gating.",
        "<b>Talking Point:</b> Honest boundaries: 32x32 manifold vs. unconstrained 4K images. Transparent OOD warnings. Conclude and open for questions."
    ])
]

for title, points in steps:
    card_flowables = [Paragraph(f"<b>{title}</b>", bold_gold), Spacer(1, 4)]
    for pt in points:
        card_flowables.append(Paragraph(f"• {pt}", body_style))
    t = Table([[card_flowables]], colWidths=[520])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

doc.build(elements)
print(f"Successfully generated Demo Walkthrough PDF: {OUTPUT_PDF}")
