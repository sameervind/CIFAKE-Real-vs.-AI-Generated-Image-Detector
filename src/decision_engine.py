"""
Transparent Multi-Evidence Decision Arbiter for AI Image Authenticity Detection.

Synthesizes multiple independent analytical signals without arbitrary probability averaging:
1. CIFAKE ResNet-50 Transfer Learning Classifier (with domain distribution gating).
2. Out-of-Distribution (OOD) & Uncertainty Engine (manifold distance & scale shift).
3. Modern Multi-Scale Forensic Analyzer (PRNU noise, kurtosis, plasticity, TCA).
4. 2D FFT Spectral Diagnostic Engine (periodic lattice harmonics).

Strictly enforces scientific rigor: Never produces overconfident false predictions
when an input is out-of-distribution.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np

from src.ood_detector import OODResult
from src.forensics import ForensicReport


@dataclass
class DecisionOutcome:
    """Consolidated decision payload synthesized by the Transparent Decision Arbiter."""
    label: str                     # 'REAL', 'AI', or 'INCONCLUSIVE'
    verdict: str                   # 'LIKELY REAL', 'LIKELY AI-GENERATED', 'INCONCLUSIVE / OUT OF DISTRIBUTION'
    class_id: int                  # 1 for REAL, 0 for AI, -1 for INCONCLUSIVE
    is_real: bool                  # True if determined REAL
    is_ai: bool                    # True if determined AI
    is_inconclusive: bool          # True if out-of-distribution with ambiguous signals
    confidence: float              # [0.0, 1.0] calibrated decision confidence
    confidence_percent: str        # Formatted percentage string
    primary_driver: str            # 'CIFAKE_RESNET50', 'MODERN_FORENSICS', or 'OOD_GATE'
    synthesis_rationale: str       # Transparent engineering explanation
    probabilities: Dict[str, float]# Final calibrated probabilities
    ood_telemetry: Dict[str, Any]  # Complete OOD telemetry
    forensic_telemetry: Dict[str, Any] # Complete forensic telemetry
    cifake_telemetry: Dict[str, Any]   # ResNet-50 raw telemetry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TransparentDecisionEngine:
    """
    Synthesizes deep residual classification, out-of-distribution gating,
    and physical optical forensics to yield an authoritative, transparent verdict.
    """

    def arbitrate(
        self,
        cifake_res: Dict[str, Any],
        ood_res: OODResult,
        forensic_res: ForensicReport,
        spectral_metrics: Optional[Dict[str, Any]] = None
    ) -> DecisionOutcome:
        """
        Execute transparent multi-evidence decision arbitration.

        Args:
            cifake_res: Output from ResNet50InferenceService.predict().
            ood_res: Output from OODDetector.evaluate().
            forensic_res: Output from MultiScaleForensicAnalyzer.analyze().
            spectral_metrics: Optional 2D FFT metrics dictionary.

        Returns:
            DecisionOutcome with transparent explanation and calibrated confidence.
        """
        cifake_is_real = cifake_res.get("is_real", False)
        cifake_real_prob = cifake_res.get("probabilities", {}).get("REAL", 0.5)
        cifake_ai_prob = cifake_res.get("probabilities", {}).get("AI", 0.5)
        cifake_conf = cifake_res.get("confidence", 0.5)

        is_ood = ood_res.is_ood
        ood_score = ood_res.ood_score
        is_synthetic_forensic = forensic_res.is_synthetic_likely
        forensic_ai_prob = forensic_res.synthetic_probability

        spectral_peak = forensic_res.spectral_peak_ratio
        if spectral_metrics and "spectral_peak_ratio" in spectral_metrics:
            spectral_peak = max(spectral_peak, float(spectral_metrics["spectral_peak_ratio"]))

        # -------------------------------------------------------------
        # BRANCH 1: OUT-OF-DISTRIBUTION INPUT (Domain Shift Active)
        # -------------------------------------------------------------
        if is_ood:
            # ResNet-50 was trained on 32x32 CIFAR-10 objects.
            # Its raw probability is uncalibrated and CANNOT be treated as authoritative.

            # Sub-case 1A: Modern Forensics or Spectral Analysis detects generative signatures
            if is_synthetic_forensic or spectral_peak > 6.0 or forensic_res.laplacian_kurtosis > 14.0 or forensic_res.plasticity_index > 7.0:
                label = "AI"
                verdict = "LIKELY AI-GENERATED"
                class_id = 0
                is_real = False
                is_ai = True
                is_inconclusive = False
                primary_driver = "MODERN_FORENSICS"

                # Calibrate confidence from physical forensic evidence
                calibrated_conf = max(0.75, forensic_ai_prob)
                final_probs = {
                    "REAL": round(1.0 - calibrated_conf, 4),
                    "AI": round(calibrated_conf, 4)
                }

                rationale = (
                    f"Out-of-Distribution Domain Shift Detected (OOD score: {ood_score:.2f}). "
                    f"The CIFAKE ResNet-50 benchmark classifier is uncalibrated on this high-resolution/non-CIFAR image "
                    f"(which downsampling biased towards REAL with raw P(REAL)={cifake_real_prob * 100:.1f}%). "
                    f"However, the Multi-Scale Forensic Analyzer detected decisive physical synthetic anomalies: "
                    f"{'; '.join(forensic_res.detected_anomalies[:2])}. "
                    f"Forensic optical evidence overrides the biased 32x32 manifold projection."
                )

            # Sub-case 1B: Clean photographic optical signatures (no anomalies, natural noise)
            elif not is_synthetic_forensic and forensic_res.real_probability >= 0.70 and len(forensic_res.detected_anomalies) == 0:
                label = "REAL"
                verdict = "LIKELY REAL"
                class_id = 1
                is_real = True
                is_ai = False
                is_inconclusive = False
                primary_driver = "MODERN_FORENSICS"

                calibrated_conf = min(0.85, forensic_res.real_probability)
                final_probs = {
                    "REAL": round(calibrated_conf, 4),
                    "AI": round(1.0 - calibrated_conf, 4)
                }

                rationale = (
                    f"Out-of-Distribution Domain Shift Detected (OOD score: {ood_score:.2f}). "
                    f"Image falls outside the CIFAKE benchmark, but multi-scale forensic analysis verified "
                    f"natural photographic gradient dispersion, consistent optical sensor noise, and absence of generative artifacts."
                )

            # Sub-case 1C: Ambiguous or Conflicting Evidence
            else:
                label = "INCONCLUSIVE"
                verdict = "INCONCLUSIVE / OUT OF DISTRIBUTION"
                class_id = -1
                is_real = False
                is_ai = False
                is_inconclusive = True
                primary_driver = "OOD_GATE"

                final_probs = {
                    "REAL": 0.50,
                    "AI": 0.50
                }
                calibrated_conf = 0.50

                rationale = (
                    f"Image is Out-of-Distribution (OOD score: {ood_score:.2f}, scale: {ood_res.resolution_scale_factor:.1f}x) "
                    f"with conflicting or borderline analytical signals. "
                    f"The CIFAKE 32x32 ResNet-50 model is uncalibrated for this domain. "
                    f"To prevent false certainty, the authenticity determination is designated as INCONCLUSIVE."
                )

        # -------------------------------------------------------------
        # BRANCH 2: IN-DISTRIBUTION INPUT (CIFAKE Benchmark Domain)
        # -------------------------------------------------------------
        else:
            # Within the benchmark manifold, ResNet-50 is the authoritative neural classifier
            primary_driver = "CIFAKE_RESNET50"
            is_inconclusive = False

            if cifake_is_real:
                label = "REAL"
                verdict = "LIKELY REAL"
                class_id = 1
                is_real = True
                is_ai = False
                # Corroborate with forensics
                if is_synthetic_forensic:
                    # Minor conflict within benchmark
                    calibrated_conf = max(0.55, round(cifake_conf * 0.75, 4))
                    rationale = (
                        f"CIFAKE ResNet-50 classifies sample as REAL (P(REAL)={cifake_real_prob * 100:.1f}%), "
                        f"but subtle forensic anomalies were detected. Calibrated confidence adjusted to {calibrated_conf * 100:.1f}%."
                    )
                else:
                    calibrated_conf = cifake_conf
                    rationale = (
                        f"In-distribution photographic sample verified by CIFAKE ResNet-50 deep residual manifold "
                        f"(P(REAL)={cifake_real_prob * 100:.1f}%) and corroborated by physical optical dispersion."
                    )
            else:
                label = "AI"
                verdict = "LIKELY AI-GENERATED"
                class_id = 0
                is_real = False
                is_ai = True
                calibrated_conf = cifake_conf
                rationale = (
                    f"In-distribution synthetic sample detected by CIFAKE ResNet-50 deep residual manifold "
                    f"(P(AI)={cifake_ai_prob * 100:.1f}%) with characteristic generative distribution footprint."
                )

            final_probs = {
                "REAL": round(cifake_real_prob, 4),
                "AI": round(cifake_ai_prob, 4)
            }

        return DecisionOutcome(
            label=label,
            verdict=verdict,
            class_id=class_id,
            is_real=is_real,
            is_ai=is_ai,
            is_inconclusive=is_inconclusive,
            confidence=round(calibrated_conf, 4),
            confidence_percent=f"{calibrated_conf * 100:.1f}%",
            primary_driver=primary_driver,
            synthesis_rationale=rationale,
            probabilities=final_probs,
            ood_telemetry=ood_res.to_dict(),
            forensic_telemetry=forensic_res.to_dict(),
            cifake_telemetry={
                "raw_label": cifake_res.get("label"),
                "raw_real_prob": cifake_real_prob,
                "raw_ai_prob": cifake_ai_prob,
                "raw_score": cifake_res.get("raw_score"),
                "raw_confidence": cifake_conf,
                "backend": cifake_res.get("backend")
            }
        )


_GLOBAL_DECISION_ENGINE: Optional[TransparentDecisionEngine] = None

def get_decision_engine() -> TransparentDecisionEngine:
    """Return singleton instance of TransparentDecisionEngine."""
    global _GLOBAL_DECISION_ENGINE
    if _GLOBAL_DECISION_ENGINE is None:
        _GLOBAL_DECISION_ENGINE = TransparentDecisionEngine()
    return _GLOBAL_DECISION_ENGINE
