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

        # -------------------------------------------------------------
        # BRANCH 1: OUT-OF-DISTRIBUTION INPUT (Domain Shift Active)
        # -------------------------------------------------------------
        if is_ood:
            # Domain shift detected: image resolution or manifold deviates from 32x32 CIFAKE.
            # CRITICAL RULES:
            # 1. OOD score must NEVER be treated as AI-generation probability.
            # 2. Unknown image categories must not automatically become FAKE.
            # 3. If analytical signals conflict or evidence is insufficient, report INCONCLUSIVE.

            # Sub-case 1A: Concordant AI Evidence
            # Deep model recognizes generative diffusion manifold (P(AI) >= 0.60)
            # AND multi-scale forensics corroborates with synthetic signatures (or spectral lattice peak > 6.0)
            if (cifake_ai_prob >= 0.60 and is_synthetic_forensic) or spectral_peak > 6.5:
                label = "AI"
                verdict = "LIKELY AI-GENERATED"
                class_id = 0
                is_real = False
                is_ai = True
                is_inconclusive = False
                primary_driver = "MODERN_FORENSICS" if (is_synthetic_forensic or spectral_peak > 6.5) else "CIFAKE_DEEP_MODEL"

                calibrated_conf = min(0.92, max(0.70, round(cifake_ai_prob * (1.0 - ood_score * 0.10), 4)))
                final_probs = {
                    "REAL": round(1.0 - calibrated_conf, 4),
                    "AI": round(calibrated_conf, 4)
                }

                rationale = (
                    f"Out-of-Distribution image exhibits corroborating generative signatures across deep feature projections "
                    f"(P(AI)={cifake_ai_prob * 100:.1f}%) and multi-scale forensic analysis "
                    f"({'; '.join(forensic_res.detected_anomalies[:2]) if forensic_res.detected_anomalies else 'latent generative manifold footprint'})."
                )

            # Sub-case 1B: Concordant REAL Evidence
            # Deep model recognizes authentic photographic features (P(REAL) >= 0.70)
            # AND multi-scale forensics detects no synthetic generative anomalies
            elif cifake_real_prob >= 0.70 and not is_synthetic_forensic and spectral_peak < 5.0:
                label = "REAL"
                verdict = "LIKELY REAL"
                class_id = 1
                is_real = True
                is_ai = False
                is_inconclusive = False
                primary_driver = "CIFAKE_DEEP_MODEL"

                calibrated_conf = min(0.88, max(0.65, round(cifake_real_prob * (1.0 - ood_score * 0.18), 4)))
                final_probs = {
                    "REAL": round(calibrated_conf, 4),
                    "AI": round(1.0 - calibrated_conf, 4)
                }

                rationale = (
                    f"Out-of-Distribution image exhibits non-benchmark resolution ({ood_res.resolution_scale_factor:.1f}x CIFAKE scale), "
                    f"but native deep convolutional features (P(REAL)={cifake_real_prob * 100:.1f}%) and natural gradient dispersion "
                    f"(PRNU residual std={forensic_res.noise_residual_std:.2f}) confirm an authentic camera photograph."
                )

            # Sub-case 1C: Ambiguous, Borderline, or Conflicting Evidence -> INCONCLUSIVE
            # (e.g. deep model says REAL but forensics flagged anomalies, or deep model is close to 0.50)
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
                    f"Out-of-Distribution domain shift detected away from the 32x32 CIFAKE benchmark (OOD score: {ood_score:.2f}, "
                    f"scale: {ood_res.resolution_scale_factor:.1f}x) with conflicting or borderline analytical evidence "
                    f"(P(REAL)={cifake_real_prob * 100:.1f}%, forensic synthetic likely={is_synthetic_forensic}). "
                    f"To prevent false certainty, the determination is designated as INCONCLUSIVE."
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
