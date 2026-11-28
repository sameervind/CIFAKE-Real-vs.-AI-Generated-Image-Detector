"""
Transparent Multi-Evidence Decision Arbiter for AI Image Authenticity Detection.

Synthesizes independent analytical signals without arbitrary probability averaging:
1. CIFAKE Native CIFAR-CNN Classifier (authoritative deep learning model).
2. Out-of-Distribution (OOD) & Uncertainty Engine (scale shift & manifold distance).
3. Supplementary Multi-Scale Optical Diagnostics (PRNU noise, kurtosis, plasticity, TCA).
4. Supplementary 2D FFT Spectral Diagnostics (periodic lattice harmonics).

Strictly enforces scientific rigor:
- Never uses heuristic anomaly scores or spectral peaks alone to force AI predictions.
- Never converts OOD status into AI probability.
- Returns INCONCLUSIVE when classifiers cannot reliably support a verdict on OOD inputs.
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
    primary_driver: str            # 'CIFAR_CNN' or 'OOD_GATE'
    synthesis_rationale: str       # Transparent engineering explanation
    probabilities: Dict[str, float]# Decision probabilities
    raw_probabilities: Dict[str, float] # Unmodified raw neural network probabilities
    ood_telemetry: Dict[str, Any]  # Complete OOD telemetry
    forensic_telemetry: Dict[str, Any] # Complete forensic diagnostic telemetry
    cifake_telemetry: Dict[str, Any]   # Model raw telemetry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TransparentDecisionEngine:
    """
    Synthesizes deep model classification with out-of-distribution gating
    and supplementary optical diagnostics to yield an authoritative, transparent verdict.
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
            cifake_res: Output from CIFAKEInferenceService.predict().
            ood_res: Output from OODDetector.evaluate().
            forensic_res: Output from MultiScaleForensicAnalyzer.analyze().
            spectral_metrics: Optional 2D FFT metrics dictionary.

        Returns:
            DecisionOutcome with transparent explanation, raw probabilities, and confidence.
        """
        cifake_is_real = cifake_res.get("is_real", False)
        cifake_real_prob = cifake_res.get("probabilities", {}).get("REAL", 0.5)
        cifake_ai_prob = cifake_res.get("probabilities", {}).get("AI", 0.5)
        cifake_conf = cifake_res.get("confidence", 0.5)
        model_name = cifake_res.get("model_name", "cifar_cnn").upper()

        is_ood = ood_res.is_ood
        ood_score = ood_res.ood_score

        raw_probs = {
            "REAL": round(cifake_real_prob, 4),
            "AI": round(cifake_ai_prob, 4)
        }

        # -------------------------------------------------------------
        # BRANCH 1: OUT-OF-DISTRIBUTION INPUT (Domain Shift Active)
        # -------------------------------------------------------------
        if is_ood:
            # Domain shift detected: image resolution or manifold deviates from 32x32 CIFAKE.
            # CRITICAL RULES:
            # 1. OOD score must NEVER be treated as AI-generation probability.
            # 2. Unknown image categories must not automatically become FAKE.
            # 3. Spectral peak alone must never establish AI generation.
            # 4. If the deep model cannot decisively support a verdict on this domain, return INCONCLUSIVE.

            # Sub-case 1A: Deep model decisively detects synthetic diffusion manifold features (>= 85%)
            if cifake_ai_prob >= 0.85:
                label = "AI"
                verdict = "LIKELY AI-GENERATED"
                class_id = 0
                is_real = False
                is_ai = True
                is_inconclusive = False
                primary_driver = model_name

                # Discount confidence due to domain shift scale
                calibrated_conf = min(0.92, max(0.70, round(cifake_ai_prob * (1.0 - ood_score * 0.10), 4)))
                final_probs = {
                    "REAL": round(1.0 - calibrated_conf, 4),
                    "AI": round(calibrated_conf, 4)
                }
                rationale = (
                    f"Out-of-Distribution domain shift detected away from 32x32 CIFAKE benchmark (scale: {ood_res.resolution_scale_factor:.1f}x). "
                    f"However, the trained CIFAKE {model_name} feature extractor decisively identified generative diffusion signatures "
                    f"(raw P(AI)={cifake_ai_prob * 100:.1f}%)."
                )

            # Sub-case 1B: Deep model decisively detects authentic photographic features (>= 85%)
            elif cifake_real_prob >= 0.85:
                label = "REAL"
                verdict = "LIKELY REAL"
                class_id = 1
                is_real = True
                is_ai = False
                is_inconclusive = False
                primary_driver = model_name

                calibrated_conf = min(0.92, max(0.70, round(cifake_real_prob * (1.0 - ood_score * 0.10), 4)))
                final_probs = {
                    "REAL": round(calibrated_conf, 4),
                    "AI": round(1.0 - calibrated_conf, 4)
                }
                rationale = (
                    f"Out-of-Distribution domain shift detected away from 32x32 CIFAKE benchmark (scale: {ood_res.resolution_scale_factor:.1f}x). "
                    f"However, the trained CIFAKE {model_name} feature extractor decisively identified authentic photographic features "
                    f"(raw P(REAL)={cifake_real_prob * 100:.1f}%)."
                )

            # Sub-case 1C: Borderline, Non-Decisive, or Ambiguous Evidence -> INCONCLUSIVE
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
                    f"scale: {ood_res.resolution_scale_factor:.1f}x) with non-decisive deep model probabilities "
                    f"(P(REAL)={cifake_real_prob * 100:.1f}%, P(AI)={cifake_ai_prob * 100:.1f}%). "
                    f"To prevent false certainty on unsupported image distributions, the determination is designated as INCONCLUSIVE."
                )

        # -------------------------------------------------------------
        # BRANCH 2: IN-DISTRIBUTION INPUT (CIFAKE Benchmark Domain)
        # -------------------------------------------------------------
        else:
            primary_driver = model_name
            is_inconclusive = False

            if cifake_is_real:
                label = "REAL"
                verdict = "LIKELY REAL"
                class_id = 1
                is_real = True
                is_ai = False
                calibrated_conf = cifake_conf
                rationale = (
                    f"In-distribution photographic sample verified by trained CIFAKE {model_name} model "
                    f"(raw P(REAL)={cifake_real_prob * 100:.1f}%)."
                )
            else:
                label = "AI"
                verdict = "LIKELY AI-GENERATED"
                class_id = 0
                is_real = False
                is_ai = True
                calibrated_conf = cifake_conf
                rationale = (
                    f"In-distribution synthetic sample detected by trained CIFAKE {model_name} model "
                    f"(raw P(AI)={cifake_ai_prob * 100:.1f}%)."
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
            raw_probabilities=raw_probs,
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
