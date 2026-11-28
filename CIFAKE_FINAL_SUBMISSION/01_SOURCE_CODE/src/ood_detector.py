"""
Out-of-Distribution (OOD) and Uncertainty Engine for CIFAKE.

Detects covariate, scale, and semantic domain shifts away from the CIFAKE benchmark
(32x32 CIFAR-10 photographic and 2022 Stable Diffusion v1.4 synthetic distribution).
Prevents overconfident false predictions on out-of-distribution inputs such as
modern high-resolution photorealistic portraits and institutional interiors.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

DEFAULT_CALIBRATION_PATH = Path("assets/ood_calibration.json")


@dataclass
class OODResult:
    """Structured telemetry output for Out-of-Distribution evaluation."""
    is_ood: bool
    ood_score: float  # [0.0, 1.0], 1.0 = completely out-of-distribution
    verdict: str  # 'IN_DISTRIBUTION', 'BORDERLINE', 'OUT_OF_DISTRIBUTION'
    confidence_discount: float  # [0.10, 1.00] discount factor for raw CIFAKE classifier
    min_class_distance: float
    distance_to_real_centroid: float
    distance_to_ai_centroid: float
    distance_to_overall_centroid: float
    cosine_sim_real: float
    cosine_sim_ai: float
    energy_score: float
    predictive_entropy: float
    resolution_scale_factor: float
    native_resolution: Tuple[int, int]
    domain_shift_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["native_resolution"] = list(self.native_resolution)
        return d


class OODDetector:
    """
    Evaluates whether an input image falls within the CIFAKE benchmark distribution
    or represents an Out-of-Distribution (OOD) domain shift.
    """

    def __init__(self, calibration_path: Optional[Union[str, Path]] = None):
        calib_file = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
        self.calibration = self._load_calibration(calib_file)

        # In-distribution reference vectors and thresholds
        self.centroid_real = np.array(self.calibration.get("centroid_real", []), dtype=np.float32)
        self.centroid_ai = np.array(self.calibration.get("centroid_ai", []), dtype=np.float32)
        self.centroid_overall = np.array(self.calibration.get("centroid_overall", []), dtype=np.float32)
        self.feature_std = np.array(self.calibration.get("feature_std", []), dtype=np.float32)

        self.id_dist_max = float(self.calibration.get("id_distance_max", 16.0))
        self.id_dist_p95 = float(self.calibration.get("id_distance_p95", 14.8))
        self.id_class_dist_max = float(self.calibration.get("id_class_distance_max", 9.5))
        self.id_norm_max = float(self.calibration.get("id_norm_max", 32.0))
        self.temperature = float(self.calibration.get("temperature", 1.0))
        self.benchmark_resolution = tuple(self.calibration.get("benchmark_resolution", [32, 32]))

    def _load_calibration(self, calib_file: Path) -> Dict[str, Any]:
        if not calib_file.exists():
            return {
                "benchmark_resolution": [32, 32],
                "id_distance_max": 16.0,
                "id_distance_p95": 14.8,
                "id_class_distance_max": 9.5,
                "id_norm_max": 32.0,
                "temperature": 1.0
            }
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def compute_energy(self, raw_sigmoid_score: float) -> float:
        """
        Compute free energy score E(x; T) for binary sigmoid logit.
        Sigmoid: p = 1 / (1 + exp(-z)) => z = log(p / (1 - p)).
        Logits: [0, z]. Free energy E(x; T) = -T * log(1 + exp(z / T)).
        """
        p = np.clip(raw_sigmoid_score, 1e-7, 1.0 - 1e-7)
        logit = math.log(p / (1.0 - p))
        T = max(self.temperature, 0.1)
        # Numerically stable log1p(exp)
        scaled = logit / T
        if scaled > 50:
            free_energy = -T * scaled
        elif scaled < -50:
            free_energy = 0.0
        else:
            free_energy = -T * math.log(1.0 + math.exp(scaled))
        return float(free_energy)

    def compute_entropy(self, real_prob: float) -> float:
        """Compute binary Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
        p = float(np.clip(real_prob, 1e-7, 1.0 - 1e-7))
        q = 1.0 - p
        return float(-p * math.log2(p) - q * math.log2(q))

    def evaluate(
        self,
        image_rgb: np.ndarray,
        feature_embedding: Optional[np.ndarray] = None,
        raw_score: Optional[float] = None
    ) -> OODResult:
        """
        Execute comprehensive multi-factor OOD and uncertainty evaluation.

        Args:
            image_rgb: Full native resolution RGB image array (uint8).
            feature_embedding: Optional 128-dim dense feature embedding from ResNet50.
            raw_score: Optional raw sigmoid output score P(REAL) from ResNet50.

        Returns:
            OODResult containing manifold distances, energy scores, and domain shift diagnosis.
        """
        reasons = []
        h, w = image_rgb.shape[:2]
        native_res = (int(h), int(w))

        # 1. Spatial Resolution & Downsampling Domain Shift
        # CIFAKE benchmark is strictly 32x32.
        scale_h = h / float(self.benchmark_resolution[0])
        scale_w = w / float(self.benchmark_resolution[1])
        scale_factor = round(float(max(scale_h, scale_w)), 2)

        spatial_ood_score = 0.0
        if scale_factor > 8.0:  # e.g., > 256x256
            spatial_ood_score = min(1.0, 0.40 + (scale_factor / 30.0) * 0.40)
            reasons.append(
                f"High native resolution ({w}x{h}, {scale_factor:.1f}x CIFAKE benchmark scale): "
                f"Input resolution significantly exceeds native 32x32 benchmark manifold."
            )
        elif scale_factor > 2.5:  # e.g., > 80x80
            spatial_ood_score = 0.25
            reasons.append(
                f"Non-benchmark resolution ({w}x{h}, {scale_factor:.1f}x benchmark scale)."
            )

        # 2. Embedding Manifold Distance (if feature embedding is provided)
        d_real = 0.0
        d_ai = 0.0
        d_overall = 0.0
        min_class_dist = 0.0
        cos_real = 0.0
        cos_ai = 0.0
        manifold_ood_score = 0.0

        if feature_embedding is not None and len(self.centroid_real) > 0:
            feat = feature_embedding.flatten().astype(np.float32)
            norm = float(np.linalg.norm(feat))

            d_real = float(np.linalg.norm(feat - self.centroid_real))
            d_ai = float(np.linalg.norm(feat - self.centroid_ai))
            d_overall = float(np.linalg.norm(feat - self.centroid_overall))
            min_class_dist = float(min(d_real, d_ai))

            norm_real = float(np.linalg.norm(self.centroid_real))
            norm_ai = float(np.linalg.norm(self.centroid_ai))

            cos_real = float(np.dot(feat, self.centroid_real) / (norm * norm_real + 1e-8))
            cos_ai = float(np.dot(feat, self.centroid_ai) / (norm * norm_ai + 1e-8))

            # Check manifold thresholds
            if min_class_dist > self.id_class_dist_max:
                excess_ratio = (min_class_dist - self.id_class_dist_max) / max(self.id_class_dist_max, 1.0)
                manifold_ood_score = min(1.0, 0.50 + excess_ratio * 0.50)
                reasons.append(
                    f"Feature manifold displacement: distance to nearest class centroid ({min_class_dist:.2f}) "
                    f"exceeds in-distribution maximum ({self.id_class_dist_max:.2f})."
                )
            elif d_overall > self.id_dist_p95:
                manifold_ood_score = 0.35
                reasons.append(
                    f"Feature centroid distance ({d_overall:.2f}) lies in outer 5% tail of ID distribution."
                )

            # Check feature norm
            if norm < (self.id_norm_max * 0.40):
                manifold_ood_score = max(manifold_ood_score, 0.45)
                reasons.append(
                    f"Abnormally low feature norm ({norm:.2f} vs ID avg {self.calibration.get('id_norm_mean', 23.3):.2f})."
                )

        # 3. Energy Score & Predictive Uncertainty
        energy = 0.0
        entropy = 0.0
        if raw_score is not None:
            energy = self.compute_energy(raw_score)
            entropy = self.compute_entropy(raw_score)

        # 4. Composite OOD Score
        # Weight spatial scale shift and manifold distance
        if feature_embedding is not None and len(self.centroid_real) > 0:
            composite_ood = 0.55 * manifold_ood_score + 0.45 * spatial_ood_score
        else:
            composite_ood = spatial_ood_score

        # Determine verdict
        if composite_ood >= 0.50 or (scale_factor > 8.0 and min_class_dist > self.id_class_dist_max * 0.90):
            is_ood = True
            verdict = "OUT_OF_DISTRIBUTION"
            # Discount CIFAKE classifier down to 0.15 - 0.45
            confidence_discount = min(0.50, max(0.15, round(1.0 - composite_ood, 2)))
        elif composite_ood >= 0.25:
            is_ood = False
            verdict = "BORDERLINE"
            confidence_discount = max(0.50, round(1.0 - composite_ood * 0.60, 2))
            reasons.append("Sample exhibits minor covariate domain shift away from CIFAKE benchmark.")
        else:
            is_ood = False
            verdict = "IN_DISTRIBUTION"
            confidence_discount = 1.00

        return OODResult(
            is_ood=is_ood,
            ood_score=round(float(composite_ood), 4),
            verdict=verdict,
            confidence_discount=confidence_discount,
            min_class_distance=round(min_class_dist, 4),
            distance_to_real_centroid=round(d_real, 4),
            distance_to_ai_centroid=round(d_ai, 4),
            distance_to_overall_centroid=round(d_overall, 4),
            cosine_sim_real=round(cos_real, 4),
            cosine_sim_ai=round(cos_ai, 4),
            energy_score=round(energy, 4),
            predictive_entropy=round(entropy, 4),
            resolution_scale_factor=scale_factor,
            native_resolution=native_res,
            domain_shift_reasons=reasons
        )


_GLOBAL_OOD_DETECTOR: Optional[OODDetector] = None

def get_ood_detector(calibration_path: Optional[Union[str, Path]] = None) -> OODDetector:
    """Return singleton instance of OODDetector."""
    global _GLOBAL_OOD_DETECTOR
    if _GLOBAL_OOD_DETECTOR is None:
        _GLOBAL_OOD_DETECTOR = OODDetector(calibration_path=calibration_path)
    return _GLOBAL_OOD_DETECTOR
