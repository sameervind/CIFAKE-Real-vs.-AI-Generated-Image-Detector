"""
Modern Multi-Scale Forensic Analyzer for AI Image Authenticity Detection.

Performs physical optical and statistical signal forensics directly on native,
un-downsampled images to identify synthetic generative traces that deep CNNs
miss after spatial downsampling:
1. Sensor PRNU noise residual and cross-channel optical correlation.
2. High-frequency Laplacian dispersion and heavy-tailed kurtosis.
3. 2D Spectral off-axis power and periodic lattice resonance.
4. Transverse chromatic aberration (TCA) physical lens consistency.
5. Surface texture plasticity and micro-structural smoothing.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image


@dataclass
class ForensicReport:
    """Structured report detailing multi-scale forensic analysis (supplementary optical diagnostics)."""
    spectral_peak_ratio: float    # Peak to mean FFT energy ratio
    off_axis_max_power: float     # Off-axis 2D FFT synthetic harmonic power
    high_to_low_freq_ratio: float # High-frequency to low-frequency energy ratio
    laplacian_kurtosis: float     # Gradient distribution kurtosis
    laplacian_variance: float     # Edge energy variance
    noise_residual_std: float     # High-pass sensor noise residual std
    plasticity_index: float       # Edge sharpness / flat micro-texture ratio
    chromatic_edge_diff: float    # R-B channel misalignment along edges
    detected_anomalies: List[str] # Descriptive diagnostic notes
    anomaly_count: int            # Total number of diagnostic anomaly flags
    forensic_explanation: str     # Human-readable physical diagnostic summary
    # Deprecated fields retained for serialization backward compatibility (NOT calibrated probabilities)
    synthetic_probability: Optional[float] = None
    real_probability: Optional[float] = None
    is_synthetic_likely: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiScaleForensicAnalyzer:
    """
    Forensic engine analyzing native un-downsampled optical and spectral signals.
    """

    def __init__(
        self,
        kurtosis_threshold: float = 14.0,
        spectral_peak_threshold: float = 6.0,
        off_axis_threshold: float = 0.35,
        plasticity_threshold: float = 6.5
    ):
        self.kurtosis_threshold = kurtosis_threshold
        self.spectral_peak_threshold = spectral_peak_threshold
        self.off_axis_threshold = off_axis_threshold
        self.plasticity_threshold = plasticity_threshold

    def analyze(self, image_input: Union[str, np.ndarray, Image.Image]) -> ForensicReport:
        """
        Execute full multi-scale forensic analysis on the native resolution image.

        Args:
            image_input: File path, numpy RGB array, or PIL Image.

        Returns:
            ForensicReport containing physical forensic metrics and AI probability.
        """
        rgb = self._load_rgb(image_input)
        h, w, c = rgb.shape
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        anomalies = []

        # -------------------------------------------------------------
        # 1. 2D FFT Spectral Analysis on Native Image
        # -------------------------------------------------------------
        # Crop or resize large image centrally to maximum 512x512 for FFT efficiency
        fft_size = min(512, max(32, min(h, w)))
        if h != fft_size or w != fft_size:
            # Central crop to preserve native pixel grid without interpolation resampling
            start_y = max(0, (h - fft_size) // 2)
            start_x = max(0, (w - fft_size) // 2)
            crop_gray = gray[start_y:start_y + fft_size, start_x:start_x + fft_size]
        else:
            crop_gray = gray

        f_transform = np.fft.fft2(crop_gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        mag_spectrum = np.log(np.abs(f_shift) + 1.0)

        cy, cx = mag_spectrum.shape[0] // 2, mag_spectrum.shape[1] // 2
        # Mask DC center
        mag_no_dc = mag_spectrum.copy()
        cv2.circle(mag_no_dc, (cx, cy), max(2, fft_size // 16), 0, -1)

        peak_val = float(np.max(mag_no_dc))
        mean_val = float(np.mean(mag_no_dc)) + 1e-6
        spectral_peak_ratio = round(peak_val / mean_val, 3)

        # Off-axis power (diagonal grid artifacts typical of GAN/diffusion upscalers)
        diag_mask = np.zeros_like(mag_spectrum, dtype=bool)
        for i in range(fft_size):
            for j in range(fft_size):
                if abs((i - cy) - (j - cx)) <= 2 or abs((i - cy) + (j - cx)) <= 2:
                    diag_mask[i, j] = True
        diag_mask[cy - 3:cy + 4, cx - 3:cx + 4] = False
        off_axis_power = float(np.mean(mag_spectrum[diag_mask])) / (mean_val + 1e-6)
        off_axis_power = round(off_axis_power, 3)

        # Frequency ratio: outer 40% vs inner 40%
        y, x = np.ogrid[:fft_size, :fft_size]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        high_freq_mask = dist_from_center > (0.50 * max_r)
        low_freq_mask = (dist_from_center <= (0.50 * max_r)) & (dist_from_center > (0.05 * max_r))

        high_energy = float(np.mean(mag_spectrum[high_freq_mask])) if np.sum(high_freq_mask) > 0 else 1.0
        low_energy = float(np.mean(mag_spectrum[low_freq_mask])) if np.sum(low_freq_mask) > 0 else 1.0
        high_to_low_ratio = round(high_energy / max(low_energy, 1e-6), 4)

        if spectral_peak_ratio > self.spectral_peak_threshold:
            anomalies.append(
                f"Spectral flag: Elevated Fourier peak-to-mean ratio ({spectral_peak_ratio:.2f} > {self.spectral_peak_threshold:.1f}) "
                f"suggests possible periodic lattice or sampling patterns."
            )

        if off_axis_power > self.off_axis_threshold and spectral_peak_ratio > 4.5:
            anomalies.append(
                f"Off-axis frequency flag: ({off_axis_power:.3f} > {self.off_axis_threshold:.2f}) "
                f"suggests possible diagonal upsampling harmonics."
            )

        # -------------------------------------------------------------
        # 2. Laplacian Dispersion & Gradient Kurtosis
        # -------------------------------------------------------------
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_var = float(np.var(lap))
        lap_mean = float(np.mean(lap))
        lap_std = float(np.std(lap)) + 1e-6
        lap_kurtosis = float(np.mean(((lap - lap_mean) / lap_std) ** 4))
        lap_kurtosis = round(lap_kurtosis, 2)

        if lap_kurtosis > self.kurtosis_threshold:
            anomalies.append(
                f"Gradient kurtosis flag ({lap_kurtosis:.1f} > {self.kurtosis_threshold:.1f}): "
                f"Heavy-tailed gradient distribution (may arise from synthetic edges or camera depth-of-field bokeh)."
            )

        # -------------------------------------------------------------
        # 3. Sensor PRNU High-Pass Residual Analysis
        # -------------------------------------------------------------
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
        noise_res = gray.astype(np.float32) - blurred.astype(np.float32)
        noise_std = round(float(np.std(noise_res)), 3)

        # -------------------------------------------------------------
        # 4. Texture Plasticity Index
        # -------------------------------------------------------------
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        p85 = float(np.percentile(grad_mag, 85))
        p30 = float(np.percentile(grad_mag, 30))

        edge_mask = grad_mag > p85
        flat_mask = grad_mag < p30

        flat_micro_std = float(np.std(gray[flat_mask])) if np.sum(flat_mask) > 0 else 1.0
        edge_sharpness = float(np.mean(grad_mag[edge_mask])) if np.sum(edge_mask) > 0 else 1.0
        plasticity_ratio = round(edge_sharpness / max(flat_micro_std, 1.0), 2)

        if plasticity_ratio > self.plasticity_threshold and min(h, w) > 64:
            anomalies.append(
                f"Plasticity flag ({plasticity_ratio:.1f} > {self.plasticity_threshold:.1f}): "
                f"High edge-to-flat ratio (common in both diffusion smoothing and studio portraits with blurred backgrounds)."
            )

        # -------------------------------------------------------------
        # 5. Transverse Chromatic Aberration (TCA) Check
        # -------------------------------------------------------------
        r_ch = rgb[:, :, 0].astype(np.float32)
        b_ch = rgb[:, :, 2].astype(np.float32)
        diff_rb = np.abs(r_ch - b_ch)
        chromatic_edge_diff = round(float(np.mean(diff_rb[edge_mask])) if np.sum(edge_mask) > 0 else 0.0, 2)

        # -------------------------------------------------------------
        # 6. Diagnostic Telemetry Summary (Non-Authoritative)
        # -------------------------------------------------------------
        if anomalies:
            explanation = (
                f"Supplementary optical diagnostics: {len(anomalies)} signal deviation(s) noted. "
                f"Laplacian kurtosis={lap_kurtosis:.1f}, spectral peak ratio={spectral_peak_ratio:.2f}x, "
                f"surface plasticity={plasticity_ratio:.1f}, PRNU noise std={noise_std:.2f}. "
                f"Note: Optical diagnostics are supplementary observations, not a probabilistic AI verdict."
            )
        else:
            explanation = (
                f"Supplementary optical diagnostics: Optical measurements align with typical photographic frequency profiles. "
                f"Laplacian kurtosis={lap_kurtosis:.1f}, spectral peak ratio={spectral_peak_ratio:.2f}x."
            )

        return ForensicReport(
            spectral_peak_ratio=spectral_peak_ratio,
            off_axis_max_power=off_axis_power,
            high_to_low_freq_ratio=high_to_low_ratio,
            laplacian_kurtosis=lap_kurtosis,
            laplacian_variance=round(lap_var, 2),
            noise_residual_std=noise_std,
            plasticity_index=plasticity_ratio,
            chromatic_edge_diff=chromatic_edge_diff,
            detected_anomalies=anomalies,
            anomaly_count=len(anomalies),
            forensic_explanation=explanation,
            synthetic_probability=None,
            real_probability=None,
            is_synthetic_likely=False,
            confidence=0.0
        )

    def _load_rgb(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Helper to convert input to RGB uint8 numpy array."""
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise FileNotFoundError(f"Cannot load image at {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            return arr
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")


_GLOBAL_FORENSIC_ANALYZER: Optional[MultiScaleForensicAnalyzer] = None

def get_forensic_analyzer() -> MultiScaleForensicAnalyzer:
    """Return singleton instance of MultiScaleForensicAnalyzer."""
    global _GLOBAL_FORENSIC_ANALYZER
    if _GLOBAL_FORENSIC_ANALYZER is None:
        _GLOBAL_FORENSIC_ANALYZER = MultiScaleForensicAnalyzer()
    return _GLOBAL_FORENSIC_ANALYZER
