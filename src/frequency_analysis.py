"""
Frequency Domain & Spatial Artifact Analysis for AI-Generated Image Detection.
Implements 2D Discrete Fourier Transform (2D FFT), Azimuthal Spectral Profiling,
High-Frequency Energy Ratio, and Spatial Gradient Variance.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


def compute_fft_spectrum(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D Fast Fourier Transform magnitude spectrum of an image.
    
    Args:
        image_rgb: HxWx3 uint8 or float32 image in RGB format.
        
    Returns:
        magnitude_spectrum: 2D array of log magnitude spectrum.
        fft_shift: 2D complex array of shifted FFT.
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()
        
    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray = gray.astype(np.float32) / 255.0
        
    # 2D FFT and center shift
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Log magnitude spectrum for visualization
    magnitude = np.abs(fft_shift)
    magnitude_spectrum = np.log(1.0 + magnitude)
    
    # Normalize to [0, 1]
    norm_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
    
    return norm_spectrum, fft_shift


def azimuthal_average(image: np.ndarray, center: Tuple[int, int] = None) -> np.ndarray:
    """
    Calculate the azimuthally averaged 1D radial profile of a 2D power spectrum.
    
    Args:
        image: 2D 1-channel array (e.g. power spectrum).
        center: (y, x) center coords. Defaults to center of image.
        
    Returns:
        1D array containing radial average from center (low freq) to edge (high freq).
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
        
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    
    # Sum values in each radial bin
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    
    # Avoid division by zero
    radial_profile = tbin / np.maximum(nr, 1)
    return radial_profile


def extract_artifact_features(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Extract frequency domain and spatial artifact metrics from an image.
    
    AI-generated images (especially from diffusion models & GANs upsampled via transposed conv)
    frequently display:
    1. Anomalous high-frequency energy tails in Fourier space.
    2. Subtle grid/checkerboard spectral spikes at specific harmonics.
    3. Over-smoothing or artificial edge gradients in spatial domain.
    """
    # Ensure uint8 RGB
    if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
            
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. 2D FFT
    norm_spectrum, fft_shift = compute_fft_spectrum(image_rgb)
    power_spectrum = np.abs(fft_shift) ** 2
    
    # 2. Radial frequency profile
    radial_prof = azimuthal_average(power_spectrum)
    max_radius = len(radial_prof)
    
    if max_radius > 4:
        split_low = int(max_radius * 0.3)
        split_high = int(max_radius * 0.7)
        low_freq_energy = float(np.sum(radial_prof[:split_low]))
        mid_freq_energy = float(np.sum(radial_prof[split_low:split_high]))
        high_freq_energy = float(np.sum(radial_prof[split_high:]))
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-8
        
        high_to_total_ratio = high_freq_energy / total_energy
        high_to_low_ratio = high_freq_energy / (low_freq_energy + 1e-8)
    else:
        high_to_total_ratio = 0.1
        high_to_low_ratio = 0.1
        
    # 3. Spatial laplacian variance (blurriness / edge crispness)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 4. Color channel covariance / distribution
    r_chan, g_chan, b_chan = cv2.split(image_rgb)
    color_std_r = float(np.std(r_chan))
    color_std_g = float(np.std(g_chan))
    color_std_b = float(np.std(b_chan))
    color_std_mean = float((color_std_r + color_std_g + color_std_b) / 3.0)
    
    # 5. Checkerboard / periodic grid detection in high-frequency annulus
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    dist_from_dc = np.hypot(x_idx - cx, y_idx - cy)
    high_freq_mask = dist_from_dc >= (0.31 * max(h, w))
    
    mag_shift = np.abs(fft_shift)
    if np.any(high_freq_mask):
        outer_max = float(np.max(mag_shift[high_freq_mask]))
        outer_mean = float(np.mean(mag_shift[high_freq_mask]))
        peak_ratio = float(outer_max / (outer_mean + 1e-8))
    else:
        outer_max = 0.0
        peak_ratio = 1.0
    
    return {
        "magnitude_spectrum": norm_spectrum,
        "radial_profile": radial_prof.tolist(),
        "high_to_total_ratio": round(high_to_total_ratio, 6),
        "high_to_low_ratio": round(high_to_low_ratio, 6),
        "laplacian_variance": round(laplacian_var, 3),
        "color_std_mean": round(color_std_mean, 3),
        "spectral_peak_ratio": round(peak_ratio, 3),
        "high_freq_max_power": round(outer_max, 3)
    }


class FrequencyArtifactDetector:
    """
    Statistical Fourier and Spatial Texture classifier for AI-generated image detection.
    Detects upsampling checkerboard patterns, spectral power decay anomalies,
    and high-frequency energy signatures.
    """

    def __init__(self):
        # Calibrated weights based on CIFAKE benchmark distributions
        self.peak_ratio_threshold = 14.0
        self.outer_power_threshold = 18.0

    def detect(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image array and return Real vs AI probabilities.
        """
        features = extract_artifact_features(image_np)
        peak_ratio = features["spectral_peak_ratio"]
        outer_power = features["high_freq_max_power"]

        # Synthetic diffusion images show distinctive high-frequency checkerboard harmonics
        if outer_power >= self.outer_power_threshold or peak_ratio >= self.peak_ratio_threshold:
            # AI / Synthetic
            excess = max((outer_power - self.outer_power_threshold) / 10.0, (peak_ratio - self.peak_ratio_threshold) / 10.0)
            ai_prob = float(np.clip(0.82 + 0.12 * excess, 0.75, 0.98))
            real_prob = float(1.0 - ai_prob)
        else:
            # Real photographic image
            deficit = (self.outer_power_threshold - outer_power) / self.outer_power_threshold
            real_prob = float(np.clip(0.80 + 0.15 * deficit, 0.70, 0.96))
            ai_prob = float(1.0 - real_prob)

        class_id = 1 if real_prob >= 0.5 else 0
        class_name = "REAL" if class_id == 1 else "AI"
        confidence = real_prob if class_id == 1 else ai_prob

        return {
            "class_name": class_name,
            "class_id": class_id,
            "real_probability": real_prob,
            "ai_probability": ai_prob,
            "confidence": confidence,
            "raw_score": real_prob,
            "artifacts": features
        }

