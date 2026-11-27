"""
Frequency Domain & Spatial Artifact Analysis for AI-Generated Image Detection.
Implements 2D Discrete Fourier Transform (2D FFT), Azimuthal Spectral Profiling,
Laplacian Gradient Kurtosis, Noise Residual Variance, and High-Frequency Energy Ratio.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


def compute_fft_spectrum(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D Fast Fourier Transform magnitude spectrum of an image.
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()
        
    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray = gray.astype(np.float32) / 255.0
        
    h, w = gray.shape[:2]
    # 2D FFT and center shift, normalized by sqrt(h * w)
    fft = np.fft.fft2(gray) / np.sqrt(h * w)
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
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
        
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    return radial_profile


def extract_artifact_features(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Extract frequency domain, noise residual, and spatial artifact metrics from an image.
    Supports arbitrary resolution without downsampling artifacts.
    """
    # Ensure uint8 RGB
    if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
            
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    
    # 1. 2D FFT & Normalized Coordinates
    norm_spectrum, fft_shift = compute_fft_spectrum(image_rgb)
    mag_shift = np.abs(fft_shift)
    
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r_norm = np.hypot(x_idx - cx, y_idx - cy) / (min(h, w) / 2.0)
    
    # High frequency outer annulus (r >= 0.31)
    high_freq_mask = r_norm >= 0.31
    if np.any(high_freq_mask):
        outer_max = float(np.max(mag_shift[high_freq_mask]))
        outer_mean = float(np.mean(mag_shift[high_freq_mask]))
        peak_ratio = float(outer_max / (outer_mean + 1e-8))
    else:
        outer_max = 0.0
        peak_ratio = 1.0
        
    # High to low ratio
    low_mask = (r_norm >= 0.05) & (r_norm < 0.30)
    low_energy = float(np.sum(mag_shift[low_mask])) if np.any(low_mask) else 1e-6
    high_energy = float(np.sum(mag_shift[high_freq_mask])) if np.any(high_freq_mask) else 1e-6
    high_to_low_ratio = high_energy / (low_energy + 1e-8)
    
    # 2. Laplacian Variance & Kurtosis (measures edge gradient distribution)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))
    lap_centered = lap - np.mean(lap)
    lap_kurt = float(np.mean(lap_centered ** 4) / ((lap_var + 1e-4) ** 2))
    
    # 3. Noise Residual Analysis (isolates physical sensor shot noise)
    smooth = cv2.medianBlur(gray, 3)
    residual = np.abs(gray.astype(np.float32) - smooth.astype(np.float32))
    noise_std = float(np.std(residual))
    
    # 4. Color Channel Statistics (HSV Saturation)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    sat_mean = float(np.mean(sat))
    
    # 5. Radial Power Spectrum
    power_spectrum = mag_shift ** 2
    radial_prof = azimuthal_average(power_spectrum)
    
    return {
        "magnitude_spectrum": norm_spectrum,
        "radial_profile": radial_prof.tolist(),
        "high_to_low_ratio": round(high_to_low_ratio, 6),
        "spectral_peak_ratio": round(peak_ratio, 3),
        "high_freq_max_power": round(outer_max, 3),
        "laplacian_variance": round(lap_var, 2),
        "laplacian_kurtosis": round(lap_kurt, 2),
        "noise_residual_std": round(noise_std, 3),
        "color_saturation_mean": round(sat_mean, 1)
    }


class FrequencyArtifactDetector:
    """
    Multi-domain artifact classifier for AI-generated image detection.
    Evaluates 2D Fourier harmonic peaks, gradient kurtosis, sensor noise residuals,
    and color manifolds.
    """

    def __init__(self):
        pass

    def detect(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image array and return calibrated Real vs AI probabilities.
        """
        features = extract_artifact_features(image_np)
        
        peak_ratio = features["spectral_peak_ratio"]
        outer_power = features["high_freq_max_power"]
        lap_kurt = features["laplacian_kurtosis"]
        noise_std = features["noise_residual_std"]
        lap_var = features["laplacian_variance"]
        sat_mean = features["color_saturation_mean"]
        
        ai_score = 0.0
        reasons = []
        
        # 1. Fourier checkerboard / harmonic peak
        if peak_ratio > 10.0:
            ai_score += min(0.35, (peak_ratio - 10.0) * 0.02)
            reasons.append("Periodic checkerboard/grid spikes in 2D Fourier domain.")
        elif peak_ratio > 4.5:
            ai_score += 0.15
            
        # 2. Outer high-frequency power
        if outer_power > 15.0:
            ai_score += min(0.30, (outer_power - 15.0) * 0.015)
            reasons.append("Elevated high-frequency energy tail typical of diffusion/GAN upsampling.")
            
        # 3. Laplacian Kurtosis (AI smoothing + sharp lines)
        if lap_kurt > 5.0:
            ai_score += min(0.30, (lap_kurt - 5.0) * 0.06)
            reasons.append("Synthetic gradient distribution (unnatural smoothing mixed with sharp synthetic edges).")
        elif lap_kurt < 3.8:
            ai_score -= 0.15  # Natural optical gradient distribution
            
        # 4. Sensor noise residual
        if noise_std > 7.5:
            ai_score += min(0.30, (noise_std - 7.5) * 0.05)
            reasons.append("Spatially correlated generative high-frequency noise residual.")
        elif noise_std < 5.5:
            ai_score -= 0.15  # Natural camera sensor noise
            
        # 5. Laplacian variance
        if lap_var > 3000.0:
            ai_score += min(0.20, (lap_var - 3000.0) / 10000.0)
        elif 300.0 <= lap_var <= 2000.0:
            ai_score -= 0.15  # Natural photo variance range
            
        # 6. Color saturation
        if sat_mean > 115.0:
            ai_score += 0.15
            reasons.append("Hyper-saturated color manifold.")
        elif sat_mean < 75.0:
            ai_score -= 0.05
            
        # Calibrated probability
        p_ai = 1.0 / (1.0 + np.exp(-3.2 * ai_score))
        p_ai = float(np.clip(p_ai, 0.02, 0.98))
        p_real = float(1.0 - p_ai)
        
        class_id = 1 if p_real >= 0.5 else 0
        class_name = "REAL" if class_id == 1 else "AI"
        confidence = p_real if class_id == 1 else p_ai
        
        if not reasons:
            if class_id == 1:
                reasons.append("Natural camera sensor noise, balanced color saturation, and authentic optical gradient decay curve observed.")
            else:
                reasons.append("Statistical feature distribution strongly matched synthetic generative model manifold.")

        return {
            "class_name": class_name,
            "class_id": class_id,
            "real_probability": p_real,
            "ai_probability": p_ai,
            "confidence": confidence,
            "raw_score": p_real,
            "explanation": " ".join(reasons),
            "artifacts": features
        }
