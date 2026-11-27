"""
CIFAKE Dataset utility for downloading or creating sample evaluation datasets.
CIFAKE contains 60,000 real CIFAR-10 images and 60,000 synthetic images generated via Stable Diffusion 1.4.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2


def generate_synthetic_cifake_sample(
    output_dir: str = "data/sample_cifake",
    samples_per_class: int = 50
):
    """
    Generate authentic CIFAR-10-like real sample images and diffusion-like synthetic samples
    for immediate testing and pipeline validation.
    """
    out_path = Path(output_dir)
    real_dir = out_path / "REAL"
    fake_dir = out_path / "FAKE"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    print(f"Generating {samples_per_class} sample REAL images in {real_dir}...")
    for i in range(samples_per_class):
        # Real camera images have natural smooth gradient, optical blur, and Poisson/Gaussian sensor noise
        base = np.zeros((32, 32, 3), dtype=np.float32)
        # Random natural color gradient
        c1 = rng.uniform(0.2, 0.8, size=(3,))
        c2 = rng.uniform(0.1, 0.7, size=(3,))
        for y in range(32):
            w = y / 31.0
            base[y, :, :] = (1 - w) * c1 + w * c2

        # Add geometric shapes (objects)
        pt1 = (rng.integers(5, 15), rng.integers(5, 15))
        pt2 = (rng.integers(16, 28), rng.integers(16, 28))
        obj_color = rng.uniform(0.1, 0.9, size=(3,))
        cv2.rectangle(base, pt1, pt2, obj_color.tolist(), -1)
        
        # Natural optical blur
        base = cv2.GaussianBlur(base, (3, 3), 0.8)
        
        # Natural sensor noise (1/f decay + gaussian)
        sensor_noise = rng.normal(0, 0.04, base.shape).astype(np.float32)
        img_real = np.clip(base + sensor_noise, 0.0, 1.0)
        img_real_uint8 = (img_real * 255).astype(np.uint8)
        
        # Save as BGR for OpenCV
        cv2.imwrite(str(real_dir / f"real_{i:04d}.jpg"), cv2.cvtColor(img_real_uint8, cv2.COLOR_RGB2BGR))

    print(f"Generating {samples_per_class} sample FAKE/AI images in {fake_dir}...")
    for i in range(samples_per_class):
        # AI/Diffusion images have subtle upsampling periodic grids, over-saturated hues, and high-freq spectral peaks
        base = np.zeros((32, 32, 3), dtype=np.float32)
        c1 = rng.uniform(0.3, 0.9, size=(3,))
        c2 = rng.uniform(0.2, 0.85, size=(3,))
        for y in range(32):
            for x in range(32):
                base[y, x, :] = 0.5 * c1 + 0.5 * c2 * np.sin(x / 3.0) * np.cos(y / 3.0)

        # Upsampling checkerboard artifact (characteristic of transposed conv / latent decoders)
        grid = np.zeros((32, 32, 1), dtype=np.float32)
        grid[::2, ::2, 0] = 0.08
        grid[1::2, 1::2, 0] = -0.08
        
        # Smooth synthetic object
        cv2.circle(base, (16, 16), rng.integers(6, 12), rng.uniform(0.4, 1.0, size=(3,)).tolist(), -1)
        base = cv2.GaussianBlur(base, (3, 3), 0.4)
        
        img_fake = np.clip(base + grid, 0.0, 1.0)
        img_fake_uint8 = (img_fake * 255).astype(np.uint8)
        cv2.imwrite(str(fake_dir / f"fake_{i:04d}.jpg"), cv2.cvtColor(img_fake_uint8, cv2.COLOR_RGB2BGR))

    print(f"Sample dataset ready at {output_dir}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sample CIFAKE dataset.")
    parser.add_argument("--samples", type=int, default=50, help="Samples per class.")
    parser.add_argument("--out", type=str, default="data/sample_cifake", help="Output directory.")
    args = parser.parse_args()
    generate_synthetic_cifake_sample(output_dir=args.out, samples_per_class=args.samples)
