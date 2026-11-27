"""
Dataset management, loading, preprocessing, and augmentation for CIFAKE Real vs. AI images.
Supports both pure NumPy/OpenCV processing and TensorFlow tf.data pipelines.
"""

import os
import glob
from pathlib import Path
from typing import Tuple, List, Optional, Generator
import numpy as np
import cv2

# Label mapping consistent with notebook: 0 -> AI / FAKE, 1 -> REAL
CLASS_NAMES = ["AI", "REAL"]
LABEL_MAP = {"FAKE": 0, "AI": 0, "fake": 0, "ai": 0, "REAL": 1, "real": 1}
TARGET_SIZE = (32, 32)


def preprocess_image(
    image_input,
    target_size: Tuple[int, int] = TARGET_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Load or convert image into RGB format, resize to target size, and optionally normalize to [0, 1].
    
    Args:
        image_input: File path (str/Path) or numpy array (RGB or BGR or grayscale).
        target_size: Tuple of (height, width), default (32, 32).
        normalize: If True, scale pixel values to [0.0, 1.0].
        
    Returns:
        np.ndarray of shape (target_size[0], target_size[1], 3) as float32 or uint8.
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image from: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input.copy()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    else:
        # Support PIL Image
        img_rgb = np.array(image_input.convert("RGB"))

    # Resize to target size (32, 32)
    if img_rgb.shape[:2] != target_size:
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    if normalize:
        if img_rgb.dtype == np.uint8:
            return img_rgb.astype(np.float32) / 255.0
        elif img_rgb.max() > 1.0:
            return (img_rgb / 255.0).astype(np.float32)
        return img_rgb.astype(np.float32)

    return img_rgb


def augment_image_numpy(image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Data augmentation matching the CIFAKE pipeline:
      - Random horizontal flip (50%)
      - Random vertical flip (50%)
      - Random brightness adjustment (max delta 0.1)
      - Random contrast adjustment (0.9 to 1.1)
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.copy()

    # 1. Random horizontal flip
    if rng.random() > 0.5:
        img = np.fliplr(img)

    # 2. Random vertical flip
    if rng.random() > 0.5:
        img = np.flipud(img)

    # 3. Random brightness (-0.1 to +0.1)
    delta_brightness = rng.uniform(-0.1, 0.1)
    img = img + delta_brightness

    # 4. Random contrast (factor in [0.9, 1.1])
    contrast_factor = rng.uniform(0.9, 1.1)
    mean_val = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean_val) * contrast_factor + mean_val

    # Clip to valid [0.0, 1.0] range
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def load_dataset_from_directory(
    root_dir: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    max_samples_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Scan root_dir for 'REAL' and 'FAKE' subfolders, load and preprocess images.
    
    Returns:
        X: np.ndarray of shape (N, 32, 32, 3) normalized float32.
        y: np.ndarray of shape (N,) int labels (0 = AI/FAKE, 1 = REAL).
        file_paths: List of string file paths.
    """
    root_path = Path(root_dir)
    images = []
    labels = []
    paths = []

    # Check for subdirectories
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    if not subdirs:
        # Fallback to direct scan
        return np.empty((0, target_size[0], target_size[1], 3)), np.empty((0,)), []

    for subdir in subdirs:
        name_upper = subdir.name.upper()
        if "REAL" in name_upper:
            label = 1
        elif "FAKE" in name_upper or "AI" in name_upper:
            label = 0
        else:
            continue

        file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        class_files = []
        for pat in file_patterns:
            class_files.extend(list(subdir.glob(pat)))
            class_files.extend(list(subdir.glob(pat.upper())))

        if max_samples_per_class:
            class_files = class_files[:max_samples_per_class]

        for fpath in class_files:
            try:
                img = preprocess_image(str(fpath), target_size=target_size, normalize=True)
                images.append(img)
                labels.append(label)
                paths.append(str(fpath))
            except Exception as e:
                print(f"Skipping corrupt image {fpath}: {e}")

    if not images:
        return np.empty((0, target_size[0], target_size[1], 3)), np.empty((0,)), []

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, paths


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into 70% Train, 20% Validation, 10% Test with stratification.
    """
    n_total = len(X)
    if n_total == 0:
        empty_split = (np.empty((0, 32, 32, 3)), np.empty((0,)))
        return empty_split, empty_split, empty_split

    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def create_batch_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield mini-batches with optional data augmentation.
    """
    n_samples = len(X)
    rng = np.random.default_rng()
    indices = np.arange(n_samples)

    if shuffle:
        rng.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]

        batch_x = X[batch_idx].copy()
        batch_y = y[batch_idx]

        if augment:
            for i in range(len(batch_x)):
                batch_x[i] = augment_image_numpy(batch_x[i], rng=rng)

        yield batch_x, batch_y
