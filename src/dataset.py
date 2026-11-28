"""
Dataset management, loading, unified preprocessing, and augmentation for CIFAKE Real vs. AI images.
Provides shared ImageNet-standard preprocessing for both training and inference pipelines.
Supports multi-structure directory discovery (flat REAL/FAKE or train/test splits),
genuine stratified partitioning, and memory-efficient tf.data pipelines.
"""

import os
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Official CIFAKE class labels: 0 = AI / FAKE, 1 = REAL
CLASS_NAMES = ["AI", "REAL"]
LABEL_MAP = {
    "FAKE": 0, "AI": 0, "fake": 0, "ai": 0, "synthetic": 0, "SYNTHETIC": 0,
    "REAL": 1, "real": 1, "photograph": 1, "PHOTOGRAPH": 1
}

# Pretrained ResNet50 input dimension
TARGET_SIZE = (224, 224)


def preprocess_image(
    image_input: Any,
    target_size: Tuple[int, int] = TARGET_SIZE,
    apply_imagenet_norm: bool = True
) -> np.ndarray:
    """
    Unified Image Preprocessing Pipeline for both Training and Inference.
    
    1. Loads or extracts raw RGB image array from path, PIL Image, or numpy array.
    2. Resizes to target resolution (224, 224) using area or linear interpolation.
    3. Applies standard ImageNet ResNet50 zero-centering:
       - Converts RGB to BGR
       - Subtracts ImageNet channel means: [103.939, 116.779, 123.68]
       
    Args:
        image_input: File path (str/Path), PIL Image, or numpy array.
        target_size: Tuple of (height, width), default (224, 224).
        apply_imagenet_norm: If True, applies ResNet50 ImageNet mean subtraction.
        
    Returns:
        np.ndarray of shape (target_size[0], target_size[1], 3) as float32.
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
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure uint8 [0, 255] base
    if img_rgb.dtype != np.uint8:
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255.0).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)

    # Consistent spatial resizing
    h_curr, w_curr = img_rgb.shape[:2]
    h_tgt, w_tgt = target_size
    if (h_curr, w_curr) != (h_tgt, w_tgt):
        interp = cv2.INTER_AREA if (h_curr > h_tgt or w_curr > w_tgt) else cv2.INTER_LINEAR
        img_rgb = cv2.resize(img_rgb, (w_tgt, h_tgt), interpolation=interp)

    img_float = img_rgb.astype(np.float32)

    if apply_imagenet_norm:
        # Standard ResNet50 ImageNet preprocessing (caffe style: RGB -> BGR with mean subtraction)
        # BGR channel means: B=103.939, G=116.779, R=123.68
        bgr = img_float[:, :, ::-1].copy()
        bgr[:, :, 0] -= 103.939
        bgr[:, :, 1] -= 116.779
        bgr[:, :, 2] -= 123.68
        return bgr

    return img_float


def augment_image_numpy(image_bgr: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Training-only spatial augmentation:
      - Random horizontal flip (50%)
      - Subtle brightness shift (+/- 8 units)
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image_bgr.copy()

    # 1. Random horizontal flip
    if rng.random() > 0.5:
        img = np.fliplr(img)

    # 2. Subtle brightness adjustment
    delta = rng.uniform(-8.0, 8.0)
    img = img + delta

    return img.astype(np.float32)


def discover_dataset_structure(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect and validate the structure of the CIFAKE dataset directory.
    Supports:
      Structure A: data_dir/{REAL, FAKE}/
      Structure B: data_dir/{train, test}/{REAL, FAKE}/
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir}")

    has_train = (root / "train").is_dir()
    has_test = (root / "test").is_dir()

    if has_train and has_test:
        return {
            "mode": "split_train_test",
            "train_dir": root / "train",
            "test_dir": root / "test",
            "val_dir": root / "val" if (root / "val").is_dir() else None
        }

    # Check for direct class directories
    subdirs = [d.name.upper() for d in root.iterdir() if d.is_dir()]
    has_real = any("REAL" in s for s in subdirs)
    has_fake = any("FAKE" in s or "AI" in s for s in subdirs)

    if has_real and has_fake:
        return {
            "mode": "flat_class_dirs",
            "root_dir": root
        }

    raise ValueError(
        f"Unrecognized dataset directory structure in '{data_dir}'. "
        "Expected either direct 'REAL/' and 'FAKE/' subdirectories, "
        "or partitioned 'train/' and 'test/' directories containing 'REAL/' and 'FAKE/'."
    )


def scan_directory_for_labeled_files(class_dir: Path) -> List[Tuple[str, int]]:
    """Recursively scan a directory for valid images and map to binary labels."""
    labeled_files = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Determine class from folder name
    dir_name = class_dir.name.upper()
    if "REAL" in dir_name:
        label = 1
    elif "FAKE" in dir_name or "AI" in dir_name:
        label = 0
    else:
        return []

    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            labeled_files.append((str(file_path), label))

    return labeled_files


def load_dataset_from_directory(
    data_dir: Union[str, Path],
    max_samples_per_class: Optional[int] = None,
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Scan data_dir for images, load with unified ResNet50 preprocessing, and validate distributions.
    
    Returns:
        X: np.ndarray of shape (N, target_size[0], target_size[1], 3) float32 (ImageNet normalized).
        y: np.ndarray of shape (N,) int labels (0 = AI, 1 = REAL).
        paths: List of file path strings.
    """
    root = Path(data_dir)
    structure = discover_dataset_structure(root)

    labeled_entries: List[Tuple[str, int]] = []

    if structure["mode"] == "split_train_test":
        target_dirs = [structure["train_dir"]]
        for t_dir in target_dirs:
            for s_dir in t_dir.iterdir():
                if s_dir.is_dir():
                    labeled_entries.extend(scan_directory_for_labeled_files(s_dir))
    else:
        for s_dir in root.iterdir():
            if s_dir.is_dir():
                labeled_entries.extend(scan_directory_for_labeled_files(s_dir))

    if not labeled_entries:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32), np.empty((0,), dtype=np.int32), []

    # Separate by class for max_samples_per_class limit if requested
    class_0 = [e for e in labeled_entries if e[1] == 0]
    class_1 = [e for e in labeled_entries if e[1] == 1]

    if max_samples_per_class:
        class_0 = class_0[:max_samples_per_class]
        class_1 = class_1[:max_samples_per_class]

    selected_entries = class_0 + class_1
    np.random.default_rng(42).shuffle(selected_entries)

    images = []
    labels = []
    paths = []

    for fpath, lbl in selected_entries:
        try:
            img = preprocess_image(fpath, target_size=target_size, apply_imagenet_norm=True)
            images.append(img)
            labels.append(lbl)
            paths.append(fpath)
        except Exception as e:
            print(f"Skipping unreadable/corrupt image: {fpath} ({e})")

    if not images:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32), np.empty((0,), dtype=np.int32), []

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
    Genuine Stratified Dataset Partitioning using scikit-learn.
    Guarantees exact class distribution representation across Train (70%), Validation (20%), and Test (10%).
    Prevents data leakage.
    """
    from sklearn.model_selection import train_test_split

    n_total = len(X)
    if n_total == 0:
        empty = (np.empty((0, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32), np.empty((0,), dtype=np.int32))
        return empty, empty, empty

    # Check if both classes are present
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or min(counts) < 2:
        # Fallback to simple random split if insufficient samples per class for stratification
        indices = np.arange(n_total)
        np.random.default_rng(seed).shuffle(indices)
        n_tr = int(n_total * train_ratio)
        n_vl = int(n_total * val_ratio)
        tr_idx, vl_idx, te_idx = indices[:n_tr], indices[n_tr:n_tr + n_vl], indices[n_tr + n_vl:]
        return (X[tr_idx], y[tr_idx]), (X[vl_idx], y[vl_idx]), (X[te_idx], y[te_idx])

    # First split off the Test set using exact sample count
    n_test = max(1, int(round(n_total * test_ratio)))
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=n_test,
        stratify=y,
        random_state=seed
    )

    # Second split Train vs Validation using exact sample count
    n_val = max(1, int(round(n_total * val_ratio)))
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=n_val,
        stratify=y_train_val,
        random_state=seed
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_tf_data_pipeline(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    target_size: Tuple[int, int] = TARGET_SIZE,
    is_training: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42
):
    """
    High-Performance Memory-Efficient tf.data Pipeline.
    Loads and processes images in parallel without loading entire dataset into RAM.
    Applies training-only augmentations and prefetches batches with tf.data.AUTOTUNE.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to construct tf.data pipelines.")

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(file_paths), shuffle_buffer), seed=seed)

    def parse_and_preprocess(path, label):
        raw_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)

        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=10.0)

        # ResNet50 ImageNet Preprocessing: RGB -> BGR with mean subtraction
        # BGR channel means: [103.939, 116.779, 123.68]
        img_bgr = img[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        img_pre = img_bgr - mean
        return img_pre, tf.cast(label, tf.float32)

    ds = ds.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


CIFAKE_NATIVE_SHAPE = (32, 32, 3)


def load_cifake_cache(
    cache_dir: Union[str, Path] = "data/cifake_cache",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load authoritative CIFAKE dataset from cached numpy memory-mapped arrays.
    Applies genuine stratified split to the 100,000 training images to create
    reproducible Train and Validation partitions, keeping the 20,000 official Test split untouched.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        where images are shape (N, 32, 32, 3) uint8 and labels are shape (N,) int32.
    """
    from sklearn.model_selection import train_test_split

    cache_path = Path(cache_dir)
    train_img_path = cache_path / "train_images.npy"
    train_lbl_path = cache_path / "train_labels.npy"
    test_img_path = cache_path / "test_images.npy"
    test_lbl_path = cache_path / "test_labels.npy"

    if not all(p.exists() for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]):
        raise FileNotFoundError(
            f"Cached CIFAKE numpy files not found in {cache_dir}. "
            "Please ensure data/cifake_cache contains train_images.npy, train_labels.npy, etc."
        )

    # Load arrays
    X_train_full = np.load(str(train_img_path))
    y_train_full = np.load(str(train_lbl_path))
    X_test = np.load(str(test_img_path))
    y_test = np.load(str(test_lbl_path))

    # Stratified split for train vs validation
    indices = np.arange(len(y_train_full))
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=seed
    )

    X_train, y_train = X_train_full[idx_train], y_train_full[idx_train]
    X_val, y_val = X_train_full[idx_val], y_train_full[idx_val]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_cifake_native_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    is_training: bool = False,
    augment: bool = True,
    seed: int = 42
):
    """
    Construct high-throughput tf.data pipeline for native 32x32 images.
    Keeps images in uint8 until GPU/CPU tensor graph to minimize RAM bandwidth.
    Applies training-only conservative spatial data augmentation (horizontal flip, random crop).
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to construct tf.data pipelines.")

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(images), 20000), seed=seed, reshuffle_each_iteration=True)

    def preprocess_tensor(img, lbl):
        # Cast to float32 [0, 255] for augmentation / scaling
        img = tf.cast(img, tf.float32)

        if is_training and augment:
            # Random horizontal flip
            img = tf.image.random_flip_left_right(img)
            # 4-pixel reflection pad and random crop back to 32x32
            img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
            img = tf.image.random_crop(img, [32, 32, 3])

        return img, tf.cast(lbl, tf.float32)

    ds = ds.map(preprocess_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
