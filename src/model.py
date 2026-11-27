"""
ResNet50 Transfer Learning Model Architecture & Inference Service for CIFAKE.
Implements ImageNet-pretrained ResNet50 backbone with GlobalAveragePooling2D,
BatchNormalization, Dropout(0.4), and Dense(1, sigmoid) output representing P(REAL).
Strictly eliminates silent fallback heuristics: ResNet50 is the primary, authoritative engine.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, Model
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.dataset import TARGET_SIZE, preprocess_image, CLASS_NAMES

# Default checkpoint path
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "resnet50_cifake.keras"


class ModelUnavailableError(Exception):
    """Raised when the trained ResNet50 model weights cannot be loaded."""
    pass


def build_resnet50_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 0.001,
    freeze_backbone: bool = True
) -> Any:
    """
    Builds the production transfer-learning ResNet50 architecture:
    
    Input (224, 224, 3)
      ↓
    ResNet50 Backbone (Pretrained on ImageNet, include_top=False)
      ↓
    GlobalAveragePooling2D
      ↓
    Dense(128, activation='relu')
      ↓
    BatchNormalization
      ↓
    Dropout(0.4)
      ↓
    Dense(1, activation='sigmoid')  -> P(REAL)
    
    Labels:
      0 = AI / SYNTHETIC
      1 = REAL PHOTOGRAPH
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to build the ResNet50 model. "
            "Install with: pip install tensorflow"
        )

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    if freeze_backbone:
        base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_feature")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_real")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cifake_resnet50_detector")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def configure_fine_tuning(
    model: Any,
    fine_tune_from_layer: str = "conv5_block1_out",
    fine_tune_lr: float = 1e-5
) -> Any:
    """
    Stage B Fine-Tuning Configuration:
    - Unfreezes top ResNet50 residual blocks (starting at `conv5_block1_out`).
    - Keeps early feature extractor layers frozen.
    - Keeps BatchNormalization layers frozen in inference mode to prevent statistic corruption.
    - Recompiles with a low learning rate (~1e-5).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model) or "resnet50" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        base_model = model

    base_model.trainable = True

    # Freeze all layers before fine_tune_from_layer
    freeze_active = True
    for layer in base_model.layers:
        if layer.name == fine_tune_from_layer:
            freeze_active = False

        if freeze_active or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


class ResNet50InferenceService:
    """
    Authoritative ResNet50 Model Inference Service.
    Loads and caches the trained Keras model, performs verified preprocessing,
    and returns genuine calibrated neural network prediction probabilities.
    Never silently falls back to heuristic engines.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = None
        self.input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
        self.backend = "tensorflow_keras"
        self._load_model(model_path)

    def _resolve_model_path(self, model_path: Optional[str] = None) -> Path:
        """Resolve checkpoint path from argument, environment variable, or default."""
        if model_path:
            return Path(model_path)
        env_path = os.getenv("CIFAKE_MODEL_PATH")
        if env_path:
            return Path(env_path)
        return DEFAULT_MODEL_PATH

    def _load_model(self, model_path: Optional[str] = None):
        """Load trained ResNet50 checkpoint."""
        resolved = self._resolve_model_path(model_path)
        self.model_path = str(resolved)

        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for ResNet50 inference. "
                "Install with: pip install tensorflow"
            )

        if not resolved.exists():
            raise ModelUnavailableError(
                f"Trained model checkpoint not found at: {resolved}. "
                "Train the model first using: python -m src.train --data_dir data/cifake "
                "or specify a valid checkpoint path via --model or CIFAKE_MODEL_PATH."
            )

        try:
            self.model = tf.keras.models.load_model(str(resolved))
            print(f"[ResNet50Service] Loaded trained checkpoint from: {resolved}")
        except Exception as e:
            raise ModelUnavailableError(
                f"Failed to load Keras model from {resolved}: {e}"
            )

    @property
    def active_backend(self) -> str:
        return "tensorflow_keras (ResNet50)"

    def is_loaded(self) -> bool:
        """Return True if model checkpoint is loaded and active."""
        return self.model is not None

    def is_model_loaded(self) -> bool:
        """Alias for is_loaded."""
        return self.is_loaded()

    def predict(self, image_input: Any, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Execute ResNet50 inference on an input image.
        
        Args:
            image_input: File path, PIL Image, or numpy array.
            threshold: Decision boundary for REAL class (default 0.5).
            
        Returns:
            Dictionary with label, class_id, probabilities, confidence, and backend.
        """
        if not self.is_loaded():
            raise ModelUnavailableError(
                "No trained ResNet50 model is currently loaded. "
                "Please train or provide a model checkpoint before requesting predictions."
            )

        # Standard shared preprocessing
        preprocessed = preprocess_image(image_input, target_size=TARGET_SIZE, apply_imagenet_norm=True)
        batch = np.expand_dims(preprocessed, axis=0)

        # Raw sigmoid output: P(REAL)
        # Using __call__ for fast eager inference instead of model.predict overhead
        pred_tensor = self.model(batch, training=False)
        raw_pred = float(pred_tensor.numpy()[0][0])
        
        # Clip to avoid division by zero or extreme rounding issues while preserving genuine probabilities
        real_prob = float(np.clip(raw_pred, 0.0001, 0.9999))
        ai_prob = float(1.0 - real_prob)

        is_real = real_prob >= threshold
        class_id = 1 if is_real else 0
        class_name = "REAL" if is_real else "AI"
        confidence = real_prob if is_real else ai_prob

        return {
            "label": class_name,
            "class_id": class_id,
            "is_real": is_real,
            "is_ai": not is_real,
            "probabilities": {
                "REAL": round(real_prob, 4),
                "AI": round(ai_prob, 4)
            },
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "raw_score": round(raw_pred, 6),
            "threshold": threshold,
            "backend": "tensorflow_keras",
            "model_name": "ResNet50",
            "checkpoint_path": self.model_path
        }


# Singleton service holder
_GLOBAL_SERVICE: Optional[ResNet50InferenceService] = None


def get_inference_service(model_path: Optional[str] = None, force_reload: bool = False) -> ResNet50InferenceService:
    """Retrieve or initialize the singleton ResNet50 inference service."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None or force_reload:
        _GLOBAL_SERVICE = ResNet50InferenceService(model_path=model_path)
    return _GLOBAL_SERVICE
