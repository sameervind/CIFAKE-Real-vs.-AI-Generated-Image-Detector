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
    from tensorflow.keras import layers, models, Model, regularizers
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.optimizers import Adam, AdamW
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.dataset import TARGET_SIZE, preprocess_image, CLASS_NAMES

# Default checkpoint paths
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
BEST_MODEL_PATH = DEFAULT_MODEL_DIR / "best_cifake_model.keras"
RESNET50_MODEL_PATH = DEFAULT_MODEL_DIR / "resnet50_cifake.keras"
DEFAULT_MODEL_PATH = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else RESNET50_MODEL_PATH


class ModelUnavailableError(Exception):
    """Raised when the trained model weights cannot be loaded."""
    pass


# ==============================================================================
# ARCHITECTURE A: LIGHTWEIGHT CIFAR CNN (Native 32x32, 6 Conv Layers, GAP)
# ==============================================================================

def build_cifar_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4
) -> Any:
    """
    Lightweight 6-layer Convolutional Neural Network designed specifically for native
    32x32 CIFAR imagery. Extremely CPU-efficient (~306k parameters).

    Architecture:
      Conv2D(32, 3x3) -> BN -> ReLU -> Conv2D(32, 3x3) -> BN -> ReLU -> MaxPool2D(2,2) -> Dropout(0.2)
      Conv2D(64, 3x3) -> BN -> ReLU -> Conv2D(64, 3x3) -> BN -> ReLU -> MaxPool2D(2,2) -> Dropout(0.3)
      Conv2D(128, 3x3) -> BN -> ReLU -> Conv2D(128, 3x3) -> BN -> ReLU -> MaxPool2D(2,2) -> Dropout(0.4)
      GlobalAveragePooling2D -> Dense(128, ReLU) -> BN -> Dropout(0.4) -> Dense(1, Sigmoid)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the model.")

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Block 1 (32x32 -> 16x16)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2 (16x16 -> 8x8)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 3 (8x8 -> 4x4)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    feat = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_feature")(x)
    x = layers.BatchNormalization()(feat)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="output_real")(x)

    model = Model(inputs=inputs, outputs=out, name="cifar_cnn")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# ARCHITECTURE B: CIFAR RESNET (He et al. 2016 for 32x32 CIFAR Imagery)
# ==============================================================================

def _resnet_conv_block(inputs, num_filters=16, kernel_size=3, strides=1, activation="relu", batch_norm=True, l2_reg=1e-4):
    conv = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg)
    )
    x = conv(inputs)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x


def build_cifar_resnet(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    depth: int = 20,
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4
) -> Any:
    """
    Classical He et al. ResNet architecture adapted for 32x32 CIFAR imagery.
    Depth must be 6n + 2 (e.g. 20, 32, 44, 56).
    
    Stem:
      3x3 Conv with stride 1 and 16 filters (NO early max-pooling).
      Preserves full 32x32 discrete spatial frequency information.
      
    Stages:
      Stage 1: n residual blocks with 16 filters (32x32)
      Stage 2: n residual blocks with 32 filters (16x16, stride 2)
      Stage 3: n residual blocks with 64 filters (8x8, stride 2)
      
    Head:
      GlobalAveragePooling2D -> Dense(128, ReLU) -> BN -> Dropout(0.3) -> Dense(1, Sigmoid)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the model.")

    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n + 2 (e.g. 20, 32, 44, 56)")
    num_res_blocks = (depth - 2) // 6

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Stem: 3x3 conv, stride 1 (no pooling)
    x = _resnet_conv_block(x, num_filters=16, strides=1, l2_reg=l2_reg)

    # 3 Residual Stages
    num_filters = 16
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # downsample
            y = _resnet_conv_block(x, num_filters=num_filters, strides=strides, l2_reg=l2_reg)
            y = _resnet_conv_block(y, num_filters=num_filters, activation=None, l2_reg=l2_reg)
            if stack > 0 and res_block == 0:
                # 1x1 projection shortcut
                x = _resnet_conv_block(x, num_filters=num_filters, kernel_size=1, strides=strides,
                                       activation=None, batch_norm=True, l2_reg=l2_reg)
            x = layers.add([x, y])
            x = layers.Activation("relu")(x)
        num_filters *= 2

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    feat = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_feature")(x)
    x = layers.BatchNormalization()(feat)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="output_real")(x)

    model = Model(inputs=inputs, outputs=out, name=f"cifar_resnet{depth}")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cifar_resnet20(learning_rate: float = 0.001, l2_reg: float = 1e-4) -> Any:
    return build_cifar_resnet(input_shape=(32, 32, 3), depth=20, learning_rate=learning_rate, l2_reg=l2_reg)


def build_cifar_resnet32(learning_rate: float = 0.001, l2_reg: float = 1e-4) -> Any:
    return build_cifar_resnet(input_shape=(32, 32, 3), depth=32, learning_rate=learning_rate, l2_reg=l2_reg)


# ==============================================================================
# ARCHITECTURE C: RESNET-50 TRANSFER LEARNING (ImageNet 224x224 Backbone)
# ==============================================================================


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
        if BEST_MODEL_PATH.exists():
            return BEST_MODEL_PATH
        if RESNET50_MODEL_PATH.exists():
            return RESNET50_MODEL_PATH
        return BEST_MODEL_PATH

    def _load_model(self, model_path: Optional[str] = None):
        """Load trained model checkpoint."""
        resolved = self._resolve_model_path(model_path)
        self.model_path = str(resolved)

        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for CIFAKE inference. "
                "Install with: pip install tensorflow"
            )

        if not resolved.exists():
            # Check for remote download URL from environment or Streamlit secrets
            model_url = os.getenv("CIFAKE_MODEL_URL")
            if not model_url:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and "CIFAKE_MODEL_URL" in st.secrets:
                        model_url = st.secrets["CIFAKE_MODEL_URL"]
                except Exception:
                    pass

            if model_url:
                try:
                    print(f"[CIFAKEService] Downloading model from {model_url} to {resolved}...")
                    import urllib.request
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(model_url, str(resolved))
                    print(f"[CIFAKEService] Successfully downloaded model to {resolved}")
                except Exception as dl_err:
                    print(f"[CIFAKEService] Failed to download model from {model_url}: {dl_err}")

        if not resolved.exists():
            raise ModelUnavailableError(
                f"Trained model checkpoint not found at: {resolved}. "
                "Train the model first using: python -m src.train "
                "or click 'Initialize Model Engine' in the dashboard, "
                "or specify a valid checkpoint path via --model or CIFAKE_MODEL_PATH."
            )

        try:
            self.model = tf.keras.models.load_model(str(resolved))
            # Determine input shape
            in_shape = self.model.input_shape
            if in_shape and len(in_shape) >= 3 and in_shape[1] is not None:
                self.input_shape = (in_shape[1], in_shape[2], in_shape[3] if len(in_shape) > 3 else 3)
            else:
                self.input_shape = (32, 32, 3)

            try:
                dense_layer = self.model.get_layer("dense_feature")
                out_layer = self.model.get_layer("output_real")
                self.feature_extractor = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[dense_layer.output, out_layer.output]
                )
            except Exception:
                self.feature_extractor = None
            print(f"[CIFAKEService] Loaded trained checkpoint from: {resolved} (Input shape: {self.input_shape})")
        except Exception as e:
            raise ModelUnavailableError(
                f"Failed to load Keras model from {resolved}: {e}"
            )

    @property
    def active_backend(self) -> str:
        model_name = getattr(self.model, "name", "CIFAKE_Model") if self.model else "None"
        return f"tensorflow_keras ({model_name})"

    def is_loaded(self) -> bool:
        """Return True if model checkpoint is loaded and active."""
        return self.model is not None

    def is_model_loaded(self) -> bool:
        """Alias for is_loaded."""
        return self.is_loaded()

    def predict(self, image_input: Any, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Execute neural network inference on an input image.
        Dynamically adapts preprocessing to the model's native resolution.
        
        Args:
            image_input: File path, PIL Image, or numpy array.
            threshold: Decision boundary for REAL class (default 0.5).
            
        Returns:
            Dictionary with label, class_id, probabilities, confidence, and backend.
        """
        if not self.is_loaded():
            raise ModelUnavailableError(
                "No trained CIFAKE model is currently loaded. "
                "Please train or provide a model checkpoint before requesting predictions."
            )

        tgt_h, tgt_w = self.input_shape[0], self.input_shape[1]
        is_imagenet = (tgt_h == 224 and tgt_w == 224)

        # Standard preprocessing matched to model architecture
        preprocessed = preprocess_image(image_input, target_size=(tgt_h, tgt_w), apply_imagenet_norm=is_imagenet)
        batch = np.expand_dims(preprocessed, axis=0)

        # Raw sigmoid output: P(REAL) and 128-dim dense manifold embedding
        feature_emb = None
        if getattr(self, "feature_extractor", None) is not None:
            try:
                feat_t, pred_t = self.feature_extractor(batch, training=False)
                raw_pred = float(pred_t.numpy()[0][0])
                feature_emb = feat_t.numpy()[0]
            except Exception:
                pred_tensor = self.model(batch, training=False)
                raw_pred = float(pred_tensor.numpy()[0][0])
        else:
            pred_tensor = self.model(batch, training=False)
            raw_pred = float(pred_tensor.numpy()[0][0])
        
        # Clip to avoid division by zero or extreme rounding issues while preserving genuine probabilities
        real_prob = float(np.clip(raw_pred, 0.0001, 0.9999))
        ai_prob = float(1.0 - real_prob)

        is_real = real_prob >= threshold
        class_id = 1 if is_real else 0
        class_name = "REAL" if is_real else "AI"
        confidence = real_prob if is_real else ai_prob
        model_name = getattr(self.model, "name", "CIFAKE_Detector")

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
            "model_name": model_name,
            "checkpoint_path": self.model_path,
            "feature_embedding": feature_emb
        }


# Type alias for backward compatibility
CIFAKEInferenceService = ResNet50InferenceService


# Singleton service holder
_GLOBAL_SERVICE: Optional[ResNet50InferenceService] = None


def get_inference_service(model_path: Optional[str] = None, force_reload: bool = False) -> ResNet50InferenceService:
    """Retrieve or initialize the singleton ResNet50 inference service."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None or force_reload:
        _GLOBAL_SERVICE = ResNet50InferenceService(model_path=model_path)
    return _GLOBAL_SERVICE


def reset_inference_service():
    """Reset singleton inference service cache."""
    global _GLOBAL_SERVICE
    _GLOBAL_SERVICE = None
