"""
ResNet50 Model Architecture for AI-Generated vs Real Image Classification
Based on the CIFAKE dataset (CIFAR-10 real vs Stable Diffusion synthetic).
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_resnet50_model(input_shape: Tuple[int, int, int] = (32, 32, 3),
                         learning_rate: float = 0.001) -> Any:
    """
    Builds the exact ResNet50 transfer learning model architecture
    from the original project notebook.

    Architecture:
    1. Pre-trained ResNet50 base (weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    2. Flatten layer
    3. Dense(256, activation='relu')
    4. BatchNormalization()
    5. Dropout(0.5)
    6. Dense(128, activation='relu')
    7. Dense(1, activation='sigmoid')  # Binary: 0 = AI, 1 = REAL
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the Keras ResNet50 model.")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Do not freeze base layers (allow fine-tuning on CIFAKE dataset)
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='dense_feature_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.5, name='dropout_1'),
        layers.Dense(128, activation='relu', name='dense_feature_2'),
        layers.Dense(1, activation='sigmoid', name='output_classifier')
    ], name="ai_image_classifier_resnet50")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_lr_scheduler():
    """Learning rate scheduler that decays learning rate by 5% every epoch."""
    if not TF_AVAILABLE:
        return None
    def lr_scheduler(epoch, lr):
        return lr * 0.95
    return LearningRateScheduler(lr_scheduler)


def get_early_stopping(patience: int = 3):
    """Early stopping callback with restore_best_weights."""
    if not TF_AVAILABLE:
        return None
    return EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)


class ResNet50Detector:
    """
    Production detector wrapper supporting both full TensorFlow Keras ResNet50
    and high-performance lightweight Fourier/Spatial statistical inference.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.backend = None
        self.model_path = model_path
        self._init_model(model_path)

    def _init_model(self, model_path: Optional[str]):
        if TF_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.backend = "tensorflow_keras"
                print(f"[Detector] Loaded TensorFlow Keras model from {model_path}")
                return
            except Exception as e:
                print(f"[Detector] Warning: Could not load Keras model ({e}). Using native engine.")

        # Native feature & spectral analyzer backend
        from .frequency_analysis import FrequencyArtifactDetector
        self.fallback_engine = FrequencyArtifactDetector()
        self.backend = "spectral_feature_engine"
        print("[Detector] Initialized high-fidelity Spectral & Spatial Detection Engine.")

    def predict(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Predict whether an image is REAL (1) or AI-GENERATED (0).
        
        Args:
            image_np: RGB numpy array of shape (32, 32, 3) or (H, W, 3), uint8 or float [0, 1]

        Returns:
            dict containing:
                - class_name: 'REAL' or 'AI'
                - class_id: 1 for REAL, 0 for AI
                - real_probability: float [0, 1]
                - ai_probability: float [0, 1]
                - confidence: float [0.5, 1.0]
                - backend: str
        """
        if self.backend == "tensorflow_keras" and self.model is not None:
            # Preprocess image
            if image_np.dtype == np.uint8:
                img_scaled = image_np.astype(np.float32) / 255.0
            else:
                img_scaled = image_np.astype(np.float32)
                if img_scaled.max() > 1.0:
                    img_scaled = img_scaled / 255.0

            if img_scaled.shape[:2] != (32, 32):
                import cv2
                img_scaled = cv2.resize(img_scaled, (32, 32))

            batch = np.expand_dims(img_scaled, axis=0)
            y_pred = float(self.model.predict(batch, verbose=0)[0][0])
            
            real_prob = y_pred
            ai_prob = 1.0 - real_prob
            class_id = 1 if real_prob > 0.5 else 0
            class_name = 'REAL' if class_id == 1 else 'AI'
            confidence = max(real_prob, ai_prob)

            return {
                "class_name": class_name,
                "class_id": class_id,
                "real_probability": real_prob,
                "ai_probability": ai_prob,
                "confidence": confidence,
                "backend": "tensorflow_keras",
                "raw_score": y_pred
            }
        else:
            # Native spectral/feature detector
            result = self.fallback_engine.detect(image_np)
            result["backend"] = "spectral_feature_engine"
            return result

    def predict_proba(self, images: np.ndarray):
        """
        Return probability of class 1 (REAL) for a single image or batch.
        
        Args:
            images: shape (32, 32, 3) or (N, 32, 32, 3)
            
        Returns:
            float (single image) or np.ndarray (batch)
        """
        if images.ndim == 3:
            res = self.predict(images)
            return float(res["real_probability"])
        elif images.ndim == 4:
            probs = [float(self.predict(img)["real_probability"]) for img in images]
            return np.array(probs, dtype=np.float32)
        else:
            raise ValueError(f"Expected 3D or 4D array, got shape {images.shape}")

