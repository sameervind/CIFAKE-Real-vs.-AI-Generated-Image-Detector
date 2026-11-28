import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.dataset import preprocess_image, TARGET_SIZE

def generate_calibration():
    print("Generating OOD calibration parameters...", flush=True)
    model = tf.keras.models.load_model("model/resnet50_cifake.keras")
    dense_layer = model.get_layer("dense_feature")
    feature_model = tf.keras.Model(inputs=model.input, outputs=dense_layer.output)

    real_paths = sorted(list(Path("samples/real").glob("*.jpg")))
    ai_paths = sorted(list(Path("samples/ai").glob("*.jpg")))

    real_batch = np.array([preprocess_image(str(p), target_size=TARGET_SIZE, apply_imagenet_norm=True) for p in real_paths], dtype=np.float32)
    ai_batch = np.array([preprocess_image(str(p), target_size=TARGET_SIZE, apply_imagenet_norm=True) for p in ai_paths], dtype=np.float32)

    real_embs = feature_model.predict(real_batch, batch_size=len(real_batch), verbose=0)
    ai_embs = feature_model.predict(ai_batch, batch_size=len(ai_batch), verbose=0)
    all_embs = np.vstack([real_embs, ai_embs])

    mu_real = np.mean(real_embs, axis=0)
    mu_ai = np.mean(ai_embs, axis=0)
    mu_all = np.mean(all_embs, axis=0)

    std_all = np.std(all_embs, axis=0) + 1e-6

    # In-distribution distance distribution
    dists_to_all = [float(np.linalg.norm(e - mu_all)) for e in all_embs]
    dists_to_class = [float(np.linalg.norm(e - mu_real)) for e in real_embs] + [float(np.linalg.norm(e - mu_ai)) for e in ai_embs]

    norms = [float(np.linalg.norm(e)) for e in all_embs]

    calibration_data = {
        "dataset_name": "CIFAKE",
        "benchmark_resolution": [32, 32],
        "feature_dim": int(all_embs.shape[1]),
        "centroid_real": [round(float(v), 5) for v in mu_real],
        "centroid_ai": [round(float(v), 5) for v in mu_ai],
        "centroid_overall": [round(float(v), 5) for v in mu_all],
        "feature_std": [round(float(v), 5) for v in std_all],
        "id_distance_mean": round(float(np.mean(dists_to_all)), 4),
        "id_distance_std": round(float(np.std(dists_to_all)), 4),
        "id_distance_max": round(float(np.max(dists_to_all)), 4),
        "id_distance_p95": round(float(np.percentile(dists_to_all, 95)), 4),
        "id_class_distance_mean": round(float(np.mean(dists_to_class)), 4),
        "id_class_distance_max": round(float(np.max(dists_to_class)), 4),
        "id_norm_mean": round(float(np.mean(norms)), 4),
        "id_norm_max": round(float(np.max(norms)), 4),
        "temperature": 1.0,
        "energy_threshold": -0.05
    }

    out_path = Path("assets/ood_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"OOD calibration saved to {out_path}!", flush=True)

if __name__ == "__main__":
    generate_calibration()
