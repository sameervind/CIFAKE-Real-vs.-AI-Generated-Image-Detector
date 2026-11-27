"""
Unit tests for FastAPI REST API endpoints.
"""

import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ResNet50" in data["model"]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert data["accuracy"] > 0.90


def test_samples_endpoint():
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert data["real_count"] > 0
    assert data["ai_count"] > 0


def test_predict_endpoint_with_image():
    # Create small test image buffer
    img = Image.new("RGB", (32, 32), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        params={"threshold": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["REAL", "AI-GENERATED"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "REAL" in data["probabilities"]
    assert "AI" in data["probabilities"]
    assert "explanation" in data
    assert data["processing_time_ms"] > 0


def test_predict_endpoint_invalid_file():
    buf = io.BytesIO(b"not an image file")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", buf, "text/plain")}
    )
    assert response.status_code == 400
