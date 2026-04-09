import pytest
from src.config import CLASS_NAMES
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_class_names_configured():
    assert 'COVID' in CLASS_NAMES
    assert 'NEUMONIA' in CLASS_NAMES
    assert 'NORMALL' in CLASS_NAMES
    assert len(CLASS_NAMES) == 3

def test_api_metrics_endpoint_exists():
    response = client.get("/metrics")
    # Even if MLFlow isn't hydrated, it should return 200 with an empty list
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_validation():
    # Should reject files that aren't images
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"hola mundo", "text/plain")}
    )
    assert response.status_code == 400
    assert "no es un formato de imagen" in response.json()["detail"]
