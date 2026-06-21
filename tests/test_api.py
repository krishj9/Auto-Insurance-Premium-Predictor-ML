from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import get_prediction_service
from backend.app.main import app


class FakePredictionService:
    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "random_forest",
            "rmse": 12.34,
            "rmsle": 0.12,
            "saved_at_utc": "2026-06-21T10:00:00+00:00",
        }

    def predict(self, features: dict[str, Any]) -> float:
        _ = features
        return 512.34


class FailingPredictionService(FakePredictionService):
    def predict(self, features: dict[str, Any]) -> float:
        _ = features
        raise ValueError("Invalid payload values for prediction.")


@pytest.fixture
def api_client() -> TestClient:
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feature_schema_endpoint_returns_non_empty_features(api_client: TestClient) -> None:
    response = api_client.get("/feature-schema")
    assert response.status_code == 200
    payload = response.json()
    assert "features" in payload
    assert len(payload["features"]) > 0


def test_models_endpoint_uses_dependency_injection(api_client: TestClient) -> None:
    response = api_client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["champion_model"] == "random_forest"
    assert payload["metrics"]["rmse"] == pytest.approx(12.34)


def test_predict_endpoint_returns_prediction_for_valid_payload(api_client: TestClient) -> None:
    payload = {
        "driver_age": 40,
        "driver_experience": 20,
        "previous_accidents": 1,
        "annual_mileage_x1000_km": 18,
        "car_manufacturing_year": 2017,
        "car_age": 8,
    }

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_premium"] > 0
    assert body["model_name"] == "random_forest"


def test_predict_endpoint_returns_422_for_invalid_payload(api_client: TestClient) -> None:
    invalid_payload = {
        "driver_age": 14,
        "driver_experience": 20,
    }

    response = api_client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_prediction_error_handler_returns_structured_error() -> None:
    app.dependency_overrides[get_prediction_service] = lambda: FailingPredictionService()
    with TestClient(app) as client:
        payload = {
            "driver_age": 40,
            "driver_experience": 20,
            "previous_accidents": 1,
            "annual_mileage_x1000_km": 18,
            "car_manufacturing_year": 2017,
            "car_age": 8,
        }
        response = client.post("/predict", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 400
    error_payload = response.json()
    assert error_payload["error"]["code"] == "prediction_input_error"
    assert "Invalid payload values" in error_payload["error"]["message"]
