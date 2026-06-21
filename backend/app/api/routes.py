from typing import Any

from fastapi import APIRouter, Depends

from backend.app.schemas.requests import PredictRequest
from backend.app.schemas.responses import (
    FeatureField,
    FeatureSchemaResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)
from backend.app.services.predictor import PredictionService


router = APIRouter()
_prediction_service: PredictionService | None = None


class PredictionError(Exception):
    """Domain-level prediction exception handled globally by FastAPI."""

    def __init__(self, message: str, status_code: int = 400, code: str = "prediction_error") -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


def set_prediction_service(service: PredictionService | None) -> None:
    global _prediction_service
    _prediction_service = service


def get_prediction_service() -> PredictionService:
    if _prediction_service is None:
        raise PredictionError(
            message="Prediction service is not initialised.",
            status_code=503,
            code="service_unavailable",
        )
    return _prediction_service


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/feature-schema", response_model=FeatureSchemaResponse)
async def get_feature_schema() -> FeatureSchemaResponse:
    return FeatureSchemaResponse(features=_build_feature_fields())


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info(service: PredictionService = Depends(get_prediction_service)) -> ModelInfoResponse:
    metadata = service.get_model_metadata()
    champion_model = str(metadata.get("model_name", "unknown"))
    trained_at = str(metadata.get("saved_at_utc", "unknown"))
    metrics = {
        "rmse": float(metadata.get("rmse", 0.0)),
        "rmsle": float(metadata.get("rmsle", 0.0)),
    }
    return ModelInfoResponse(champion_model=champion_model, metrics=metrics, trained_at=trained_at)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest, service: PredictionService = Depends(get_prediction_service)
) -> PredictResponse:
    try:
        predicted_premium = service.predict(request.model_dump())
    except PredictionError:
        raise
    except ValueError as exc:
        raise PredictionError(
            message=str(exc),
            status_code=400,
            code="prediction_input_error",
        ) from exc
    except Exception as exc:
        raise PredictionError(
            message="Prediction failed due to an internal error.",
            status_code=500,
            code="prediction_failure",
        ) from exc

    metadata = service.get_model_metadata()
    model_name = str(metadata.get("model_name", "unknown"))
    model_version = str(metadata.get("saved_at_utc", "unknown"))
    return PredictResponse(
        predicted_premium=float(predicted_premium),
        model_name=model_name,
        model_version=model_version,
    )


def _build_feature_fields() -> list[FeatureField]:
    examples: dict[str, Any] = {
        "driver_age": 35,
        "driver_experience": 12,
        "previous_accidents": 1,
        "annual_mileage_x1000_km": 20,
        "car_manufacturing_year": 2016,
        "car_age": 9,
    }
    feature_fields: list[FeatureField] = []
    for field_name, field_info in PredictRequest.model_fields.items():
        annotation = field_info.annotation
        field_type = getattr(annotation, "__name__", str(annotation))
        feature_fields.append(
            FeatureField(
                name=field_name,
                type=field_type,
                description=field_info.description or "",
                example=examples.get(field_name),
            )
        )
    return feature_fields
