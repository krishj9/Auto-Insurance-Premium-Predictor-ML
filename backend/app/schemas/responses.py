from typing import Any

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    predicted_premium: float
    model_name: str
    model_version: str
    disclaimer: str = "Demo prediction only; this is not an actual insurance quote."


class FeatureField(BaseModel):
    name: str
    type: str
    description: str
    example: Any


class FeatureSchemaResponse(BaseModel):
    features: list[FeatureField] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    champion_model: str
    metrics: dict[str, float]
    trained_at: str


class HealthResponse(BaseModel):
    status: str
