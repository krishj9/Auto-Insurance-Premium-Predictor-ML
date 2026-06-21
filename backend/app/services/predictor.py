from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import load_config
from backend.app.features.feature_config import engineer_features
from backend.app.models.registry import load_champion


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUEST_TO_MODEL_COLUMNS = {
    "driver_age": "Driver Age",
    "driver_experience": "Driver Experience",
    "previous_accidents": "Previous Accidents",
    "annual_mileage_x1000_km": "Annual Mileage (x1000 km)",
    "car_manufacturing_year": "Car Manufacturing Year",
    "car_age": "Car Age",
}


class PredictionService:
    """Load champion artifacts and serve single-row premium predictions."""

    def __init__(self, models_dir: str = "artifacts/models/") -> None:
        self._models_dir = models_dir
        self._config = load_config(path=str(PROJECT_ROOT / "configs/config.yaml"))
        self._model: Any | None = None
        self._preprocessor: Any | None = None
        self._metadata: dict[str, Any] | None = None

    def load(self) -> None:
        model, preprocessor, metadata = load_champion(models_dir=self._models_dir)
        self._model = model
        self._preprocessor = preprocessor
        self._metadata = metadata

    def predict(self, features: dict[str, Any]) -> float:
        if self._model is None or self._preprocessor is None:
            raise ValueError("PredictionService is not loaded. Call load() before predict().")
        if not isinstance(features, dict):
            raise ValueError("features must be a dictionary.")

        missing_keys = [key for key in REQUEST_TO_MODEL_COLUMNS if key not in features]
        if missing_keys:
            raise ValueError(f"Missing required features: {missing_keys}")

        model_row = {
            model_column: features[request_field]
            for request_field, model_column in REQUEST_TO_MODEL_COLUMNS.items()
        }
        input_df = pd.DataFrame([model_row])
        engineered_df = engineer_features(df=input_df, config=self._config)

        transformed_row = self._preprocessor.transform(engineered_df)
        prediction_array = np.asarray(self._model.predict(transformed_row), dtype=float).ravel()
        if prediction_array.size != 1:
            raise ValueError("Expected a single prediction output.")
        return float(prediction_array[0])

    def get_model_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise ValueError("PredictionService is not loaded. Call load() before reading metadata.")
        return self._metadata
