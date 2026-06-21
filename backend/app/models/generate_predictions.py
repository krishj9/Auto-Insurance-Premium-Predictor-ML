from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import load_config
from backend.app.data.loader import load_test
from backend.app.features.feature_config import engineer_features
from backend.app.models.registry import load_champion


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def generate_test_predictions(config_path: str = "configs/config.yaml") -> Path:
    """Generate prediction CSV for test dataset using champion artifacts."""
    resolved_config_path = _resolve_path(config_path)
    config = load_config(path=str(resolved_config_path))
    test_df = load_test(config=config)

    champion_model, champion_preprocessor, _metadata = load_champion(models_dir="artifacts/models/")
    engineered_test_df = engineer_features(df=test_df, config=config)

    target_column = _get_target_column(config)
    if target_column in engineered_test_df.columns:
        engineered_test_df = engineered_test_df.drop(columns=[target_column])

    transformed_test = champion_preprocessor.transform(engineered_test_df)
    predictions = champion_model.predict(transformed_test)

    if len(test_df.columns) > 0:
        id_values = test_df.iloc[:, 0]
    else:
        id_values = pd.Series(test_df.index, name="ID")

    output_df = pd.DataFrame(
        {
            "ID": id_values.values,
            "PredictedPremium": predictions,
        }
    )

    submissions_dir = _resolve_submissions_dir(config=config)
    output_path = submissions_dir / "predictions.csv"
    output_df.to_csv(output_path, index=False)
    return output_path


def _resolve_submissions_dir(config: dict[str, Any]) -> Path:
    artifacts_config = config.get("artifacts")
    if not isinstance(artifacts_config, dict):
        raise ValueError("Missing or invalid 'artifacts' section in config.")

    submissions_path = artifacts_config.get("submissions_path")
    if not isinstance(submissions_path, str) or not submissions_path.strip():
        raise ValueError("Missing or invalid 'artifacts.submissions_path' in config.")

    resolved_submissions_path = _resolve_path(submissions_path)
    resolved_submissions_path.mkdir(parents=True, exist_ok=True)
    return resolved_submissions_path


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Missing or invalid 'data.target_column' in config.")
    return target_column


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    generated_path = generate_test_predictions()
    print(f"Saved predictions to {generated_path}")
