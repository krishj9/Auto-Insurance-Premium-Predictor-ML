import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.core.config import load_config
from backend.app.data.loader import load_test, load_train
from backend.app.data.outlier_filter import filter_outliers
from backend.app.data.schema_validation import SchemaValidationError, validate_schema
from backend.app.evaluation.metrics import rmse, rmsle
from backend.app.features.feature_config import engineer_features, get_feature_lists
from backend.app.features.preprocessing import build_preprocessor
from backend.app.models.model_definitions import build_linear_regression, build_random_forest, build_xgboost


log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_training(config_path: str = "configs/config.yaml") -> None:
    """Run the Stage 2 training orchestration workflow end-to-end."""
    _configure_logging()
    resolved_config_path = _resolve_path(config_path)
    config = load_config(path=str(resolved_config_path))

    train_df = load_train(config=config)
    _run_optional_schema_validation(train_df=train_df, config=config)

    filtered_df = filter_outliers(df=train_df, config=config)
    engineered_df = engineer_features(df=filtered_df, config=config)

    target_column = _get_target_column(config=config)
    if target_column not in engineered_df.columns:
        raise ValueError(f"Target column '{target_column}' is missing after feature engineering.")

    x_data = engineered_df.drop(columns=[target_column])
    y_data = engineered_df[target_column].to_numpy(dtype=float)

    numeric_features, categorical_features = get_feature_lists(df=engineered_df, config=config)
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    split_config = _get_split_config(config=config)
    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=float(split_config["test_size"]),
        random_state=int(split_config["random_state"]),
    )
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)

    models_dir, metrics_dir = _prepare_artifact_directories(config=config)
    model_builders = {
        "linear_regression": build_linear_regression,
        "random_forest": lambda: build_random_forest(config=config),
        "xgboost": lambda: build_xgboost(config=config),
    }

    model_metrics: dict[str, dict[str, float]] = {}
    for model_name, model_builder in model_builders.items():
        model = model_builder()
        model.fit(x_train_processed, y_train)
        predictions = np.asarray(model.predict(x_val_processed), dtype=float)

        metrics_payload = {
            "rmse": rmse(y_true=y_val, y_pred=predictions),
            "rmsle": rmsle(y_true=y_val, y_pred=predictions),
        }
        model_metrics[model_name] = metrics_payload

        metrics_path = metrics_dir / f"{model_name}_metrics.json"
        model_path = models_dir / f"{model_name}.pkl"
        _write_json(metrics_path, metrics_payload)
        joblib.dump(model, model_path)
        log.info("Saved model=%s metrics=%s model_path=%s", model_name, metrics_payload, model_path)

    preprocessor_path = models_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    log.info("Saved preprocessor to %s", preprocessor_path)
    _log_comparison_table(model_metrics=model_metrics)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _run_optional_schema_validation(train_df: pd.DataFrame, config: dict[str, Any]) -> None:
    try:
        test_df = load_test(config=config)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Skipping optional schema validation; failed to load test data: %s", exc)
        return

    target_column = _get_target_column(config=config)
    test_for_validation = test_df.copy()
    if target_column in test_for_validation.columns:
        log.warning(
            "Test dataframe includes target '%s'; dropping it before optional schema validation.",
            target_column,
        )
        test_for_validation = test_for_validation.drop(columns=[target_column])

    try:
        validate_schema(train_df=train_df, test_df=test_for_validation, config=config)
    except SchemaValidationError as exc:
        log.warning("Optional schema validation failed: %s", exc)
    else:
        log.info("Optional schema validation passed.")


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Missing or invalid 'data.target_column' in config.")
    return target_column


def _get_split_config(config: dict[str, Any]) -> dict[str, Any]:
    split_config = config.get("split")
    if not isinstance(split_config, dict):
        raise ValueError("Missing or invalid 'split' section in config.")

    required_keys = ("test_size", "random_state")
    for key in required_keys:
        if key not in split_config:
            raise ValueError(f"Missing required 'split.{key}' in config.")
    return split_config


def _prepare_artifact_directories(config: dict[str, Any]) -> tuple[Path, Path]:
    artifacts_config = config.get("artifacts")
    if not isinstance(artifacts_config, dict):
        raise ValueError("Missing or invalid 'artifacts' section in config.")

    models_path_value = artifacts_config.get("models_path")
    metrics_path_value = artifacts_config.get("metrics_path")
    if not isinstance(models_path_value, str) or not isinstance(metrics_path_value, str):
        raise ValueError("Missing or invalid artifact paths in config.")

    models_dir = _resolve_path(models_path_value)
    metrics_dir = _resolve_path(metrics_path_value)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, metrics_dir


def _write_json(path: Path, payload: dict[str, float]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_comparison_table(model_metrics: dict[str, dict[str, float]]) -> None:
    log.info("Model comparison (lower is better):")
    for model_name, metrics_payload in sorted(model_metrics.items(), key=lambda item: item[1]["rmsle"]):
        log.info(
            "  model=%s | rmse=%.6f | rmsle=%.6f",
            model_name,
            metrics_payload["rmse"],
            metrics_payload["rmsle"],
        )


if __name__ == "__main__":
    run_training()
