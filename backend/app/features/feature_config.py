from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_feature_lists(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists excluding the target."""
    target_column = _get_target_column(config=config)
    feature_columns = [column for column in df.columns if column != target_column]

    features_config = config.get("features")
    if isinstance(features_config, dict):
        configured_numeric = features_config.get("numeric_features")
        configured_categorical = features_config.get("categorical_features")
        if isinstance(configured_numeric, list) and isinstance(configured_categorical, list):
            numeric_features = [column for column in configured_numeric if column in feature_columns]
            categorical_features = [column for column in configured_categorical if column in feature_columns]
            return numeric_features, categorical_features

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for column in feature_columns:
        if is_numeric_dtype(df[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def engineer_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply lightweight feature derivations used by Stage 2 training."""
    engineered_df = df.copy()
    engineered_df = _derive_vehicle_age(engineered_df=engineered_df, config=config)
    engineered_df = _clean_engine_columns(engineered_df=engineered_df)
    return engineered_df


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Missing or invalid 'data.target_column' in config.")

    return target_column


def _derive_vehicle_age(engineered_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if "vehicle_age" in engineered_df.columns:
        return engineered_df

    year_column = _resolve_vehicle_year_column(engineered_df)
    if year_column is None:
        return engineered_df

    current_year = _resolve_current_year(config=config)
    engineered_df["vehicle_age"] = current_year - engineered_df[year_column]
    return engineered_df.drop(columns=[year_column])


def _resolve_vehicle_year_column(df: pd.DataFrame) -> str | None:
    known_year_columns = ("vehicle_year", "car_manufacturing_year", "Car Manufacturing Year")
    for year_column in known_year_columns:
        if year_column in df.columns:
            return year_column
    return None


def _resolve_current_year(config: dict[str, Any]) -> int:
    data_config = config.get("data")
    if isinstance(data_config, dict):
        configured_year = data_config.get("current_year")
        if isinstance(configured_year, int):
            return configured_year
    return datetime.now(timezone.utc).year


def _clean_engine_columns(engineered_df: pd.DataFrame) -> pd.DataFrame:
    for column in engineered_df.columns:
        is_engine_column = "engine" in column.lower()
        if not is_engine_column or is_numeric_dtype(engineered_df[column]):
            continue

        extracted_values = engineered_df[column].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        engineered_df[column] = pd.to_numeric(extracted_values, errors="coerce")

    return engineered_df
