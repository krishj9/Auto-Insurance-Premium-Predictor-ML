from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


class SchemaValidationError(ValueError):
    """Raised when train/test dataframes fail schema validation checks."""


@dataclass(frozen=True)
class SchemaReport:
    target_column: str
    train_rows: int
    test_rows: int
    train_columns: list[str]
    test_columns: list[str]
    feature_columns: list[str]
    profile: dict[str, dict[str, Any]]


def validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict[str, Any]) -> SchemaReport:
    """Validate train/test schema expectations and return a profile report."""
    target_column = _get_target_column(config=config)
    _validate_target_presence(train_df=train_df, test_df=test_df, target_column=target_column)
    feature_columns = _validate_feature_compatibility(
        train_columns=train_df.columns.tolist(),
        test_columns=test_df.columns.tolist(),
        target_column=target_column,
    )

    profile = _build_profile(train_df=train_df, test_df=test_df)
    return SchemaReport(
        target_column=target_column,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_columns=train_df.columns.tolist(),
        test_columns=test_df.columns.tolist(),
        feature_columns=feature_columns,
        profile=profile,
    )


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise SchemaValidationError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise SchemaValidationError("Missing or invalid 'data.target_column' in config.")

    return target_column


def _validate_target_presence(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> None:
    if target_column not in train_df.columns:
        raise SchemaValidationError(f"Target column '{target_column}' is missing from train dataframe.")

    if target_column in test_df.columns:
        raise SchemaValidationError(f"Target column '{target_column}' must be absent from test dataframe.")


def _validate_feature_compatibility(train_columns: list[str], test_columns: list[str], target_column: str) -> list[str]:
    train_features = [column for column in train_columns if column != target_column]
    train_feature_set = set(train_features)
    test_feature_set = set(test_columns)

    if train_feature_set != test_feature_set:
        missing_in_test = sorted(train_feature_set - test_feature_set)
        unexpected_in_test = sorted(test_feature_set - train_feature_set)
        raise SchemaValidationError(
            "Feature mismatch between train and test dataframes. "
            f"Missing in test: {missing_in_test}. Unexpected in test: {unexpected_in_test}."
        )

    return sorted(train_feature_set)


def _build_profile(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    train_profile = _profile_dataframe(df=train_df)
    test_profile = _profile_dataframe(df=test_df)
    return {"train": train_profile, "test": test_profile}


def _profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    profile: dict[str, Any] = {}

    for column_name in df.columns:
        series = df[column_name]
        null_count = int(series.isna().sum())
        null_percentage = float((null_count / row_count) * 100) if row_count > 0 else 0.0
        column_profile: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_percentage": null_percentage,
        }

        if is_numeric_dtype(series):
            numeric_series = series.dropna()
            if numeric_series.empty:
                column_profile.update(
                    {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "median": None,
                        "std": None,
                    }
                )
            else:
                column_profile.update(
                    {
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean": float(numeric_series.mean()),
                        "median": float(numeric_series.median()),
                        "std": float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0,
                    }
                )
        else:
            column_profile["cardinality"] = int(series.nunique(dropna=True))

        profile[column_name] = column_profile

    return profile
