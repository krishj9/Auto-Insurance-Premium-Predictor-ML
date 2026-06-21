import logging
from typing import Any

import pandas as pd


log = logging.getLogger(__name__)


def filter_outliers(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Filter target outliers using quantile bounds from config."""
    target_column = _get_target_column(config=config)
    outliers_config = _get_outliers_config(config=config)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataframe.")

    is_enabled = bool(outliers_config["enabled"])
    if not is_enabled:
        log.info("Outlier filtering disabled; returning %s rows unchanged", len(df))
        return df.copy()

    lower_quantile = float(outliers_config["lower_quantile"])
    upper_quantile = float(outliers_config["upper_quantile"])
    _validate_quantiles(lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    lower_bound = float(df[target_column].quantile(lower_quantile))
    upper_bound = float(df[target_column].quantile(upper_quantile))
    filtered_df = df[df[target_column].between(lower_bound, upper_bound, inclusive="both")].copy()

    removed_rows = len(df) - len(filtered_df)
    log.info(
        "Outlier filtering applied for %s: removed=%s remaining=%s lower_bound=%s upper_bound=%s",
        target_column,
        removed_rows,
        len(filtered_df),
        lower_bound,
        upper_bound,
    )
    return filtered_df


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Missing or invalid 'data.target_column' in config.")

    return target_column


def _get_outliers_config(config: dict[str, Any]) -> dict[str, Any]:
    outliers_config = config.get("outliers")
    if not isinstance(outliers_config, dict):
        raise ValueError("Missing or invalid 'outliers' section in config.")

    required_keys = ("enabled", "lower_quantile", "upper_quantile")
    for required_key in required_keys:
        if required_key not in outliers_config:
            raise ValueError(f"Missing required 'outliers.{required_key}' in config.")

    return outliers_config


def _validate_quantiles(lower_quantile: float, upper_quantile: float) -> None:
    if lower_quantile < 0.0 or upper_quantile > 1.0:
        raise ValueError("Outlier quantiles must be within [0.0, 1.0].")

    if lower_quantile >= upper_quantile:
        raise ValueError("Outlier lower_quantile must be less than upper_quantile.")
