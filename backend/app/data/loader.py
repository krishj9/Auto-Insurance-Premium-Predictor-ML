import logging
from pathlib import Path
from typing import Any

import pandas as pd


log = logging.getLogger(__name__)


def load_train(config: dict[str, Any]) -> pd.DataFrame:
    """Load the training CSV configured under data.train_path."""
    train_path = _resolve_data_path(config=config, key="train_path")
    train_df = pd.read_csv(train_path)
    log.info("Loaded train data with %s rows and %s columns", train_df.shape[0], train_df.shape[1])
    return train_df


def load_test(config: dict[str, Any]) -> pd.DataFrame:
    """Load the test CSV configured under data.test_path."""
    test_path = _resolve_data_path(config=config, key="test_path")
    test_df = pd.read_csv(test_path)
    log.info("Loaded test data with %s rows and %s columns", test_df.shape[0], test_df.shape[1])
    return test_df


def _resolve_data_path(config: dict[str, Any], key: str) -> Path:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    raw_path = data_config.get(key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Missing or invalid 'data.{key}' in config.")

    csv_path = Path(raw_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Configured CSV path does not exist: {csv_path}")

    return csv_path
