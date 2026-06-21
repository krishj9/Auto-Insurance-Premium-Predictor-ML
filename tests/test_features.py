from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from backend.app.features.feature_config import engineer_features, get_feature_lists
from backend.app.features.preprocessing import build_preprocessor


def test_engineer_features_derives_vehicle_age_and_drops_vehicle_year() -> None:
    source_df = pd.DataFrame(
        {
            "vehicle_year": [2020, 2018],
            "engine_cc": ["1498 cc", "1197 cc"],
            "segment": ["A", "B"],
            "Insurance Premium ($)": [500.0, 650.0],
        }
    )
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)", "current_year": 2026}}

    engineered_df = engineer_features(df=source_df, config=config)

    assert "vehicle_year" not in engineered_df.columns
    assert "vehicle_age" in engineered_df.columns
    assert engineered_df["vehicle_age"].tolist() == [6, 8]
    assert engineered_df["engine_cc"].tolist() == [1498.0, 1197.0]


def test_get_feature_lists_infers_numeric_and_categorical_features() -> None:
    df = pd.DataFrame(
        {
            "vehicle_age": [6, 8],
            "engine_cc": [1498.0, 1197.0],
            "segment": ["A", "B"],
            "Insurance Premium ($)": [500.0, 650.0],
        }
    )
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)"}}

    numeric_features, categorical_features = get_feature_lists(df=df, config=config)

    assert set(numeric_features) == {"vehicle_age", "engine_cc"}
    assert categorical_features == ["segment"]


def test_get_feature_lists_uses_configured_feature_lists_when_present() -> None:
    df = pd.DataFrame(
        {
            "vehicle_age": [6, 8],
            "engine_cc": [1498.0, 1197.0],
            "segment": ["A", "B"],
            "Insurance Premium ($)": [500.0, 650.0],
        }
    )
    config: dict[str, Any] = {
        "data": {"target_column": "Insurance Premium ($)"},
        "features": {
            "numeric_features": ["engine_cc"],
            "categorical_features": ["segment", "missing_column"],
        },
    }

    numeric_features, categorical_features = get_feature_lists(df=df, config=config)

    assert numeric_features == ["engine_cc"]
    assert categorical_features == ["segment"]


def test_build_preprocessor_returns_unfitted_column_transformer() -> None:
    preprocessor = build_preprocessor(
        numeric_features=["vehicle_age", "engine_cc"],
        categorical_features=["segment"],
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")


def test_build_preprocessor_applies_imputer_encoder_and_optional_scaler() -> None:
    df = pd.DataFrame(
        {
            "vehicle_age": [6.0, None, 10.0],
            "engine_cc": [1498.0, 1197.0, None],
            "segment": ["A", "B", np.nan],
        }
    )
    config: dict[str, Any] = {"preprocessing": {"scale_numeric": True}}
    preprocessor = build_preprocessor(
        numeric_features=["vehicle_age", "engine_cc"],
        categorical_features=["segment"],
        config=config,
    )

    transformed = preprocessor.fit_transform(df)
    num_pipeline = preprocessor.named_transformers_["num"]
    assert num_pipeline.named_steps["imputer"].strategy == "median"
    assert isinstance(num_pipeline.named_steps["scaler"], StandardScaler)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 3
