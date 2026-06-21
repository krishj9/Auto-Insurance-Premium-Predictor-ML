from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str], config: dict[str, Any] | None = None
) -> ColumnTransformer:
    """Build an unfitted preprocessing ColumnTransformer."""
    use_standard_scaler = _should_use_standard_scaler(config=config)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_standard_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _should_use_standard_scaler(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False

    preprocessing_config = config.get("preprocessing")
    if not isinstance(preprocessing_config, dict):
        return False

    scale_numeric = preprocessing_config.get("scale_numeric")
    return bool(scale_numeric)
