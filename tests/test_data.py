from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend.app.data.loader import load_test, load_train
from backend.app.data.outlier_filter import filter_outliers
from backend.app.data.schema_validation import SchemaValidationError, validate_schema


def test_load_train_and_load_test_return_expected_shapes(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    train_df = pd.DataFrame(
        [
            {"age": 30, "mileage": 12000, "Insurance Premium ($)": 500.0},
            {"age": 45, "mileage": 9000, "Insurance Premium ($)": 650.0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"age": 31, "mileage": 10000},
            {"age": 46, "mileage": 8000},
        ]
    )

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    config: dict[str, Any] = {
        "data": {
            "train_path": str(train_csv),
            "test_path": str(test_csv),
            "target_column": "Insurance Premium ($)",
        }
    }

    loaded_train = load_train(config=config)
    loaded_test = load_test(config=config)

    assert isinstance(loaded_train, pd.DataFrame)
    assert isinstance(loaded_test, pd.DataFrame)
    assert loaded_train.shape == (2, 3)
    assert loaded_test.shape == (2, 2)


def test_validate_schema_passes_on_valid_data() -> None:
    train_df = pd.DataFrame(
        {
            "age": [30, 45, 50],
            "segment": ["A", "B", "A"],
            "Insurance Premium ($)": [500.0, 650.0, 800.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [31, 46],
            "segment": ["A", "C"],
        }
    )
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)"}}

    report = validate_schema(train_df=train_df, test_df=test_df, config=config)

    assert report.target_column == "Insurance Premium ($)"
    assert report.train_rows == 3
    assert report.test_rows == 2
    assert set(report.feature_columns) == {"age", "segment"}
    assert "age" in report.profile["train"]
    assert "segment" in report.profile["train"]
    assert "mean" in report.profile["train"]["age"]
    assert "cardinality" in report.profile["train"]["segment"]


def test_validate_schema_raises_when_target_missing_in_train() -> None:
    train_df = pd.DataFrame({"age": [30, 45]})
    test_df = pd.DataFrame({"age": [31, 46]})
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)"}}

    with pytest.raises(SchemaValidationError):
        validate_schema(train_df=train_df, test_df=test_df, config=config)


def test_validate_schema_raises_when_target_present_in_test() -> None:
    train_df = pd.DataFrame({"age": [30, 45], "Insurance Premium ($)": [500.0, 650.0]})
    test_df = pd.DataFrame({"age": [31, 46], "Insurance Premium ($)": [510.0, 620.0]})
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)"}}

    with pytest.raises(SchemaValidationError):
        validate_schema(train_df=train_df, test_df=test_df, config=config)


def test_validate_schema_raises_on_feature_mismatch() -> None:
    train_df = pd.DataFrame(
        {
            "age": [30, 45],
            "mileage": [12000, 9000],
            "Insurance Premium ($)": [500.0, 650.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [31, 46],
            "vehicle_age": [5, 7],
        }
    )
    config: dict[str, Any] = {"data": {"target_column": "Insurance Premium ($)"}}

    with pytest.raises(SchemaValidationError):
        validate_schema(train_df=train_df, test_df=test_df, config=config)


def test_filter_outliers_removes_extreme_rows_and_keeps_in_range_rows() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "Insurance Premium ($)": [100.0, 105.0, 110.0, 115.0, 1000.0],
        }
    )
    config: dict[str, Any] = {
        "data": {"target_column": "Insurance Premium ($)"},
        "outliers": {"enabled": True, "lower_quantile": 0.0, "upper_quantile": 0.75},
    }

    filtered_df = filter_outliers(df=df, config=config)

    assert len(filtered_df) == 4
    assert 1000.0 not in filtered_df["Insurance Premium ($)"].tolist()
    assert filtered_df["Insurance Premium ($)"].max() <= 115.0
