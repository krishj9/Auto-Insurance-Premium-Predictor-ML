import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_config
from backend.app.data.loader import load_test, load_train
from backend.app.data.outlier_filter import filter_outliers
from backend.app.data.schema_validation import SchemaReport, validate_schema


log = logging.getLogger(__name__)


def run_eda(config_path: str = str(PROJECT_ROOT / "configs/config.yaml")) -> None:
    """Run Stage 1 EDA workflow and save required plots."""
    _configure_logging()
    config = load_config(config_path)
    target_column = _get_target_column(config)
    plots_dir = _resolve_plots_dir(config)

    train_df = load_train(config)
    test_df = load_test(config)
    schema_validation_test_df = test_df.copy()
    if target_column in schema_validation_test_df.columns:
        log.warning(
            "Test dataframe includes target column '%s'; dropping it for schema validation.",
            target_column,
        )
        schema_validation_test_df = schema_validation_test_df.drop(columns=[target_column])

    schema_report = validate_schema(train_df=train_df, test_df=schema_validation_test_df, config=config)

    _log_schema_profile(schema_report)
    _save_raw_distribution_plot(train_df=train_df, target_column=target_column, plots_dir=plots_dir)
    _save_log_distribution_plot(train_df=train_df, target_column=target_column, plots_dir=plots_dir)
    _save_sorted_plot(train_df=train_df, target_column=target_column, plots_dir=plots_dir)

    filtered_df = filter_outliers(df=train_df, config=config)
    _save_post_filter_distribution_plot(filtered_df=filtered_df, target_column=target_column, plots_dir=plots_dir)
    _log_outlier_policy(config=config, original_rows=len(train_df), filtered_rows=len(filtered_df))

    log.info("EDA artifacts generated under %s", plots_dir)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sns.set_theme(style="whitegrid")


def _get_target_column(config: dict[str, Any]) -> str:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing or invalid 'data' section in config.")

    target_column = data_config.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Missing or invalid 'data.target_column' in config.")

    return target_column


def _resolve_plots_dir(config: dict[str, Any]) -> Path:
    artifacts_config = config.get("artifacts")
    if not isinstance(artifacts_config, dict):
        raise ValueError("Missing or invalid 'artifacts' section in config.")

    plots_path = artifacts_config.get("plots_path")
    if not isinstance(plots_path, str) or not plots_path.strip():
        raise ValueError("Missing or invalid 'artifacts.plots_path' in config.")

    plots_dir = Path(plots_path)
    if not plots_dir.is_absolute():
        plots_dir = PROJECT_ROOT / plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _log_schema_profile(schema_report: SchemaReport) -> None:
    log.info(
        "Schema validated: train_rows=%s test_rows=%s feature_count=%s",
        schema_report.train_rows,
        schema_report.test_rows,
        len(schema_report.feature_columns),
    )
    print("Schema profile summary:")
    for dataset_name in ("train", "test"):
        dataset_profile = schema_report.profile[dataset_name]
        profile_frame = pd.DataFrame.from_dict(dataset_profile, orient="index")
        print(f"\n[{dataset_name}]")
        print(profile_frame.fillna(""))


def _save_raw_distribution_plot(train_df: pd.DataFrame, target_column: str, plots_dir: Path) -> None:
    figure_path = plots_dir / "premium_distribution.png"
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df[target_column], kde=True, bins=30)
    plt.title("Premium Distribution")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    log.info("Saved plot: %s", figure_path)


def _save_log_distribution_plot(train_df: pd.DataFrame, target_column: str, plots_dir: Path) -> None:
    figure_path = plots_dir / "premium_log_distribution.png"
    log_series = np.log1p(train_df[target_column])
    plt.figure(figsize=(10, 6))
    sns.histplot(log_series, kde=True, bins=30)
    plt.title("Log-Transformed Premium Distribution")
    plt.xlabel(f"log1p({target_column})")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    log.info("Saved plot: %s", figure_path)


def _save_sorted_plot(train_df: pd.DataFrame, target_column: str, plots_dir: Path) -> None:
    figure_path = plots_dir / "premium_sorted.png"
    sorted_values = train_df[target_column].sort_values().reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_values.index, sorted_values.values)
    plt.title("Sorted Premium Values")
    plt.xlabel("Sorted Index")
    plt.ylabel(target_column)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    log.info("Saved plot: %s", figure_path)


def _save_post_filter_distribution_plot(filtered_df: pd.DataFrame, target_column: str, plots_dir: Path) -> None:
    figure_path = plots_dir / "premium_distribution_post_filter.png"
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df[target_column], kde=True, bins=30)
    plt.title("Premium Distribution (Post Outlier Filter)")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    log.info("Saved plot: %s", figure_path)


def _log_outlier_policy(config: dict[str, Any], original_rows: int, filtered_rows: int) -> None:
    outliers_config = config.get("outliers")
    if not isinstance(outliers_config, dict):
        raise ValueError("Missing or invalid 'outliers' section in config.")

    enabled = bool(outliers_config.get("enabled"))
    lower_quantile = outliers_config.get("lower_quantile")
    upper_quantile = outliers_config.get("upper_quantile")
    removed_rows = original_rows - filtered_rows
    log.info(
        "Outlier policy: enabled=%s lower_quantile=%s upper_quantile=%s removed_rows=%s remaining_rows=%s",
        enabled,
        lower_quantile,
        upper_quantile,
        removed_rows,
        filtered_rows,
    )
    print(
        "Outlier policy:"
        f" enabled={enabled}, lower_quantile={lower_quantile}, upper_quantile={upper_quantile},"
        f" removed_rows={removed_rows}, remaining_rows={filtered_rows}"
    )


if __name__ == "__main__":
    run_eda()
