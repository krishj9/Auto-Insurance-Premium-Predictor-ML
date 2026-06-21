import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def generate_comparison_table(metrics_dir: str = "artifacts/metrics") -> pd.DataFrame:
    """Create and persist model comparison table from metrics JSON files."""
    resolved_metrics_dir = _resolve_path(metrics_dir)
    metric_files = sorted(resolved_metrics_dir.glob("*_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No metric files found in {resolved_metrics_dir}")

    rows: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "Model": metric_file.name.replace("_metrics.json", ""),
                "RMSE": float(payload["rmse"]),
                "RMSLE": float(payload["rmsle"]),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(by=["RMSLE", "RMSE"]).reset_index(drop=True)
    output_path = resolved_metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(comparison_df)
    return comparison_df


def generate_champion_plots(
    model: Any,
    preprocessor: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series | pd.DataFrame | list[float],
    y_pred: list[float] | pd.Series,
    artifacts_dir: str = "artifacts/plots",
) -> None:
    """Generate residual and predicted-vs-actual plots for champion evaluation."""
    _ = model
    _ = preprocessor
    _ = x_val
    resolved_artifacts_dir = _resolve_path(artifacts_dir)
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

    y_val_series = pd.Series(y_val, dtype=float)
    y_pred_series = pd.Series(y_pred, dtype=float)
    residuals = y_val_series - y_pred_series

    residuals_path = resolved_artifacts_dir / "residuals.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_series, residuals, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(residuals_path, dpi=150)
    plt.close()

    pred_vs_actual_path = resolved_artifacts_dir / "pred_vs_actual.png"
    min_value = float(min(y_val_series.min(), y_pred_series.min()))
    max_value = float(max(y_val_series.max(), y_pred_series.max()))
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val_series, y_pred_series, alpha=0.6)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(pred_vs_actual_path, dpi=150)
    plt.close()


def generate_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    artifacts_dir: str = "artifacts/plots",
    top_n: int = 20,
) -> None:
    """Generate feature importance plot for supported model types."""
    resolved_artifacts_dir = _resolve_path(artifacts_dir)
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_name_normalized = model_name.lower()
    if model_name_normalized == "linear_regression":
        if not hasattr(model, "coef_"):
            raise ValueError("Linear Regression model must be fitted and expose coef_.")
        importance_values = pd.Series(model.coef_, index=feature_names, dtype=float)
    else:
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model must expose feature_importances_.")
        importance_values = pd.Series(model.feature_importances_, index=feature_names, dtype=float)

    top_features = importance_values.abs().sort_values(ascending=False).head(top_n).index
    plot_values = importance_values.loc[top_features].sort_values(ascending=True)

    feature_importance_path = resolved_artifacts_dir / "feature_importance.png"
    plt.figure(figsize=(10, 8))
    plt.barh(plot_values.index, plot_values.values)
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(feature_importance_path, dpi=150)
    plt.close()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
