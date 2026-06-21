import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def select_champion(metrics_dir: str = "artifacts/metrics/") -> dict[str, Any]:
    """Select champion by RMSLE (then RMSE) from metrics JSON files."""
    resolved_metrics_dir = _resolve_path(metrics_dir)
    metric_files = sorted(resolved_metrics_dir.glob("*_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No metric files found in {resolved_metrics_dir}")

    candidates: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text(encoding="utf-8"))
        model_name = metric_file.name.replace("_metrics.json", "")
        candidates.append(
            {
                "model_name": model_name,
                "rmse": float(payload["rmse"]),
                "rmsle": float(payload["rmsle"]),
                "metrics_file": str(metric_file),
            }
        )

    candidates.sort(key=lambda candidate: (candidate["rmsle"], candidate["rmse"]))
    return candidates[0]


def save_champion(champion_name: str, metrics: dict[str, Any], models_dir: str = "artifacts/models/") -> dict[str, Any]:
    """Persist champion model, preprocessor, and metadata artifacts."""
    resolved_models_dir = _resolve_path(models_dir)
    resolved_models_dir.mkdir(parents=True, exist_ok=True)

    champion_source_path = resolved_models_dir / f"{champion_name}.pkl"
    preprocessor_source_path = resolved_models_dir / "preprocessor.pkl"
    if not champion_source_path.exists():
        raise FileNotFoundError(f"Champion model file missing: {champion_source_path}")
    if not preprocessor_source_path.exists():
        raise FileNotFoundError(f"Preprocessor file missing: {preprocessor_source_path}")

    champion_target_path = resolved_models_dir / "champion_model.pkl"
    champion_preprocessor_target_path = resolved_models_dir / "champion_preprocessor.pkl"
    metadata_path = resolved_models_dir / "champion_metadata.json"

    champion_model = joblib.load(champion_source_path)
    champion_preprocessor = joblib.load(preprocessor_source_path)
    joblib.dump(champion_model, champion_target_path)
    joblib.dump(champion_preprocessor, champion_preprocessor_target_path)

    metadata: dict[str, Any] = {
        "model_name": champion_name,
        "rmse": float(metrics["rmse"]),
        "rmsle": float(metrics["rmsle"]),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def load_champion(models_dir: str = "artifacts/models/") -> tuple[Any, Any, dict[str, Any]]:
    """Load champion model, champion preprocessor, and champion metadata."""
    resolved_models_dir = _resolve_path(models_dir)
    champion_model_path = resolved_models_dir / "champion_model.pkl"
    champion_preprocessor_path = resolved_models_dir / "champion_preprocessor.pkl"
    metadata_path = resolved_models_dir / "champion_metadata.json"

    if not champion_model_path.exists():
        raise FileNotFoundError(f"Champion model file missing: {champion_model_path}")
    if not champion_preprocessor_path.exists():
        raise FileNotFoundError(f"Champion preprocessor file missing: {champion_preprocessor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Champion metadata file missing: {metadata_path}")

    model = joblib.load(champion_model_path)
    preprocessor = joblib.load(champion_preprocessor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, preprocessor, metadata


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
