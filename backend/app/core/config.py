from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load and validate the shared YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        parsed_config: Any = yaml.safe_load(config_file)

    if not isinstance(parsed_config, dict):
        raise ValueError(f"Configuration must be a mapping at the root: {config_path}")

    return parsed_config
