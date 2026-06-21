from typing import Any
from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from xgboost import XGBRegressor


def build_linear_regression() -> LinearRegression:
    """Return an unfitted LinearRegression model."""
    return LinearRegression()


def build_random_forest(config: dict[str, Any]) -> RandomForestRegressor:
    """Return an unfitted RandomForestRegressor configured from config."""
    random_forest_config = _get_model_config(config=config, model_name="random_forest")
    return RandomForestRegressor(
        n_estimators=int(random_forest_config["n_estimators"]),
        max_depth=int(random_forest_config["max_depth"]),
        random_state=int(random_forest_config["random_state"]),
    )


def build_xgboost(config: dict[str, Any]) -> "XGBRegressor":
    """Return an unfitted XGBRegressor configured from config."""
    from xgboost import XGBRegressor

    xgboost_config = _get_model_config(config=config, model_name="xgboost")
    return XGBRegressor(
        n_estimators=int(xgboost_config["n_estimators"]),
        max_depth=int(xgboost_config["max_depth"]),
        learning_rate=float(xgboost_config["learning_rate"]),
        random_state=int(xgboost_config["random_state"]),
    )


def _get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models_config = config.get("models")
    if not isinstance(models_config, dict):
        raise ValueError("Missing or invalid 'models' section in config.")

    model_config = models_config.get(model_name)
    if not isinstance(model_config, dict):
        raise ValueError(f"Missing or invalid 'models.{model_name}' section in config.")

    return model_config
