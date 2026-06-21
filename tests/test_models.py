import sys
import types
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from backend.app.models.model_definitions import build_linear_regression, build_random_forest, build_xgboost


def test_build_linear_regression_returns_unfitted_model() -> None:
    model = build_linear_regression()
    assert isinstance(model, LinearRegression)
    assert not hasattr(model, "coef_")


def test_build_random_forest_uses_config_values() -> None:
    config: dict[str, Any] = {
        "models": {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "random_state": 42,
            }
        }
    }

    model = build_random_forest(config=config)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 200
    assert model.max_depth == 15
    assert model.random_state == 42
    assert not hasattr(model, "estimators_")


def test_build_xgboost_uses_config_values(monkeypatch: Any) -> None:
    class FakeXGBRegressor:
        def __init__(
            self, n_estimators: int, max_depth: int, learning_rate: float, random_state: int
        ) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.random_state = random_state

    fake_xgboost_module = types.ModuleType("xgboost")
    fake_xgboost_module.XGBRegressor = FakeXGBRegressor
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost_module)

    config: dict[str, Any] = {
        "models": {
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }
        }
    }

    model = build_xgboost(config=config)

    assert isinstance(model, FakeXGBRegressor)
    assert model.n_estimators == 300
    assert model.max_depth == 6
    assert model.learning_rate == 0.1
    assert model.random_state == 42
