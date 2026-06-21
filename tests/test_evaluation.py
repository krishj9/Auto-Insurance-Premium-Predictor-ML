import numpy as np
import pytest

from backend.app.evaluation.metrics import rmse, rmsle


def test_rmse_returns_expected_value() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    value = rmse(y_true=y_true, y_pred=y_pred)

    assert value == pytest.approx(10.0)


def test_rmsle_clips_negative_predictions() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, -50.0, 330.0])

    value = rmsle(y_true=y_true, y_pred=y_pred)
    expected = np.sqrt(
        np.mean(
            (
                np.log1p(y_true)
                - np.log1p(
                    np.array(
                        [
                            90.0,
                            0.0,
                            330.0,
                        ]
                    )
                )
            )
            ** 2
        )
    )

    assert value == pytest.approx(float(expected))


def test_rmsle_raises_for_negative_true_values() -> None:
    y_true = np.array([100.0, -1.0, 300.0])
    y_pred = np.array([90.0, 200.0, 330.0])

    with pytest.raises(ValueError):
        rmsle(y_true=y_true, y_pred=y_pred)


def test_metrics_raise_for_shape_mismatch() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        rmse(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError):
        rmsle(y_true=y_true, y_pred=y_pred)
