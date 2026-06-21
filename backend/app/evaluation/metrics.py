import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    y_true_array, y_pred_array = _validate_inputs(y_true=y_true, y_pred=y_pred)
    return float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared logarithmic error with non-negative clipping."""
    y_true_array, y_pred_array = _validate_inputs(y_true=y_true, y_pred=y_pred)
    if np.any(y_true_array < 0):
        raise ValueError("RMSLE requires non-negative y_true values.")

    clipped_predictions = np.clip(y_pred_array, 0.0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_array), np.log1p(clipped_predictions))))


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return y_true_array, y_pred_array
