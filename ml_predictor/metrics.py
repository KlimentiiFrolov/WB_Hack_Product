import numpy as np

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE)"""
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()