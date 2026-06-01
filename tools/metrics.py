import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    min_value = float(scores.min())
    max_value = float(scores.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - min_value) / (max_value - min_value)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    metrics = {
        "auroc": np.nan,
        "aupr": np.nan,
        "f1max": np.nan,
    }

    if np.unique(y_true).size < 2:
        return metrics

    metrics["auroc"] = float(roc_auc_score(y_true, y_score))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    metrics["aupr"] = float(auc(recall, precision))

    f1_scores = (2.0 * precision * recall) / (precision + recall + 1e-12)
    finite_mask = np.isfinite(f1_scores)
    if np.any(finite_mask):
        metrics["f1max"] = float(np.max(f1_scores[finite_mask]))

    return metrics

def format_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.5f}"