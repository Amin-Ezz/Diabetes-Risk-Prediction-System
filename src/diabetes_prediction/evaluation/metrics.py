"""Evaluation metrics computation for clinical classification tasks."""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_name: str = "target",
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic evaluation metrics for a binary target.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        target_name: Target identifier name.

    Returns:
        Dictionary of computed performance metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "target": target_name,
        "threshold": round(float(threshold), 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "specificity": round(spec, 4),
        "mcc": round(mcc, 4),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
