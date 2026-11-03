"""Standard classification metrics, computed from ACTUAL predictions only.
No metric in this file is ever fabricated -- every call requires real
y_true/y_pred arrays produced by running the model.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, average: str = "macro") -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def get_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def evidence_degradation(dess_before: dict, dess_after: dict) -> dict:
    """Delta DESS per class (dess_after - dess_before). Classes with an
    undefined (None) DESS on either side are reported as None, not silently
    skipped or zero-filled."""
    delta = {}
    for c in set(dess_before) | set(dess_after):
        b, a = dess_before.get(c), dess_after.get(c)
        delta[c] = (a - b) if (a is not None and b is not None) else None
    return delta


def reliability_shift(r_before: np.ndarray, r_after: np.ndarray) -> dict:
    """Delta R(x) statistics between two model variants (e.g. FP32 student
    vs INT8 student) evaluated on the SAME samples in the SAME order."""
    delta = np.asarray(r_after) - np.asarray(r_before)
    return {
        "mean_delta_R": float(delta.mean()),
        "std_delta_R": float(delta.std()),
        "min_delta_R": float(delta.min()),
        "max_delta_R": float(delta.max()),
        "frac_decreased": float((delta < 0).mean()),
    }
