"""
Utilities for optimizing the classification decision threshold.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


Metric = Literal["f1", "precision", "recall"]


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Metric = "f1",
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Find the classification threshold that maximizes a metric.

    Parameters
    ----------
    y_true :
        Ground truth labels.

    y_prob :
        Predicted probabilities.

    metric :
        Metric to optimize.
        Options: "f1", "precision", "recall".

    thresholds :
        Thresholds to evaluate.
        Default is 0.05 to 0.95.

    Returns
    -------
    dict
        Dictionary containing:

        - best_threshold
        - best_score
    """

    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.01)

    metric_functions = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    if metric not in metric_functions:
        raise ValueError(
            f"Unsupported metric '{metric}'."
        )

    scorer = metric_functions[metric]

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:

        predictions = (
            y_prob >= threshold
        ).astype(int)

        score = scorer(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
    }