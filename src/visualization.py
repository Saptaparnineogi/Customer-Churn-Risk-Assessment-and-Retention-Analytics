"""
Visualization utilities for model evaluation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.config import FIGURE_DIR


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
):
    """Plot confusion matrix."""

    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    return fig


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name: str,
):
    """Plot ROC curve."""

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
    )

    ax.set_title(f"{model_name} ROC Curve")

    plt.tight_layout()

    return fig


def plot_precision_recall_curve(
    model,
    X_test,
    y_test,
    model_name: str,
):
    """Plot Precision-Recall curve."""

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
    )

    ax.set_title(f"{model_name} Precision-Recall Curve")

    plt.tight_layout()

    return fig


def plot_feature_importance(
    model,
    feature_names,
    top_n: int = 20,
):
    """
    Plot feature importance for tree-based models.
    """

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            "Model does not expose feature_importances_."
        )

    importance = estimator.feature_importances_

    idx = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        np.array(feature_names)[idx][::-1],
        importance[idx][::-1],
    )

    ax.set_title("Feature Importance")

    plt.tight_layout()

    return fig


def save_figure(fig, filename):
    FIGURE_DIR.mkdir(exist_ok=True)

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )