"""Model evaluation utilities for churn prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_predictions(
    model: Any,
    X: Any,
) -> tuple[Any, Any]:
    """
    Generate class predictions and positive-class probabilities.
    """
    y_pred = model.predict(X)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{type(model).__name__} does not support predict_proba()."
        )

    y_prob = model.predict_proba(X)[:, 1]

    return y_pred, y_prob


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
) -> dict[str, float]:
    """
    Calculate binary classification metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, Any]:
    """
    Evaluate one fitted classification model.

    Returns metrics, predictions, probabilities and confusion matrix.
    """
    y_pred, y_prob = get_predictions(
        model=model,
        X=X_test,
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),
    }


def evaluate_models(
    trained_models: dict[str, Any],
    X_test: Any,
    y_test: Any,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate multiple fitted models.
    """
    results = {}

    for model_name, model in trained_models.items():
        results[model_name] = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

    return results


def results_to_dataframe(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert model metrics into a comparison DataFrame.
    """
    rows = []

    for model_name, result in results.items():
        rows.append(
            {
                "Model": model_name,
                **result["metrics"],
            }
        )

    return pd.DataFrame(rows)


def print_results(results_df: pd.DataFrame) -> None:
    """
    Print the model comparison table.
    """
    print("\nModel evaluation results:")
    print(results_df.round(4).to_string(index=False))