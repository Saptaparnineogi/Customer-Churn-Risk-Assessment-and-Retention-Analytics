"""
Model evaluation utilities.
"""

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


def get_predictions(model, X):
    """
    Generate predictions and prediction probabilities.
    """

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return y_pred, y_prob


def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):
    """
    Calculate evaluation metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }

    return metrics


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a single trained model.
    """

    y_pred, y_prob = get_predictions(model, X_test)

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    metrics["Confusion Matrix"] = confusion_matrix(
        y_test,
        y_pred,
    )

    return metrics


def evaluate_models(
    trained_models,
    X_test,
    y_test,
):
    """
    Evaluate all trained models.
    """

    results = {}

    for model_name, model in trained_models.items():

        results[model_name] = evaluate_model(
            model,
            X_test,
            y_test,
        )

    return results


def results_to_dataframe(results):
    """
    Convert evaluation dictionary into a DataFrame.
    """

    rows = []

    for model_name, metrics in results.items():

        row = {
            "Model": model_name,
            **{
                k: v
                for k, v in metrics.items()
                if k != "Confusion Matrix"
            },
        }

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="ROC-AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def print_results(results_df):
    """
    Pretty print evaluation results.
    """

    print(results_df.round(4))