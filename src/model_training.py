"""Model definitions and training utilities for churn prediction."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """
    Return the classification models used in the project.

    Parameters
    ----------
    random_state:
        Seed used for reproducible model training.

    Returns
    -------
    dict[str, BaseEstimator]
        Dictionary mapping model names to unfitted estimators.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_seed=random_state,
        ),
    }


def build_model_pipeline(
    preprocessor: BaseEstimator,
    model: BaseEstimator,
) -> Pipeline:
    """
    Combine preprocessing and classification into one sklearn pipeline.

    Parameters
    ----------
    preprocessor:
        Fitted or unfitted preprocessing transformer.

    model:
        Classification estimator.

    Returns
    -------
    Pipeline
        Unfitted preprocessing and modeling pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )


def train_model(
    model_pipeline: Pipeline,
    X_train: Any,
    y_train: Any,
) -> Pipeline:
    """
    Fit a single model pipeline.

    Parameters
    ----------
    model_pipeline:
        Pipeline containing preprocessing and a classifier.

    X_train:
        Training feature matrix.

    y_train:
        Training target values.

    Returns
    -------
    Pipeline
        Fitted model pipeline.
    """
    fitted_pipeline = clone(model_pipeline)
    fitted_pipeline.fit(X_train, y_train)

    return fitted_pipeline


def train_models(
    preprocessor: BaseEstimator,
    models: dict[str, BaseEstimator],
    X_train: Any,
    y_train: Any,
) -> dict[str, Pipeline]:
    """
    Train multiple classification models using the same preprocessing pipeline.

    Parameters
    ----------
    preprocessor:
        Unfitted preprocessing transformer.

    models:
        Dictionary mapping model names to estimators.

    X_train:
        Training feature matrix.

    y_train:
        Training target values.

    Returns
    -------
    dict[str, Pipeline]
        Dictionary mapping model names to fitted pipelines.
    """
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        pipeline = build_model_pipeline(
            preprocessor=preprocessor,
            model=model,
        )

        trained_models[model_name] = train_model(
            model_pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
        )

    return trained_models