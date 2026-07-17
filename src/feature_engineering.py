from __future__ import annotations

import numpy as np
import pandas as pd

TENURE_BINS = [-1, 12, 24, 48, 72]

TENURE_LABELS = [
    "0-12 months",
    "13-24 months",
    "25-48 months",
    "49-72 months",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group customer tenure into lifecycle stages.

    Parameters
    ----------
    df:
        Customer-level dataset containing a numeric ``tenure`` column.

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with a new ``tenure_group`` column.

    Raises
    ------
    KeyError
        If the ``tenure`` column is missing.
    ValueError
        If tenure contains values outside the expected range of 0–72 months.
    """
    if "tenure" not in df.columns:
        raise KeyError(
            "The dataframe must contain a 'tenure' column."
        )

    result = df.copy()

    tenure = pd.to_numeric(result["tenure"], errors="coerce")

    if tenure.isna().any():
        invalid_count = int(tenure.isna().sum())
        raise ValueError(
            f"'tenure' contains {invalid_count} missing or non-numeric value(s)."
        )

    if not tenure.between(0, 72).all():
        invalid_values = sorted(
            tenure.loc[~tenure.between(0, 72)].unique().tolist()
        )
        raise ValueError(
            "Tenure values must be between 0 and 72 months. "
            f"Invalid values found: {invalid_values}"
        )

    result["tenure_group"] = pd.cut(
        tenure,
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
        ordered=True,
    )

    return result


def add_family_commitment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicate whether a customer has a partner or dependents.

    A customer is considered to have family commitment when either:

    - ``Partner == 'Yes'``, or
    - ``Dependents == 'Yes'``.

    Parameters
    ----------
    df:
        Customer-level dataset containing ``Partner`` and ``Dependents``.

    Returns
    -------
    pd.DataFrame
        A copy with a binary ``has_family_commitment`` column.

    Raises
    ------
    KeyError
        If either required source column is missing.
    ValueError
        If the columns contain values other than Yes or No.
    """
    required_columns = {"Partner", "Dependents"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(
            "Missing required column(s): "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    allowed_values = {"Yes", "No"}

    for column in required_columns:
        observed_values = set(result[column].dropna().unique())
        unexpected_values = observed_values.difference(allowed_values)

        if unexpected_values:
            raise ValueError(
                f"Unexpected values in '{column}': "
                f"{sorted(unexpected_values)}. Expected Yes or No."
            )

    result["has_family_commitment"] = (
        result["Partner"].eq("Yes")
        | result["Dependents"].eq("Yes")
    ).astype(int)

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all final feature-engineering steps.

    Parameters
    ----------
    df:
        Clean customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing the original variables and engineered features.
    """
    result = add_tenure_group(df)
    result = add_family_commitment(result)

    return result