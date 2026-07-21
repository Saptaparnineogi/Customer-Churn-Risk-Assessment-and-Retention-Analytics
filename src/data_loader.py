"""Data loading utilities for the churn prediction project."""

from pathlib import Path

import pandas as pd


def load_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load the customer churn dataset.

    Parameters
    ----------
    data_path:
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    ValueError
        If the loaded dataset is empty.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
    )
    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df