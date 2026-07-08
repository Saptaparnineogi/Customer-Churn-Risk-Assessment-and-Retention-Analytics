import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def clean_data(df):
    """
    Clean the DataFrame by handling missing values and duplicates.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Fill missing values with the mean of the column
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce").fillna(0)
    return df

