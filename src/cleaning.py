"""
Reusable data-cleaning utilities for the crop-yield project.
"""

from pathlib import Path
import pandas as pd


TARGET_COLUMN = "yield_tons_per_ha"


def load_data(path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)


def basic_cleaning(df):
    """
    Apply safe, reusable cleaning operations.

    This function does not use the target to modify features and does
    not use leaderboard test data for model selection.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    data = df.copy()

    # Remove completely duplicated rows
    data = data.drop_duplicates().reset_index(drop=True)

    # Normalize column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Convert common numeric columns when possible
    for column in data.columns:
        if data[column].dtype == "object":
            converted = pd.to_numeric(data[column], errors="coerce")

            # Only replace when most non-null values are numeric
            non_null = data[column].notna().sum()

            if non_null > 0:
                numeric_count = converted.notna().sum()

                if numeric_count / non_null >= 0.90:
                    data[column] = converted

    return data


def validate_required_columns(df, required_columns):
    """
    Validate that required columns exist.

    Parameters
    ----------
    df : pandas.DataFrame
    required_columns : list[str]

    Raises
    ------
    ValueError
        If one or more columns are missing.
    """
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def save_cleaned_data(df, path):
    """
    Save a cleaned DataFrame to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    return path