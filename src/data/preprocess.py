import os
import json
import numpy as np
import pandas as pd


def load_electricity_dataset(filepath: str) -> pd.DataFrame:
    """
    Load ElectricityLoadDiagrams20112014 dataset.
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        decimal=",",
        parse_dates=[0],
        index_col=0
    )

    # Replace missing values with NaN
    df = df.replace("?", np.nan)

    # Convert all columns to float
    df = df.astype(float)

    return df


def convert_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 15-minute data to hourly resolution using mean.
    """
    hourly_df = df.resample("h").mean()
    return hourly_df

def select_households(df: pd.DataFrame, n_households: int = 100, seed: int = 42):
    """
    Randomly select a subset of households.
    """
    np.random.seed(seed)
    selected_columns = np.random.choice(df.columns, size=n_households, replace=False)
    selected_df = df[selected_columns]

    return selected_df, list(selected_columns)