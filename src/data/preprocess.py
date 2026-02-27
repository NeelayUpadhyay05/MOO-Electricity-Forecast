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

def chronological_split(df: pd.DataFrame):
    """
    Split dataset chronologically into train, validation, and test sets.
    """

    train_df = df.loc["2011-01-01":"2013-12-31"]
    val_df = df.loc["2014-01-01":"2014-06-30"]
    test_df = df.loc["2014-07-01":"2014-12-31"]

    return train_df, val_df, test_df

def normalize_per_household(train_df, val_df, test_df):
    """
    Normalize each household independently using train statistics only.
    Remove households with zero variance in training.
    """

    scaling_params = {}

    valid_columns = []
    removed_columns = []

    for col in train_df.columns:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        range_val = max_val - min_val

        if range_val == 0:
            removed_columns.append(col)
        else:
            valid_columns.append(col)

    if removed_columns:
        print(f"Removing {len(removed_columns)} households with zero variance in training.")

    # Keep only valid columns
    train_df = train_df[valid_columns]
    val_df = val_df[valid_columns]
    test_df = test_df[valid_columns]

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    for col in valid_columns:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        range_val = max_val - min_val

        scaling_params[col] = {
            "min": float(min_val),
            "max": float(max_val)
        }

        train_scaled[col] = (train_df[col] - min_val) / range_val
        val_scaled[col] = (val_df[col] - min_val) / range_val
        test_scaled[col] = (test_df[col] - min_val) / range_val

    return train_scaled, val_scaled, test_scaled, scaling_params

def save_processed_data(train_df, val_df, test_df, scaling_params, selected_ids, save_dir="data/processed"):
    """
    Save processed datasets and scaling parameters.
    """

    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(os.path.join(save_dir, "electricity_train.csv"))
    val_df.to_csv(os.path.join(save_dir, "electricity_val.csv"))
    test_df.to_csv(os.path.join(save_dir, "electricity_test.csv"))

    with open(os.path.join(save_dir, "electricity_scaling.json"), "w") as f:
        json.dump(scaling_params, f, indent=4)

    with open(os.path.join(save_dir, "selected_households.json"), "w") as f:
        json.dump(list(train_df.columns), f, indent=4)