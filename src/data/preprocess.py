import os
import json
import numpy as np
import pandas as pd


def load_electricity_dataset(filepath: str) -> pd.DataFrame:
    """
    Load ElectricityLoadDiagrams20112014 dataset.
    European format: semicolon separator, comma decimal.
    Corrupt / unparseable values are coerced to NaN.
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        decimal=",",
        parse_dates=[0],
        index_col=0,
        low_memory=False
    )

    # Replace known missing-value marker and cast everything to float
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Replace leading zeros with NaN.
       Many meters show 0 for months before installation — these are
       missing data, not genuine zero consumption.
    2. Forward-fill then backward-fill all remaining NaN gaps.
    """
    leading_replaced = 0

    for col in df.columns:
        values = df[col].values.copy().astype(float)

        # Find the first timestamp with a finite, positive reading
        valid_mask = np.isfinite(values) & (values > 0)
        if valid_mask.any():
            first_valid = int(valid_mask.argmax())
            if first_valid > 0:
                zero_mask = values[:first_valid] == 0
                leading_replaced += int(zero_mask.sum())
                values[:first_valid] = np.where(zero_mask, np.nan, values[:first_valid])
                df[col] = values

    total_nan = int(df.isna().sum().sum())

    # Fill gaps
    df = df.ffill().bfill()

    remaining = int(df.isna().sum().sum())

    print(f"  Leading zeros replaced with NaN:    {leading_replaced:,}")
    print(f"  Total NaN values imputed:            {total_nan:,}")
    if remaining > 0:
        print(f"  WARNING: {remaining:,} values still NaN (completely empty series)")

    return df


def convert_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 15-minute data to hourly resolution using mean.
    Mean is consistent with prior work on this dataset.
    """
    return df.resample("h").mean()


def chronological_split(df: pd.DataFrame):
    """
    Split dataset chronologically into train, validation, and test sets.
      Train : 2011-01-01 – 2013-12-31  (3 years)
      Val   : 2014-01-01 – 2014-06-30  (6 months)
      Test  : 2014-07-01 – 2014-12-31  (6 months)
    """
    train_df = df.loc["2011-01-01":"2013-12-31"]
    val_df   = df.loc["2014-01-01":"2014-06-30"]
    test_df  = df.loc["2014-07-01":"2014-12-31"]
    return train_df, val_df, test_df


def select_households(train_df, val_df, test_df,
                      n_households: int = 100, seed: int = 42):
    """
    Filter out constant or near-constant households (using training set statistics),
    then randomly select exactly n_households from the valid pool.

    Uses max-min range rather than variance because np.nanvar on large constant
    float64 arrays can return a tiny positive value due to floating-point
    summation accumulation, causing constant households to pass the var > 0 check.

    Splitting before selecting guarantees that the final dataset
    contains exactly n_households with meaningful variation in training.
    """
    # Identify households with meaningful range in training data.
    # Threshold 1e-6 safely excludes constant bfill-padded series while
    # keeping any household with real variation (typical range >> 1).
    ranges = train_df.max() - train_df.min()
    valid_cols = ranges[ranges > 1e-6].index.tolist()
    n_removed = len(train_df.columns) - len(valid_cols)

    if n_removed > 0:
        print(f"  Households removed (constant or near-constant in train): {n_removed}")

    if len(valid_cols) < n_households:
        raise ValueError(
            f"Only {len(valid_cols)} valid households available; "
            f"cannot select {n_households}."
        )

    np.random.seed(seed)
    selected = np.random.choice(valid_cols, size=n_households, replace=False).tolist()
    print(f"  Households selected: {n_households} from {len(valid_cols)} valid")

    return train_df[selected], val_df[selected], test_df[selected], selected


def normalize_per_household(train_df, val_df, test_df):
    """
    Min-max normalize each household independently using training statistics.
    Val/test values that fall outside [0, 1] are clipped and reported —
    this prevents out-of-distribution inputs to the model.
    """
    scaling_params = {}
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    val_clipped  = 0
    test_clipped = 0

    for col in train_df.columns:
        min_val = float(train_df[col].min())
        max_val = float(train_df[col].max())
        rng     = max_val - min_val

        scaling_params[col] = {"min": min_val, "max": max_val}

        if rng < 1e-8:
            # Should be excluded by select_households; raise to surface any logic gap.
            raise ValueError(
                f"Household '{col}' has zero range in training "
                f"(min=max={min_val:.6f}). Remove it via select_households first."
            )

        train_scaled[col] = (train_df[col] - min_val) / rng

        val_norm  = (val_df[col]  - min_val) / rng
        test_norm = (test_df[col] - min_val) / rng

        val_clipped  += int(((val_norm  < 0) | (val_norm  > 1)).sum())
        test_clipped += int(((test_norm < 0) | (test_norm > 1)).sum())

        val_scaled[col]  = val_norm.clip(0.0, 1.0)
        test_scaled[col] = test_norm.clip(0.0, 1.0)

    total_val  = val_df.size
    total_test = test_df.size

    if val_clipped > 0:
        print(f"  Val  values clipped to [0,1]: "
              f"{val_clipped:,} / {total_val:,} "
              f"({val_clipped / total_val * 100:.2f}%)")
    else:
        print("  Val  values: all within [0,1] training range")

    if test_clipped > 0:
        print(f"  Test values clipped to [0,1]: "
              f"{test_clipped:,} / {total_test:,} "
              f"({test_clipped / total_test * 100:.2f}%)")
    else:
        print("  Test values: all within [0,1] training range")

    return train_scaled, val_scaled, test_scaled, scaling_params


def save_processed_data(train_df, val_df, test_df,
                        scaling_params, selected_ids,
                        save_dir="data/processed"):
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
        json.dump(selected_ids, f, indent=4)
