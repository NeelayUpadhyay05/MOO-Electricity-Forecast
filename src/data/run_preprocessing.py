from src.utils.seed import set_seed
from src.data.preprocess import (
    load_electricity_dataset,
    impute_missing_values,
    convert_to_hourly,
    chronological_split,
    select_households,
    normalize_per_household,
    save_processed_data
)


def main():
    set_seed(42)

    filepath = "data/raw/LD2011_2014.txt"

    print("--- Loading dataset ---")
    df = load_electricity_dataset(filepath)
    print(f"  Loaded: {df.shape[0]:,} timesteps x {df.shape[1]} households")

    print("\n--- Imputing missing values ---")
    df = impute_missing_values(df)

    print("\n--- Converting to hourly resolution ---")
    hourly_df = convert_to_hourly(df)
    print(f"  Shape after resampling: {hourly_df.shape[0]:,} hours x {hourly_df.shape[1]} households")

    print("\n--- Splitting chronologically ---")
    train_df, val_df, test_df = chronological_split(hourly_df)
    print(f"  Train : {train_df.shape[0]:,} hours  (2011-01-01 to 2013-12-31)")
    print(f"  Val   : {val_df.shape[0]:,} hours  (2014-01-01 to 2014-06-30)")
    print(f"  Test  : {test_df.shape[0]:,} hours  (2014-07-01 to 2014-12-31)")

    print("\n--- Selecting households ---")
    train_df, val_df, test_df, selected_ids = select_households(
        train_df, val_df, test_df, n_households=100
    )

    print("\n--- Normalizing ---")
    train_scaled, val_scaled, test_scaled, scaling_params = normalize_per_household(
        train_df, val_df, test_df
    )

    print("\n--- Saving processed data ---")
    save_processed_data(
        train_scaled, val_scaled, test_scaled,
        scaling_params, selected_ids
    )

    print("\nPreprocessing completed successfully.")
    print(f"  Final dataset: {len(selected_ids)} households")
    print(f"  Train shape : {train_scaled.shape}")
    print(f"  Val shape   : {val_scaled.shape}")
    print(f"  Test shape  : {test_scaled.shape}")


if __name__ == "__main__":
    main()
