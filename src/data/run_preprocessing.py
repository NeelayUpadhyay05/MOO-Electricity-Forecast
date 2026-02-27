from src.utils.seed import set_seed
from src.data.preprocess import (
    load_electricity_dataset,
    convert_to_hourly,
    select_households,
    chronological_split,
    normalize_per_household,
    save_processed_data
)


def main():
    set_seed(42)

    filepath = "data/raw/LD2011_2014.txt"

    df = load_electricity_dataset(filepath)
    hourly_df = convert_to_hourly(df)
    selected_df, selected_ids = select_households(hourly_df, n_households=100)

    train_df, val_df, test_df = chronological_split(selected_df)

    train_scaled, val_scaled, test_scaled, scaling_params = normalize_per_household(
        train_df, val_df, test_df
    )

    save_processed_data(
        train_scaled,
        val_scaled,
        test_scaled,
        scaling_params,
        selected_ids
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()