import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def process_experiments_from_processed_data():
    """
    Processes both student and older adult participant data from the processed data directory.
    Uses filtered_measurements.csv if it exists, otherwise falls back to measurements.csv.
    Computes the average of each metric across trials for each participant, state, and device,
    and saves the result to average_measurements.csv in the same directory.
    """
    for group in ["students", "older_adults"]:
        dir_path = PROCESSED_DIR / group
        filtered_path = dir_path / "filtered_measurements.csv"
        default_path = dir_path / "measurements.csv"

        if filtered_path.exists():
            print(f"Using filtered measurements for {group}")
        else:
            print(f"Using default measurements for {group}")

        csv_path = filtered_path if filtered_path.exists() else default_path

        if not csv_path.exists():
            print(f"No valid CSV found for {group} in {dir_path}")
            continue

        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_columns = {"participant name", "state", "trial", "device", "metric", "value"}
        if not required_columns.issubset(df.columns):
            print(f"Missing required columns in {csv_path.name}")
            continue

        # Convert "value" column to numeric and drop NaNs
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # Compute average value across trials
        avg_df = (
            df.groupby(["participant name", "state", "device", "metric"], as_index=False)
            .agg(average_value=("value", "mean"))
        )

        # Save to file
        output_path = dir_path / "average_measurements.csv"
        avg_df.to_csv(output_path, index=False)
        print(f"Saved average metrics for {group} to {output_path}")



if __name__ == "__main__":
    process_experiments_from_processed_data()
