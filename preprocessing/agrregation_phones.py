import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def process_phone_experiments_from_processed_data():
    """
    Reads phones_measurements.csv for each group (students, older_adults),
    averages metric values across trials per (participant name, state, metric, device),
    and writes average_phones_measurements.csv in the same folder.
    """
    for group in ["students", "older_adults"]:
        dir_path = PROCESSED_DIR / group
        csv_path = dir_path / "phones_measurements.csv"

        if not csv_path.exists():
            print(f"[{group}] No phones_measurements.csv in {dir_path}")
            continue

        df = pd.read_csv(csv_path)

        # Validate required columns
        required = {"participant name", "state", "trial", "device", "metric", "value"}
        missing = required - set(df.columns)
        if missing:
            print(f"[{group}] Missing required columns: {sorted(missing)}")
            continue

        # Ensure numeric values and drop NaNs
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            print(f"[{group}] All values are NaN after coercion. Skipping.")
            continue

        groupby_cols = ["participant name", "state", "metric", "device"]

        avg_df = (
            df.groupby(groupby_cols, as_index=False)
              .agg(average_value=("value", "mean"))
              .sort_values(groupby_cols)
        )

        output_path = dir_path / "average_phones_measurements.csv"
        avg_df.to_csv(output_path, index=False)
        print(f"[{group}] Saved: {output_path}")


if __name__ == "__main__":
    process_phone_experiments_from_processed_data()
