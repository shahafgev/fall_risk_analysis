import pandas as pd
import os

# Define columns to drop
columns_to_drop = [
    'Range Ratio', 'Path length'
]

def transform_and_save_dataframe(df, state, device, save_name, average_level=False):
    """
    Transforms the dataframe (trial-level or average-level),
    merges with additional participant information,
    and saves the result to data/processed/datasets/.

    Parameters:
    - df (DataFrame): The input dataset.
    - state (str): The state to filter by (e.g., 'closed' or 'open').
    - device (str): The device to filter by (e.g., 'Force_Plate' or 'ZED_Camera').
    - save_name (str): The name for the saved CSV file (without .csv extension).
    - average_level (bool): If True, pivot without trial info (one row per participant).

    Returns:
    - DataFrame: Transformed DataFrame.
    """
    # Determine value column
    value_col = "average_value" if "average_value" in df.columns else "value"

    # Filter for the specified state and device
    df_filtered = df[
        (df["state"] == state) &
        (df["device"] == device)
    ].copy()

    # Choose pivot index
    if average_level:
        pivot_index = ["participant name"]
    else:
        pivot_index = ["participant name", "trial"]

    # Pivot metrics into wide format
    df_pivoted = df_filtered.pivot_table(
        index=pivot_index,
        columns="metric",
        values=value_col
    ).reset_index()

    # Load additional participant information
    additional_info_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'additional_participants_information.csv')
    additional_info = pd.read_csv(additional_info_path)

    # Merge (only on participant name)
    df_transformed = pd.merge(additional_info, df_pivoted, on="participant name", how="inner")

    # Drop unwanted columns before saving
    df_cleaned = df_transformed.drop(columns=[col for col in columns_to_drop if col in df_transformed.columns])

    # Define save path (relative to src/)
    save_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'ML_datasets')
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f"{save_name}.csv")

    # Save the file
    df_cleaned.to_csv(save_path, index=False)
    print(f"Transformed and merged DataFrame saved to: {save_path}")





if __name__ == "__main__":
    # =============================== Create Zed and Force Plate ML datasets ===============================
    # input file (same for all)
    input_file = "data/processed/older_adults/average_measurements.csv"
    df = pd.read_csv(input_file)

    # list of (state, device, output_name)
    configs = [
        # Zed
        ("open", "ZED_COM", "oa_averages_open_zed"),
        ("closed", "ZED_COM", "oa_averages_closed_zed"),
        # Force Plate
        ("open", "Force_Plate", "oa_averages_open_fp"),
        ("closed", "Force_Plate", "oa_averages_closed_fp"),  
    ]

    # loop through configs
    for state, device, outname in configs:
        transform_and_save_dataframe(df, state, device, outname, average_level=True)

    # =============================== Create Phones ML datasets ===============================
    # input file (same for all)
    input_file = "data/processed/older_adults/phones_measurements.csv"
    df = pd.read_csv(input_file)

    # list of (state, device, output_name)
    configs = [
        # Front Phone
        ("open", "front", "oa_averages_open_front"),
        ("closed", "front", "oa_averages_closed_front"),
        # Back Phone
        ("open", "back", "oa_averages_open_back"),
        ("closed", "back", "oa_averages_closed_back"),     
    ]

    # loop through configs
    for state, device, outname in configs:
        transform_and_save_dataframe(df, state, device, outname, average_level=True)