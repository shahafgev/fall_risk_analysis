import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def plot_rmse_heatmap(dataframe):
    def filter_and_pivot_all_values(df, metric, state):
        # Create a copy of the dataframe
        df = df.copy()

        # Ensure 'value' column exists
        if "value" not in df.columns and "average_value" in df.columns:
            df["value"] = df["average_value"]

        # Ensure 'trial' column exists for consistent pivoting
        if "trial" not in df.columns:
            df["trial"] = 1

        # Filter for the specific metric and state
        filtered_df = df[
            (df["metric"] == metric) & 
            (df["state"].str.contains(state, case=False))
        ].copy()

        # Pivot to get both devices side by side
        pivoted_df = filtered_df.pivot(
            index=["participant name", "state", "trial"], 
            columns="device", 
            values="value"
        ).reset_index()

        pivoted_df.columns.name = None
        return pivoted_df

    # List of metrics and states
    metrics = ['ML Range', 'AP Range', 'Range Ratio', 'ML Variance', 'AP Variance',
               'ML STD', 'AP STD', 'ML MAD', 'AP MAD', 'ML MedAD', 'AP MedAD',
               'ML Max abs dev', 'AP Max abs dev', 'Ellipse area', 'Path length',
               'ML RMS', 'AP RMS', 'Sway RMS']
    states = ["open", "closed"]

    # Dictionary to collect RMSE values
    rmse_values = {state: [] for state in states}

    for metric in metrics:
        for state in states:
            pivoted_df = filter_and_pivot_all_values(dataframe, metric=metric, state=state)

            if "Force_Plate" in pivoted_df.columns and "ZED_COM" in pivoted_df.columns:
                rmse = np.sqrt(mean_squared_error(pivoted_df["Force_Plate"], pivoted_df["ZED_COM"]))
            else:
                rmse = np.nan  # No data for one or both devices
            rmse_values[state].append(rmse)

    # Create a DataFrame and plot the heatmap
    rmse_df = pd.DataFrame(rmse_values, index=metrics)

    plt.figure(figsize=(4, 5))
    sns.heatmap(
        rmse_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={'label': 'RMSE'}
    )
    plt.title("RMSE Heatmap of Metrics by State", fontsize=16)
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.tight_layout()
    plt.show()
