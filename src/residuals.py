import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_residuals_by_metric_and_state(dataframe, average_level=False):
    """
    For each metric, plots residuals (ZED_COM - Force_Plate) vs Force_Plate values
    separately for 'open' and 'closed' states. Includes LOESS trend and ±2SD bands.

    Parameters:
        dataframe (pd.DataFrame): Trial- or average-level data with columns:
                                  ['participant name', 'state', 'trial', 'device', 'metric', 'value']
        average_level (bool): Whether the data is average-level (no 'trial' column). Default is False.
    """
    metrics = sorted(dataframe["metric"].unique())
    states = ["open", "closed"]

    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        fig.suptitle(f"Residuals vs Force Plate: {metric}", fontsize=14)

        for i, state in enumerate(states):
            df_filtered = dataframe[
            (dataframe["metric"] == metric) &
            (dataframe["state"].str.lower() == state)
            ].copy()

            # Rename column if average_level is True
            if average_level and "average_value" in df_filtered.columns:
                df_filtered = df_filtered.rename(columns={"average_value": "value"})

            # If no 'trial' column, create dummy value
            if "trial" not in df_filtered.columns:
                df_filtered["trial"] = 1

            # Pivot to align ZED_COM and Force_Plate
            df_pivoted = df_filtered.pivot_table(
                index=["participant name", "trial", "state"],
                columns="device",
                values="value"
            ).reset_index()

            ax = axes[i]
            if "ZED_COM" in df_pivoted.columns and "Force_Plate" in df_pivoted.columns:
                x = df_pivoted["Force_Plate"]
                residuals = df_pivoted["ZED_COM"] - df_pivoted["Force_Plate"]

                # Plot residual scatter
                sns.scatterplot(x=x, y=residuals, ax=ax)

                # Mean and SD of residuals
                mean_res = residuals.mean()
                std_res = residuals.std()

                # Axis range
                x_pos = ax.get_xlim()[1]

                # Plot ±2SD bands and mean line with text
                ax.axhline(mean_res, color='black', linewidth=1)
                ax.text(x_pos, mean_res, f"Mean: {mean_res:.2f} cm", va='bottom', ha='right', fontsize=10, color='black')

                ax.axhline(mean_res + 2 * std_res, color='black', linestyle='--')
                ax.text(x_pos, mean_res + 2 * std_res, f"+2 SD: {mean_res + 2 * std_res:.2f} cm", va='bottom', ha='right', fontsize=10, color='black')

                ax.axhline(mean_res - 2 * std_res, color='black', linestyle='--')
                ax.text(x_pos, mean_res - 2 * std_res, f"-2 SD: {mean_res - 2 * std_res:.2f} cm", va='top', ha='right', fontsize=10, color='black')

                ax.set_title(f"{state.capitalize()}")
                ax.set_xlabel("Force Plate Value (cm)")
                ax.set_ylabel("Residual (ZED - FP) (cm)" if i == 0 else "")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
