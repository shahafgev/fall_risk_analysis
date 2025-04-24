import pandas as pd
import matplotlib.pyplot as plt


def bland_altman_plot_by_state(df, metric_name, average_level=False):
    """
    Creates side-by-side Bland–Altman plots (eyes open vs. eyes closed)
    comparing 'Force_Plate' vs. 'ZED_COM' for the specified metric.

    Parameters:
    - df: DataFrame with columns:
        ['participant name', 'state', 'metric', 'device', 'value'] for trial-level
        or ['participant name', 'state', 'metric', 'device', 'average_value'] for average-level
    - metric_name: The metric to plot.
    - average_level: If True, uses 'average_value' instead of 'value'.
    """

    # Decide unit
    square_unit_metrics = {"Ellipse area", "ML Variance", "AP Variance"}
    unit = "cm²" if metric_name in square_unit_metrics else "cm"

    # Filter only for selected metric and devices
    df_metric = df[
        (df['metric'] == metric_name) & 
        (df['device'].isin(['Force_Plate', 'ZED_COM']))
    ].copy()

    # Pick which value column to use
    value_col = 'average_value' if average_level else 'value'

    # Check that the selected value column exists
    if value_col not in df_metric.columns:
        raise ValueError(f"Expected column '{value_col}' not found in the dataframe.")

    # Get all states (e.g., 'open', 'closed')
    states = df_metric['state'].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(states), figsize=(6 * len(states), 5), sharey=True)
    if len(states) == 1:
        axes = [axes]

    for i, st in enumerate(states):
        ax = axes[i]
        df_state = df_metric[df_metric['state'] == st]

        # Adjust pivot index
        pivot_index = ['participant name', 'state', 'metric']
        if not average_level:
            if 'trial' in df_state.columns:
                pivot_index.insert(1, 'trial')
            else:
                raise ValueError("Expected 'trial' column for trial-level data.")

        # Pivot to wide format
        df_pivot = df_state.pivot_table(
            index=pivot_index,
            columns='device',
            values=value_col
        ).reset_index()

        df_pivot.dropna(subset=['Force_Plate', 'ZED_COM'], inplace=True)

        # Compute Bland–Altman values
        df_pivot['mean_measure'] = (df_pivot['Force_Plate'] + df_pivot['ZED_COM']) / 2
        df_pivot['diff_measure'] = df_pivot['Force_Plate'] - df_pivot['ZED_COM']
        mean_diff = df_pivot['diff_measure'].mean()
        sd_diff = df_pivot['diff_measure'].std()

        # Plot points and lines
        ax.scatter(df_pivot['mean_measure'], df_pivot['diff_measure'], alpha=0.8)
        ax.axhline(mean_diff, color='black', linestyle='-')
        ax.axhline(mean_diff + 1.96 * sd_diff, color='black', linestyle='--')
        ax.axhline(mean_diff - 1.96 * sd_diff, color='black', linestyle='--')

        # Labels and title
        ax.set_title(f"{metric_name} - eyes {st}")
        ax.set_xlabel(f"Mean of devices ({unit})")
        ax.set_ylabel(f"Difference between devices ({unit})")
        ax.tick_params(labelleft=True)

        # Text annotations
        ax.figure.canvas.draw()
        xlim = ax.get_xlim()
        x_text = xlim[1] - 0.02 * (xlim[1] - xlim[0])
        range_diff = df_pivot['diff_measure'].max() - df_pivot['diff_measure'].min()
        offset = 0.02 * range_diff

        ax.text(x_text, mean_diff + offset,
                f"Mean/Bias : {mean_diff:.2f} {unit}", ha='right', va='bottom')
        ax.text(x_text, mean_diff + 1.96 * sd_diff + offset,
                f"+1.96 SD: {mean_diff + 1.96 * sd_diff:.2f} {unit}", ha='right', va='bottom')
        ax.text(x_text, mean_diff - 1.96 * sd_diff - offset,
                f"-1.96 SD: {mean_diff - 1.96 * sd_diff:.2f} {unit}", ha='right', va='top')

    plt.tight_layout()
    plt.show()
