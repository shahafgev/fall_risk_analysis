import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_corrleations_heatmap(dataframe):
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
    metrics = [
        "AP Range", "ML Range", "AP Variance", "ML Variance", 
        "AP MAD", "ML MAD", 
        "AP Max abs dev", "ML Max abs dev", "Ellipse area", "Path length", 
        "Sway RMS"
    ]
    states = ["open", "closed"]

    # Dictionary to collect correlations
    correlations = {state: [] for state in states}

    for metric in metrics:
        for state in states:
            pivoted_df = filter_and_pivot_all_values(dataframe, metric=metric, state=state)

            if "Force_Plate" in pivoted_df.columns and "ZED_COM" in pivoted_df.columns:
                corr = pivoted_df["Force_Plate"].corr(pivoted_df["ZED_COM"])
            else:
                corr = None  # No data for one or both devices
            correlations[state].append(corr)

    # Create a DataFrame and plot the heatmap
    correlation_df = pd.DataFrame(correlations, index=metrics)

    plt.figure(figsize=(4, 5))
    sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Correlation Heatmap of Metrics by State", fontsize=16)
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.tight_layout()
    plt.show()