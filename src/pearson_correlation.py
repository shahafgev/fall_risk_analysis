import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_correlations(dataframe):
    """
    Computes the correlation between ZED_COM and Force_Plate for each unique (metric, state)
    pair in the given DataFrame.

    Returns:
        pd.DataFrame with columns: ['metric', 'state', 'correlation']
    """
    
    def filter_and_pivot(df, metric, state):
        """
        Filters the DataFrame for a specific metric and state, then pivots it
        so that values from 'Force_Plate' and 'ZED_COM' appear as columns.

        Returns:
            A pivoted DataFrame ready for correlation computation.
        """
        df = df.copy()

        # Standardize value column name
        if "value" not in df.columns and "average_value" in df.columns:
            df["value"] = df["average_value"]

        # Ensure 'trial' column exists
        if "trial" not in df.columns:
            df["trial"] = 1

        # Filter by metric and state
        filtered_df = df[
            (df["metric"] == metric) &
            (df["state"].str.contains(state, case=False))
        ].copy()

        # Pivot: rows = trial-level records, columns = devices
        pivoted_df = filtered_df.pivot(
            index=["participant name", "state", "trial"],
            columns="device",
            values="value"
        ).reset_index()

        pivoted_df.columns.name = None
        return pivoted_df

    # Dynamically extract all unique metrics and states from the DataFrame
    metrics = sorted(dataframe["metric"].unique())
    states = sorted(dataframe["state"].unique())

    records = []

    # Compute correlation for each (metric, state) pair
    for metric in metrics:
        for state in states:
            pivoted_df = filter_and_pivot(dataframe, metric, state)
            if "Force_Plate" in pivoted_df.columns and "ZED_COM" in pivoted_df.columns:
                corr = pivoted_df["Force_Plate"].corr(pivoted_df["ZED_COM"])
            else:
                corr = None  # One of the devices is missing
            records.append({"metric": metric, "state": state, "correlation": corr})

    return pd.DataFrame(records)


def plot_corrleations_heatmap(dataframe):
    """
    Plots a heatmap showing the correlation between ZED_COM and Force_Plate
    for each metric and state in a single dataset.

    Parameters:
        dataframe (pd.DataFrame): The input trial- or average-level data.
    """
    # Get correlation values
    corr_df = compute_correlations(dataframe)

    # Reformat for heatmap: rows = metrics, columns = states
    heatmap_df = corr_df.pivot(index="metric", columns="state", values="correlation")

    # Plot heatmap
    plt.figure(figsize=(4, 5))
    sns.heatmap(
        heatmap_df,
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


def plot_correlation_change_bars(trial_level_df, average_level_df):
    """
    Compares the correlation between ZED_COM and Force_Plate for each metric/state
    in trial-level data vs. average-level data. Displays a grouped bar plot showing
    the difference (average - trial) in correlation.

    Parameters:
        trial_level_df (pd.DataFrame): Trial-level data.
        average_level_df (pd.DataFrame): Subject-level averaged data.
    """
    # Compute correlations for both datasets
    df_trial = compute_correlations(trial_level_df)
    df_avg = compute_correlations(average_level_df)

    # Merge and compute delta correlation
    merged = df_trial.merge(df_avg, on=["metric", "state"], suffixes=("_trial", "_avg"))
    merged["delta_corr"] = merged["correlation_avg"] - merged["correlation_trial"]

    # Plot grouped bar chart of correlation differences
    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=merged,
        x="delta_corr",
        y="metric",
        hue="state",
        palette="Set2"
    )
    plt.axvline(0, color='gray', linestyle='--')  # Reference line at no change
    plt.xlabel("Δ Correlation (Average - Trial)")
    plt.ylabel("Metric")
    plt.title("Change in Correlation by Metric and State")

    # Move legend outside the plot
    plt.legend(
        title="State",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()
