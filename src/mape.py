import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_mape(dataframe):
    """
    Computes the Mean Absolute Percentage Error (MAPE) between ZED_COM and Force_Plate
    for each unique (metric, state) pair in the given DataFrame.

    Returns:
        pd.DataFrame with columns: ['metric', 'state', 'mape']
    """
    def filter_and_pivot(df, metric, state):
        df = df.copy()

        # Standardize value column name
        if "value" not in df.columns and "average_value" in df.columns:
            df["value"] = df["average_value"]
        if "trial" not in df.columns:
            df["trial"] = 1

        filtered_df = df[
            (df["metric"] == metric) &
            (df["state"].str.contains(state, case=False))
        ].copy()

        pivoted_df = filtered_df.pivot(
            index=["participant name", "state", "trial"],
            columns="device",
            values="value"
        ).reset_index()

        pivoted_df.columns.name = None
        return pivoted_df

    metrics = sorted(dataframe["metric"].unique())
    states = sorted(dataframe["state"].unique())

    records = []
    for metric in metrics:
        for state in states:
            pivoted_df = filter_and_pivot(dataframe, metric, state)
            if "Force_Plate" in pivoted_df.columns and "ZED_COM" in pivoted_df.columns:
                fp = pivoted_df["Force_Plate"]
                zed = pivoted_df["ZED_COM"]
                mape = (abs(fp - zed) / fp.replace(0, pd.NA)).mean() * 100  # avoid division by zero
            else:
                mape = None
            records.append({"metric": metric, "state": state, "mape": mape})

    return pd.DataFrame(records)

def plot_mape_heatmap(dataframe):
    """
    Plots a heatmap showing the MAPE between ZED_COM and Force_Plate
    for each metric and state in a single dataset.

    Parameters:
        dataframe (pd.DataFrame): The input trial- or average-level data.
    """
    mape_df = compute_mape(dataframe)
    heatmap_df = mape_df.pivot(index="metric", columns="state", values="mape")

    plt.figure(figsize=(4, 5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={'label': 'MAPE (%)'}
    )
    plt.title("MAPE Heatmap of Metrics by State", fontsize=16)
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_mape_change_bars(trial_level_df, average_level_df):
    """
    Compares the MAPE between ZED_COM and Force_Plate for each metric/state
    in trial-level vs. average-level data. Displays a grouped bar plot showing
    the difference (average - trial) in MAPE.

    Parameters:
        trial_level_df (pd.DataFrame): Trial-level data.
        average_level_df (pd.DataFrame): Subject-level averaged data.
    """
    df_trial = compute_mape(trial_level_df)
    df_avg = compute_mape(average_level_df)

    merged = df_trial.merge(df_avg, on=["metric", "state"], suffixes=("_trial", "_avg"))
    merged["delta_mape"] = merged["mape_avg"] - merged["mape_trial"]

    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=merged,
        x="delta_mape",
        y="metric",
        hue="state",
        palette="Set2"
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Δ MAPE (Average - Trial) [%]")
    plt.ylabel("Metric")
    plt.title("Change in MAPE by Metric and State")

    plt.legend(
        title="State",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()
