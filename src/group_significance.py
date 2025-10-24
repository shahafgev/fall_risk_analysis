import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def check_normality(x, y, alpha=0.05):
    """Return True if both samples are normally distributed (Shapiro–Wilk)."""
    px = stats.shapiro(x).pvalue if len(x) >= 3 else 1.0
    py = stats.shapiro(y).pvalue if len(y) >= 3 else 1.0
    return (px > alpha) and (py > alpha)


def cohens_d(x, y):
    """Compute Cohen’s d for paired samples."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


def calculate_significance(df):
    results = []
    for device in df["device"].unique():
        for metric in df["metric"].unique():
            sub = df[(df["device"] == device) & (df["metric"] == metric)]

            # Merge open & closed values per participant
            merged = pd.merge(
                sub[sub["state"] == "open"][["participant name", "average_value"]],
                sub[sub["state"] == "closed"][["participant name", "average_value"]],
                on="participant name", suffixes=("_open", "_closed")
            )

            if len(merged) < 3:
                continue

            x = merged["average_value_open"]
            y = merged["average_value_closed"]

            # Normality check
            normal = check_normality(x, y)

            # Select test based on normality
            if normal:
                t_stat, p_val = stats.ttest_rel(x, y)
                test_name = "paired t-test"
            else:
                t_stat, p_val = stats.wilcoxon(x, y)
                test_name = "Wilcoxon signed-rank"

            # Store results
            results.append({
                "device": device,
                "metric": metric,
                "normal": normal,
                "mean_open": x.mean(),
                "sd_open": x.std(ddof=1),
                "mean_closed": y.mean(),
                "sd_closed": y.std(ddof=1),
                "t_or_z_stat": t_stat,
                "p_value": p_val,
                "cohen_d": cohens_d(y, x) if normal else None
            })

    results_df = pd.DataFrame(results)
    return results_df


def plot_open_closed_bars(results_df, device, exclude_metrics=None, row_order=None, alpha=0.05):
    """
    Plot open vs closed averages with SD for each metric of a given device,
    marking metrics with significant differences.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must include: ['device','metric','mean_open','sd_open','mean_closed','sd_closed','p_value']
    device : str
        Device name to plot (e.g. 'Force_Plate')
    exclude_metrics : list[str] or None
        Metrics to exclude
    row_order : list[str] or None
        Custom order of metrics to display on the x-axis
    alpha : float
        Significance threshold for marking metrics
    """

    df = results_df[results_df["device"] == device].copy()

    if exclude_metrics:
        df = df[~df["metric"].isin(exclude_metrics)]

    # Define order
    if row_order is not None:
        # Keep only those metrics present in the data
        row_order = [m for m in row_order if m in df["metric"].unique()]
        df["metric"] = pd.Categorical(df["metric"], categories=row_order, ordered=True)
        df = df.sort_values("metric")
    else:
        # Default: sort by difference
        df["diff"] = df["mean_closed"] - df["mean_open"]
        df = df.sort_values("diff", ascending=False)

    metrics = df["metric"].values
    idx = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(metrics)*0.6), 4))

    # Bars
    ax.bar(idx - width/2, df["mean_open"], width, yerr=df["sd_open"], capsize=3, label="Open")
    ax.bar(idx + width/2, df["mean_closed"], width, yerr=df["sd_closed"], capsize=3, label="Closed")

    # Significance markers
    for i, p in enumerate(df["p_value"]):
        if p < alpha:
            y_max = max(df["mean_open"].iloc[i] + df["sd_open"].iloc[i],
                        df["mean_closed"].iloc[i] + df["sd_closed"].iloc[i])
            ax.text(idx[i], y_max * 1.05, "*", ha="center", va="bottom", fontsize=14, color="red")

    ax.set_xticks(idx)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Mean (± SD)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_and_plot_significance(df, exclude_metrics=None, row_order=None, alpha=0.05):
    results_df = calculate_significance(df)

    for device in ["Force_Plate", "ZED_COM", "front", "back"]:
        print(device)
        plot_open_closed_bars(results_df, device=device, exclude_metrics=exclude_metrics, row_order=row_order)
