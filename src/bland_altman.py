import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import shapiro, pearsonr, beta
from src.assumption_utils import load_assumption_summary

def bland_altman_plot_by_state(df, metric_name, average_level=False, source_name=None):
    """
    Creates side-by-side Bland–Altman plots (eyes open vs. eyes closed)
    comparing 'Force_Plate' vs. 'ZED_COM' for the specified metric.
    Uses precomputed assumptions saved in CSVs.

    Parameters:
    - df: DataFrame with columns:
        ['participant name', 'state', 'metric', 'device', 'value'] or
        ['participant name', 'state', 'metric', 'device', 'average_value']
    - metric_name: The metric to plot.
    - average_level: If True, uses 'average_value' instead of 'value'.
    - source_name: str, one of 'oa_trial', 'oa_average', 'st_trial', 'st_average'
    """
    
    if source_name not in {'oa_trial', 'oa_average', 'st_trial', 'st_average'}:
        raise ValueError("source_name must be one of: 'oa_trial', 'oa_average', 'st_trial', 'st_average'")

    def harrell_davis_quantile(x, q):
        x = np.sort(x)
        n = len(x)
        i = np.arange(1, n + 1)
        weights = beta.cdf(i / n, (n + 1) * q, (n + 1) * (1 - q)) - beta.cdf((i - 1) / n, (n + 1) * q, (n + 1) * (1 - q))
        return np.sum(weights * x)

    unit = "cm²" if metric_name in {"Ellipse area", "ML Variance", "AP Variance"} else "cm"
    df_metric = df[(df['metric'] == metric_name) & (df['device'].isin(['Force_Plate', 'ZED_COM']))].copy()
    value_col = 'average_value' if average_level else 'value'

    if value_col not in df_metric.columns:
        raise ValueError(f"Expected column '{value_col}' not found in the dataframe.")

    states = df_metric['state'].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(states), figsize=(6 * len(states), 5), sharey=True)
    if len(states) == 1:
        axes = [axes]

    for i, st in enumerate(states):
        ax = axes[i]
        df_state = df_metric[df_metric['state'] == st]

        pivot_index = ['participant name', 'state', 'metric']
        if not average_level:
            if 'trial' in df_state.columns:
                pivot_index.insert(1, 'trial')
            else:
                raise ValueError("Expected 'trial' column for trial-level data.")

        df_pivot = df_state.pivot_table(index=pivot_index, columns='device', values=value_col).reset_index()
        df_pivot.dropna(subset=['Force_Plate', 'ZED_COM'], inplace=True)

        df_pivot['mean_measure'] = (df_pivot['Force_Plate'] + df_pivot['ZED_COM']) / 2
        df_pivot['diff_measure'] = df_pivot['Force_Plate'] - df_pivot['ZED_COM']
        mean_diff = df_pivot['diff_measure'].mean()
        sd_diff = df_pivot['diff_measure'].std()

        # Load precomputed assumption summary
        assumptions = load_assumption_summary(
            state=st, 
            metric=metric_name, 
            average_level=average_level, 
            source_name=source_name
        )

        summary_msg = assumptions.get("summary", "")
        normal_distr = assumptions.get("normality", "") == "Normal"

        if summary_msg and summary_msg != "Assumptions met":
            warnings.warn(
                f"Precomputed assumption for {metric_name} (eyes {st}): {summary_msg}. "
                f"Using non-parametric LOA or interpret with caution."
            )

        # Plot points and LOA
        ax.scatter(df_pivot['mean_measure'], df_pivot['diff_measure'], alpha=0.8)
        ax.axhline(mean_diff, color='black', linestyle='-')

        diffs = df_pivot['diff_measure'].values
        if not normal_distr:
            if not average_level and len(diffs) >= 80:
                loa_lower = harrell_davis_quantile(diffs, 0.025)
                loa_upper = harrell_davis_quantile(diffs, 0.975)
            else:
                loa_lower = np.percentile(diffs, 2.5)
                loa_upper = np.percentile(diffs, 97.5)
        else:
            loa_upper = mean_diff + 1.96 * sd_diff
            loa_lower = mean_diff - 1.96 * sd_diff

        ax.axhline(loa_upper, color='black', linestyle='--')
        ax.axhline(loa_lower, color='black', linestyle='--')

        ax.set_title(f"{metric_name} - eyes {st}")
        ax.set_xlabel(f"Mean of devices ({unit})")
        ax.set_ylabel(f"Difference between devices ({unit})")
        ax.tick_params(labelleft=True)

        ax.figure.canvas.draw()
        xlim = ax.get_xlim()
        x_text = xlim[1] - 0.02 * (xlim[1] - xlim[0])
        range_diff = df_pivot['diff_measure'].max() - df_pivot['diff_measure'].min()
        offset = 0.02 * range_diff

        ax.text(x_text, mean_diff + offset, f"Mean/Bias : {mean_diff:.2f} {unit}", ha='right', va='bottom')
        ax.text(x_text, loa_upper + offset, f"Upper LOA: {loa_upper:.2f} {unit}", ha='right', va='bottom')
        ax.text(x_text, loa_lower - offset, f"Lower LOA: {loa_lower:.2f} {unit}", ha='right', va='top')

    plt.tight_layout()
    plt.show()