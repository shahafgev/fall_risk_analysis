import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def compute_correlation(dataframe, method="pearson", device="ZED_COM"):
    """
    Computes the association between device and Force_Plate for each unique (metric, state).
    
    Behavior:
    - If the data are trial-level (participants have >1 trial), computes repeated-measures
      correlation (rmcorr): Pearson correlation on within-participant centered values.
      If method == "spearman", applies rank-transform first (i.e., Spearman-like rmcorr).
    - If the data are average-level (one value per participant), computes standard
      Pearson/Spearman correlation on the pooled rows.

    Returns:
        pd.DataFrame with columns: ['metric', 'state', 'correlation']
    """
    df = dataframe.copy()

    # Standardize value column name
    if "value" not in df.columns and "average_value" in df.columns:
        df["value"] = df["average_value"]

    # Ensure 'trial' exists (helps detection, but we won't force trial-level if all trials==1)
    if "trial" not in df.columns:
        df["trial"] = 1

    # Detect trial-level: at least one participant has >1 trial
    trial_counts = df.groupby("participant name")["trial"].nunique()
    trial_level = (trial_counts.max() if len(trial_counts) else 0) > 1

    def filter_and_pivot(df_in, metric, state, trial_level_flag):
        """Filter by metric/state and pivot to wide with FP/ZED columns."""
        d = df_in[
            (df_in["metric"] == metric) &
            (df_in["state"].str.contains(state, case=False))
        ].copy()
        if d.empty:
            return pd.DataFrame()

        if trial_level_flag:
            idx_cols = ["participant name", "state", "trial"]
        else:
            idx_cols = ["participant name", "state"]

        wide = (d.pivot(index=idx_cols, columns="device", values="value")
                  .reset_index())
        wide.columns.name = None
        # keep only rows with both devices present
        if not {"Force_Plate", device}.issubset(set(wide.columns)):
            return pd.DataFrame()
        wide = wide.dropna(subset=["Force_Plate", device])
        return wide

    metrics = sorted(df["metric"].unique())
    states  = sorted(df["state"].unique())

    records = []

    for metric in metrics:
        for state in states:
            wide = filter_and_pivot(df, metric, state, trial_level_flag=trial_level)
            if wide.empty:
                records.append({"metric": metric, "state": state, "correlation": None})
                continue

            fp = "Force_Plate"
            zc = device

            if trial_level:
                # -------- Repeated-measures correlation --------
                # Optionally rank-transform (Spearman-like rmcorr)
                if method.lower() == "spearman":
                    wide[fp] = wide[fp].rank(method="average")
                    wide[zc] = wide[zc].rank(method="average")

                # Within-participant centering
                grp = wide.groupby("participant name")
                wide["_fp_c"] = wide[fp] - grp[fp].transform("mean")
                wide["_zc_c"] = wide[zc] - grp[zc].transform("mean")

                # Pearson on centered values
                if len(wide) >= 2 and wide["_fp_c"].std(ddof=1) > 0 and wide["_zc_c"].std(ddof=1) > 0:
                    r, _ = stats.pearsonr(wide["_fp_c"], wide["_zc_c"])
                else:
                    r = None

                records.append({"metric": metric, "state": state, "correlation": r})

            else:
                # -------- Standard correlation (average-level) --------
                x = wide[fp]
                y = wide[zc]
                if method.lower() == "spearman":
                    r, _ = stats.spearmanr(x, y)
                else:
                    r, _ = stats.pearsonr(x, y)
                records.append({"metric": metric, "state": state, "correlation": r})

    return pd.DataFrame(records)


def plot_corrleations_heatmap(dataframe, method="pearson", device="ZED_COM", exclude_metrics=None,row_order=None):
    """
    Plots a heatmap showing the correlation between device and Force_Plate
    for each metric and state in a single dataset.

    Parameters:
        dataframe (pd.DataFrame): The input trial- or average-level data.
        method (str): Correlation method ('pearson', 'spearman', etc.).
        exclude_metrics (list): List of metric names to ignore in plotting.
        row_order (list): Custom order for metrics (rows).
    """
    # Get correlation values
    corr_df = compute_correlation(dataframe, method=method, device=device)

    # Exclude unwanted metrics if specified
    if exclude_metrics is not None:
        corr_df = corr_df[~corr_df["metric"].isin(exclude_metrics)]

    # Reformat for heatmap
    heatmap_df = corr_df.pivot(index="metric", columns="state", values="correlation")

    # Always set column order: open → closed
    heatmap_df = heatmap_df.reindex(columns=["open", "closed"])

    # Apply custom row order if provided
    if row_order is not None:
        heatmap_df = heatmap_df.reindex(row_order)

    # Plot heatmap
    plt.figure(figsize=(5, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0, vmax=1,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Correlation Heatmap of Metrics by State", fontsize=16)
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_state_difference_in_correlations(dataframe, method="pearson", device="ZED_COM", annotate=False, exclude_metrics=None, row_order=None):
    """
    Plots a horizontal bar chart of Δ correlation = (Closed − Open)
    for each metric, using Pearson or Spearman correlations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input trial- or average-level data.
    method : {"pearson","spearman"}, default "pearson"
        Correlation method for compute_correlation(..., method=).
    annotate : bool, default False
        If True, writes Δ values at the end of each bar.
    exclude_metrics : list[str] | None
        Metrics to remove before plotting.
    row_order : list[str] | None
        Custom y-axis order. If None, alphabetical.
    """
    # Compute & clean
    corr_df = compute_correlation(dataframe, device=device, method=method).copy()
    if exclude_metrics:
        corr_df = corr_df[~corr_df["metric"].isin(exclude_metrics)]

    # Normalize state labels and pivot
    corr_df["state"] = corr_df["state"].str.strip().str.lower()
    heatmap_df = corr_df.pivot(index="metric", columns="state", values="correlation")

    # Enforce fixed state order (must have both)
    heatmap_df = heatmap_df.reindex(columns=["open", "closed"])
    if "open" not in heatmap_df or "closed" not in heatmap_df:
        raise ValueError("Both 'open' and 'closed' states are required in the data.")

    # Δ = Closed − Open
    delta_df = (heatmap_df["closed"] - heatmap_df["open"]).reset_index()
    delta_df.columns = ["metric", "delta_corr"]

    # Metric order
    order = row_order if row_order else sorted(delta_df["metric"].unique().tolist())

    # Plot
    plt.figure(figsize=(4, 6))
    ax = sns.barplot(
        data=delta_df, x="delta_corr", y="metric",
        order=order, orient="h", color="steelblue"
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ Correlation (Closed − Open)")
    ax.set_ylabel("Metric")
    ax.set_title(f"Change in {method.capitalize()} Correlation Between States")

    if annotate:
        for p in ax.patches:
            val = p.get_width()
            if np.isfinite(val):
                ax.text(
                    p.get_x() + val + (0.01 if val >= 0 else -0.01),
                    p.get_y() + p.get_height() / 2,
                    f"{val:.2f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8
                )

    plt.tight_layout()
    plt.show()


def plot_correlation_change_bars(trial_level_df, average_level_df, method="pearson", device="ZED_COM", exclude_metrics=None, row_order=None, annotate=False):
    """
    Compares ZED_COM vs. Force_Plate correlation by metric/state
    in trial-level vs. average-level data. Shows grouped bars for
    Δ correlation = (average - trial).

    Parameters
    ----------
    trial_level_df : pd.DataFrame
        Trial-level data.
    average_level_df : pd.DataFrame
        Subject-level averaged data.
    method : {"pearson","spearman"}, default "pearson"
        Correlation method for compute_correlation(..., method=).
    exclude_metrics : list[str] | None
        Metrics to remove before plotting.
    row_order : list[str] | None
        Custom y-axis order. If None, alphabetical.
    annotate : bool, default False
        If True, shows Δ values at the end of each bar.
    """
    # Compute correlations
    df_trial = compute_correlation(trial_level_df, method=method, device=device).copy()
    df_avg   = compute_correlation(average_level_df,   method=method, device=device).copy()

    # Normalize states
    for df in (df_trial, df_avg):
        df["state"] = df["state"].str.strip().str.lower()

    # Exclude metrics if requested
    if exclude_metrics:
        df_trial = df_trial[~df_trial["metric"].isin(exclude_metrics)]
        df_avg   = df_avg[~df_avg["metric"].isin(exclude_metrics)]

    # Merge & compute Δ
    merged = df_trial.merge(df_avg, on=["metric", "state"], suffixes=("_trial", "_avg"))
    merged["delta_corr"] = merged["correlation_avg"] - merged["correlation_trial"]

    # Keep only the two fixed states, enforce their order in the legend
    merged = merged[merged["state"].isin(["open", "closed"])]

    # Metric order
    y_order = row_order if row_order else sorted(merged["metric"].unique().tolist())

    # Plot
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        data=merged,
        x="delta_corr",
        y="metric",
        hue="state",
        hue_order=["open", "closed"],  # fixed order
        order=y_order,
        palette="Set2"
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Δ Correlation (Average − Trial)")
    plt.ylabel("Metric")
    plt.title(f"Change in {method.capitalize()} Correlation by Metric and State")

    # Legend outside
    plt.legend(
        title="State",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )

    # Optional annotations
    if annotate:
        for p in ax.patches:
            val = p.get_width()
            if np.isfinite(val):
                ax.text(
                    p.get_x() + val + (0.01 if val >= 0 else -0.01),
                    p.get_y() + p.get_height() / 2,
                    f"{val:.2f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8
                )

    plt.tight_layout()
    plt.show()