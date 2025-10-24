import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# 1) Pair FP and ZED per (participant) for a given metric & state on AVERAGE-LEVEL data
def make_pairs_avg(df: pd.DataFrame, metric: str, state: str) -> pd.DataFrame:
    """
    Expects columns: ['participant name','state','device','metric','average_value']
    Returns wide table: [participant name, FP, ZED] for the given metric & state.
    """
    d = df[(df["metric"] == metric) & (df["state"].str.lower() == state.lower())]
    w = (d.pivot(index="participant name", columns="device", values="average_value")
           .rename(columns={"Force_Plate": "FP", "ZED_COM": "ZED"})
           .reset_index())
    w = w.dropna(subset=["FP","ZED"]).reset_index(drop=True)
    # ensure numeric
    w["FP"]  = pd.to_numeric(w["FP"],  errors="coerce")
    w["ZED"] = pd.to_numeric(w["ZED"], errors="coerce")
    w = w.dropna(subset=["FP","ZED"]).reset_index(drop=True)
    return w

# 2) Fit OLS for one metric-state: FP ~ ZED (ZED is x; FP is gold-standard y)
def fit_ols_for_metric_state(df_avg: pd.DataFrame, metric: str, state: str, robust: str | None = None):
    """
    robust: None (default OLS) or a heteroskedasticity-robust type like "HC3".
    Returns (results, paired_df)
    """
    w = make_pairs_avg(df_avg, metric, state)
    if len(w) < 3:
        raise ValueError(f"Not enough paired rows for metric='{metric}', state='{state}' (n={len(w)}).")
    model = smf.ols("FP ~ ZED", data=w)   # <-- flipped
    res = model.fit(cov_type=robust) if robust else model.fit()
    return res, w

# 3) Summarize one OLS fit into a row (parameters on FP scale)
def summarize_ols(metric: str, state: str, res, w) -> dict:
    ci = res.conf_int(alpha=0.05)

    # fitted params
    alpha = float(res.params["Intercept"])
    beta  = float(res.params["ZED"])

    # predictions on FP scale
    fp_hat = res.predict(w["ZED"])
    # model residual RMSE (FP vs Ŷ_FP)
    rmse_model = float(np.sqrt(np.mean((w["FP"] - fp_hat)**2)))
    # optional: raw disagreement without calibration
    rmse_raw   = float(np.sqrt(np.mean((w["FP"] - w["ZED"])**2)))

    ttest = res.t_test("ZED = 1")
    slope_eq1_p = float(ttest.pvalue)

    return {
        "metric": metric,
        "state": state,
        "intercept": alpha,
        "intercept_ci_lo": float(ci.loc["Intercept", 0]),
        "intercept_ci_hi": float(ci.loc["Intercept", 1]),
        "intercept_p": float(res.pvalues["Intercept"]),
        "slope_zed": beta,                              # slope on ZED
        "slope_ci_lo": float(ci.loc["ZED", 0]),
        "slope_ci_hi": float(ci.loc["ZED", 1]),
        "slope_p": float(res.pvalues["ZED"]),
        "slope_eq1_p": slope_eq1_p,
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "rmse_model": rmse_model,   # FP vs Ŷ_FP (after calibration)
        "rmse_raw": rmse_raw        # FP vs ZED (as-is)
    }

# 4) Run across all metrics & states; return tidy table
def run_ols_table_all(df_avg: pd.DataFrame, states=("open","closed"), robust: str | None = None) -> pd.DataFrame:
    rows = []
    metrics = sorted(df_avg["metric"].unique().tolist())
    for metric in metrics:
        for state in states:
            if not ((df_avg["metric"] == metric) & (df_avg["state"].str.lower() == state.lower())).any():
                continue
            try:
                res, w = fit_ols_for_metric_state(df_avg, metric, state, robust=robust)
                rows.append(summarize_ols(metric, state, res, w))
            except Exception as e:
                rows.append({"metric": metric, "state": state, "error": str(e)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["metric","state"], kind="stable").reset_index(drop=True)

# 5) Optional: pretty formatting
def prettify(df: pd.DataFrame, digits=3) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df2 = df.copy()

    # numeric rounding
    num_cols = ["intercept","intercept_p",
                "slope_zed","slope_p","slope_eq1_p",
                "r2","r2_adj","rmse_model","rmse_raw"]
    for c in num_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").round(digits)

    # build compact CI columns
    if {"intercept_ci_lo","intercept_ci_hi"}.issubset(df2.columns):
        df2["intercept_CI"] = df2.apply(
            lambda r: f"[{r['intercept_ci_lo']:.{digits}f}, {r['intercept_ci_hi']:.{digits}f}]",
            axis=1
        )
        df2 = df2.drop(columns=["intercept_ci_lo","intercept_ci_hi"])

    if {"slope_ci_lo","slope_ci_hi"}.issubset(df2.columns):
        df2["slope_CI"] = df2.apply(
            lambda r: f"[{r['slope_ci_lo']:.{digits}f}, {r['slope_ci_hi']:.{digits}f}]",
            axis=1
        )
        df2 = df2.drop(columns=["slope_ci_lo","slope_ci_hi"])

    # reorder columns nicely
    cols_order = ["metric","state",
                  "intercept","intercept_CI","intercept_p",
                  "slope_zed","slope_CI","slope_p","slope_eq1_p",
                  "r2","r2_adj","rmse_model","rmse_raw"]
    df2 = df2[[c for c in cols_order if c in df2.columns]]

    return df2


def plot_forest_params_both_states(
    summary_df: pd.DataFrame,
    figsize=None,
    slope_xlim=None,
    intercept_xlim=None,
    clip_quantiles=(0.02, 0.98),
    label_fontsize=10,
    offset=0.3,
    colors=None,
    exclude_metrics=None,
    row_order=None):
    """
    Forest-style plot of intercept (α) and slope (β) with 95% CIs for TWO states.

    Required columns in `summary_df`:
      ['metric','state','intercept','intercept_ci_lo','intercept_ci_hi',
       'slope_<zed_or_fp>','slope_ci_lo','slope_ci_hi']

    Notes:
    - State order is fixed to: open (top) → closed (bottom).
    - You can drop metrics via `exclude_metrics`.
    - You can set a custom metric order via `row_order` (otherwise alphabetical).
    """
    # --- fixed state order: open -> closed
    st0, st1 = "open", "closed"

    # --- copy & normalize
    df = summary_df.copy()
    df["state"] = df["state"].astype(str).str.strip().str.lower()

    # Restrict to the two states we plot
    df = df[df["state"].isin([st0, st1])]

    # --- optional exclusion of metrics
    if exclude_metrics:
        df = df[~df["metric"].isin(exclude_metrics)]

    # --- determine metric order
    all_metrics = df["metric"].dropna().unique().tolist()
    if len(all_metrics) == 0:
        raise ValueError("No metrics found in summary_df after filtering states/exclusions.")

    if row_order:
        # keep only metrics that exist; preserve given order
        metrics = [m for m in row_order if m in all_metrics]
    else:
        metrics = sorted(all_metrics)

    # --- pick slope column name robustly
    slope_col = "slope_zed" if "slope_zed" in df.columns else \
                ("slope_fp" if "slope_fp" in df.columns else None)
    if slope_col is None:
        raise ValueError("Expected 'slope_zed' or 'slope_fp' column not found.")

    # --- build per-state frames reindexed to full metric list
    d0 = (df[df["state"] == st0].set_index("metric").reindex(metrics))
    d1 = (df[df["state"] == st1].set_index("metric").reindex(metrics))

    # --- y coordinates
    y = np.arange(len(metrics))
    y0 = y - offset/2.0   # top (open)
    y1 = y + offset/2.0   # bottom (closed)

    # --- figure size
    if figsize is None:
        figsize = (14, 0.2 * len(metrics) + 1.2)

    # --- left label column width
    max_len = max(len(str(m)) for m in metrics)
    label_col = 0.35 + 0.015 * max_len

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[0.5*label_col, 1.0, 1.0], wspace=0.12)
    ax_lab = fig.add_subplot(gs[0, 0])
    ax_int = fig.add_subplot(gs[0, 1], sharey=ax_lab)
    ax_slp = fig.add_subplot(gs[0, 2], sharey=ax_lab)

    # --- label axis
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, len(metrics) - 0.5)
    for yi, name in enumerate(metrics):
        ax_lab.text(1.0, yi, str(name), ha="right", va="center", fontsize=label_fontsize)
    ax_lab.invert_yaxis()
    ax_lab.axis("off")

    # --- colors
    if colors is None:
        colors = {st0: "C0", st1: "C1"}
    c0, c1 = colors.get(st0, "C0"), colors.get(st1, "C1")

    # ===================== Intercept panel (ref = 0) =====================
    ei0, li0, hi0 = d0["intercept"], d0["intercept_ci_lo"], d0["intercept_ci_hi"]
    ei1, li1, hi1 = d1["intercept"], d1["intercept_ci_lo"], d1["intercept_ci_hi"]

    if intercept_xlim is None:
        all_i = np.r_[ei0.values, li0.values, hi0.values,
                      ei1.values, li1.values, hi1.values]
        all_i = all_i[np.isfinite(all_i)]
        if all_i.size == 0:
            intercept_xlim = (-1, 1)
        else:
            lo_q, hi_q = np.nanquantile(all_i, clip_quantiles)
            pad = 0.05 * max(hi_q - lo_q, 1.0)
            intercept_xlim = (lo_q - pad, hi_q + pad)

    m0 = np.isfinite(li0) & np.isfinite(hi0) & np.isfinite(ei0)
    m1 = np.isfinite(li1) & np.isfinite(hi1) & np.isfinite(ei1)
    ax_int.hlines(y0[m0], li0[m0], hi0[m0], color=c0, linewidth=2, label=st0.capitalize())
    ax_int.plot(ei0[m0], y0[m0], "o", color=c0)
    ax_int.hlines(y1[m1], li1[m1], hi1[m1], color=c1, linewidth=2, label=st1.capitalize())
    ax_int.plot(ei1[m1], y1[m1], "o", color=c1)

    ax_int.axvline(0, linestyle="--", linewidth=1)
    ax_int.set_title("Intercept (α)")
    ax_int.set_xlabel("Intercept")
    ax_int.set_xlim(intercept_xlim)
    ax_int.set_ylim(-0.5, len(metrics) - 0.5)
    ax_int.invert_yaxis()
    ax_int.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    for s in ("top", "right"):
        ax_int.spines[s].set_visible(False)

    # ======================= Slope panel (ref = 1) =======================
    eb0, lb0, hb0 = d0[slope_col], d0["slope_ci_lo"], d0["slope_ci_hi"]
    eb1, lb1, hb1 = d1[slope_col], d1["slope_ci_lo"], d1["slope_ci_hi"]

    if slope_xlim is None:
        all_b = np.r_[eb0.values, lb0.values, hb0.values,
                      eb1.values, lb1.values, hb1.values]
        all_b = all_b[np.isfinite(all_b)]
        if all_b.size == 0:
            slope_xlim = (0, 2)
        else:
            lo_q, hi_q = np.nanquantile(all_b, clip_quantiles)
            pad = 0.05 * max(hi_q - lo_q, 1.0)
            slope_xlim = (lo_q - pad, hi_q + pad)

    m0 = np.isfinite(lb0) & np.isfinite(hb0) & np.isfinite(eb0)
    m1 = np.isfinite(lb1) & np.isfinite(hb1) & np.isfinite(eb1)
    ax_slp.hlines(y0[m0], lb0[m0], hb0[m0], color=c0, linewidth=2)
    ax_slp.plot(eb0[m0], y0[m0], "o", color=c0)
    ax_slp.hlines(y1[m1], lb1[m1], hb1[m1], color=c1, linewidth=2)
    ax_slp.plot(eb1[m1], y1[m1], "o", color=c1)

    ax_slp.axvline(1, linestyle="--", linewidth=1)
    ax_slp.set_title("Slope (β)")
    ax_slp.set_xlabel("Slope (β)")
    ax_slp.set_xlim(slope_xlim)
    ax_slp.set_ylim(-0.5, len(metrics) - 0.5)
    ax_slp.invert_yaxis()
    ax_slp.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    for s in ("top", "right", "left"):
        ax_slp.spines[s].set_visible(False)
    ax_slp.set_yticks([])

    # --- Row-separator grid lines at metric boundaries
    row_edges = np.arange(-0.5, len(metrics), 1.0)
    for ax in (ax_int, ax_slp):
        ax.set_yticks(row_edges, minor=True)
        ax.grid(which="minor", axis="y", linestyle="-", linewidth=0.5, alpha=0.3)

    # --- Layout & legend (outside, right of slope panel)
    plt.subplots_adjust(right=0.84)
    handles, labels = ax_int.get_legend_handles_labels()
    ax_slp.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False, borderaxespad=0.0
    )

    plt.show()


def plot_rmse_by_metric(table_ols: pd.DataFrame,
                        rmse_col="rmse_model",            # "rmse_model" or "rmse_raw"
                        states=("open","closed"),
                        exclude_metrics=None,
                        sort=True,                        # sort by max RMSE across states
                        descending=True,
                        alphabetical=False,               # ignore sorting and go A→Z
                        figsize=(8, 5),
                        bar_height=0.35,
                        annotate=True,
                        xlim=None,
                        label_fontsize=10):
    """
    Horizontal grouped bar plot of RMSE for each metric.
    - rmse_col: "rmse_model" (FP vs Ŷ_FP) or "rmse_raw" (FP vs ZED)
    - metrics appear on the y-axis (always)
    """
    if rmse_col not in table_ols.columns:
        raise ValueError(f"{rmse_col!r} not in table_ols")

    df = table_ols.copy()
    df["state"] = df["state"].str.lower()
    keep_states = [s.lower() for s in states]
    df = df[df["state"].isin(keep_states)]

    if exclude_metrics:
        df = df[~df["metric"].isin(exclude_metrics)]

    # wide = metrics x states for the requested RMSE column
    wide = df.pivot(index="metric", columns="state", values=rmse_col)

    # ensure both state columns exist (in requested order)
    for st in keep_states:
        if st not in wide.columns:
            wide[st] = np.nan
    wide = wide[keep_states]

    # order metrics
    if alphabetical:
        order = sorted(wide.index.tolist())
    elif sort:
        order = wide.max(axis=1).sort_values(ascending=not descending).index.tolist()
    else:
        order = list(wide.index)
    wide = wide.loc[order]

    metrics = wide.index.tolist()
    n = len(metrics)
    y_base = np.arange(n)
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    # bars for each state
    for i, st in enumerate(keep_states):
        vals = wide[st].values
        ypos = y_base - (offset if i == 0 else -offset)
        ax.barh(ypos, vals, height=bar_height, label=st.capitalize())
        if annotate:
            for (y, v) in zip(ypos, vals):
                if pd.notna(v):
                    ax.text(v + 0.02, y, f"{v:.3f}", va="center", ha="left", fontsize=8)

    title = "RMSE (Calibrated): FP vs Ŷ_FP" if rmse_col == "rmse_model" else "RMSE (Raw): FP vs ZED"
    ax.set_title(title)
    ax.set_xlabel("RMSE")
    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=label_fontsize)  # <-- metric names here
    ax.invert_yaxis()                                     # largest at top (optional)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    if xlim:
        ax.set_xlim(xlim)

    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def plot_pct_delta_rmse(table_ols: pd.DataFrame,
                        states=("open","closed"),
                        exclude_metrics=None,
                        sort=True, descending=True, alphabetical=False,
                        figsize=(8, 5), bar_height=0.35,
                        annotate=True, label_fontsize=10, xlim=None):
    """
    Horizontal grouped bars of % change in RMSE:
      %ΔRMSE = 100 * (rmse_raw - rmse_model)/rmse_raw
    Positive = calibration reduces RMSE (improvement).
    """
    df = table_ols.copy()
    df["state"] = df["state"].str.lower()
    keep_states = [s.lower() for s in states]
    df = df[df["state"].isin(keep_states)]
    if exclude_metrics:
        df = df[~df["metric"].isin(exclude_metrics)]

    # ---- compute %ΔRMSE ----
    num = df["rmse_raw"] - df["rmse_model"]
    den = df["rmse_raw"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pct_delta_rmse"] = 100.0 * (num / den)
    df.loc[(~np.isfinite(den)) | (den == 0), "pct_delta_rmse"] = np.nan

    # wide format: metrics x states
    wide = df.pivot(index="metric", columns="state", values="pct_delta_rmse")

    # ensure both states exist
    for st in keep_states:
        if st not in wide.columns:
            wide[st] = np.nan
    wide = wide[keep_states]

    # ---- ordering ----
    if alphabetical:
        order = sorted(wide.index.tolist())
    elif sort:
        order = wide.max(axis=1).sort_values(ascending=not descending).index.tolist()
    else:
        order = list(wide.index)
    wide = wide.loc[order]

    metrics = wide.index.tolist()
    y_base = np.arange(len(metrics))
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    for i, st in enumerate(keep_states):
        vals = wide[st].values
        ypos = y_base - (offset if i == 0 else -offset)
        ax.barh(ypos, vals, height=bar_height, label=st.capitalize())
        if annotate:
            for (y, v) in zip(ypos, vals):
                if np.isfinite(v):
                    ax.text(v + 1, y, f"{v:.1f}%", va="center", ha="left", fontsize=9)

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("% Change in RMSE (Raw → Calibrated)")
    ax.set_xlabel("Δ RMSE (%)  (positive = improvement)")
    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=label_fontsize)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    if xlim:
        ax.set_xlim(xlim)

    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def plot_delta_rmse(table_ols: pd.DataFrame,
                    states=("open","closed"),
                    exclude_metrics=None,
                    figsize=(8,5),
                    bar_height=0.35,
                    annotate=True,
                    sort=True,
                    descending=False,
                    label_fontsize=10):
    """
    Plot the change in RMSE (raw - model) for each metric and state.
    Positive = calibration reduced RMSE.
    """

    df = table_ols.copy()
    df["state"] = df["state"].str.lower()
    
    if exclude_metrics:
        df = df[~df["metric"].isin(exclude_metrics)]
    
    df["delta_rmse"] = df["rmse_raw"] - df["rmse_model"]

    # pivot wide to align states
    wide = df.pivot(index="metric", columns="state", values="delta_rmse")

    # order metrics by max delta_rmse (optional)
    if sort:
        order = wide.max(axis=1).sort_values(ascending=not descending).index.tolist()
        wide = wide.loc[order]

    metrics = wide.index.tolist()
    n = len(metrics)
    y_base = np.arange(n)
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    for i, st in enumerate(states):
        vals = wide[st.lower()].values
        ypos = y_base - (offset if i == 0 else -offset)
        ax.barh(ypos, vals, height=bar_height, label=st.capitalize())
        if annotate:
            for (y, v) in zip(ypos, vals):
                if pd.notna(v):
                    ax.text(v + (0.02 if v >= 0 else -0.02), y,
                            f"{v:.3f}", va="center",
                            ha="left" if v >= 0 else "right", fontsize=8)

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Change in RMSE (Raw - Model)")
    ax.set_xlabel("Δ RMSE (positive = improvement)")
    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=label_fontsize)
    ax.legend(loc="best")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()


def rmse_comparison(table_ols: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy DataFrame comparing RMSE values.

    Output columns:
    - metric
    - state
    - rmse_raw      : baseline RMSE (FP vs ZED)
    - rmse_model    : calibrated RMSE (FP vs Ŷ_FP)
    - diff          : rmse_raw - rmse_model
    - pct_change    : 100 * (diff / rmse_raw)
    """
    df = table_ols.copy()
    if not {"rmse_raw", "rmse_model"}.issubset(df.columns):
        raise ValueError("table_ols must contain 'rmse_raw' and 'rmse_model' columns")

    # difference and percent change
    df["diff"] = df["rmse_raw"] - df["rmse_model"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pct_change"] = 100 * df["diff"] / df["rmse_raw"]
    df.loc[~np.isfinite(df["pct_change"]), "pct_change"] = np.nan

    # keep only relevant cols
    out = df[["metric", "state", "rmse_raw", "rmse_model", "diff", "pct_change"]].copy()
    return out.sort_values(["metric", "state"], kind="stable").round(3).sort_values(["pct_change"], ascending=False).reset_index(drop=True)


def plot_r2(table_ols: pd.DataFrame,
                states=("open","closed"),
                sort=True,
                descending=True,
                figsize=(8, 5),
                bar_height=0.35,
                annotate=True,
                label_fontsize=9):
    """
    Plot R² by metric × state.
    """
    df = table_ols.copy()
    df["state"] = df["state"].str.lower()
    keep_states = [s.lower() for s in states]
    df = df[df["state"].isin(keep_states)]

    # pivot table: metric × state
    wide = df.pivot(index="metric", columns="state", values="r2")

    # keep all states
    for st in keep_states:
        if st not in wide.columns:
            wide[st] = np.nan
    wide = wide[keep_states]

    # sorting
    if sort:
        order = wide.max(axis=1).sort_values(ascending=not descending).index.tolist()
    else:
        order = wide.index.tolist()
    wide = wide.loc[order]

    metrics = wide.index.tolist()
    n = len(metrics)
    y_base = np.arange(n)
    offset = bar_height / 2

    fig, ax = plt.subplots(figsize=figsize)

    for i, st in enumerate(keep_states):
        vals = wide[st].values
        ypos = y_base - (offset if i == 0 else -offset)
        ax.barh(ypos, vals, height=bar_height, label=st.capitalize())
        if annotate:
            for (y, v) in zip(ypos, vals):
                if pd.notna(v):
                    ax.text(v + 0.02, y, f"{v:.2f}", va="center", ha="left", fontsize=8)

    ax.set_title("R² by Metric and State")
    ax.set_xlabel("R²")
    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=label_fontsize)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def plot_regression_and_residuals(df_avg: pd.DataFrame, metric: str, state: str, robust: str | None = None):
    """
    Scatter plot (ZED vs FP) with regression line and annotations,
    plus residuals plot.
    """
    # Fit regression
    res, w = fit_ols_for_metric_state(df_avg, metric, state, robust=robust)

    # Extract stats
    intercept = res.params["Intercept"]
    slope = res.params["ZED"]
    r2 = res.rsquared

    # Predictions
    w["FP_hat"] = res.predict(w["ZED"])
    residuals = w["FP"] - w["FP_hat"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- (1) Scatter + regression line ---
    ax = axes[0]
    ax.scatter(w["ZED"], w["FP"], alpha=0.7, label="Observed")
    # regression line
    x_vals = np.linspace(w["ZED"].min(), w["ZED"].max(), 100)
    y_hat = intercept + slope * x_vals
    ax.plot(x_vals, y_hat, color="red", label="Regression line")
    # identity line
    ax.plot(x_vals, x_vals, color="black", linestyle="--", label="y=x")

    ax.set_title(f"{metric} - {state.capitalize()}")
    ax.set_xlabel("ZED")
    ax.set_ylabel("Force Plate")
    ax.legend()

    # annotate with R², slope, intercept
    txt = f"$R^2$ = {r2:.3f}\nIntercept = {intercept:.3f}\nSlope = {slope:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # --- (2) Residuals plot ---
    ax = axes[1]
    ax.scatter(w["FP_hat"], residuals, alpha=0.7)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted FP (Ŷ)")
    ax.set_ylabel("Residuals (FP − Ŷ)")

    plt.tight_layout()
    plt.show()