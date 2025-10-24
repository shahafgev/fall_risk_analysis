# src/linear_mixed_models.py
# ------------------------------------------------------------
# Linear Mixed Models (random intercept) in R (lme4/lmerTest)
# Return a tidy summary per (metric, state).
#
# Key outputs:
#   - intercept, slope_zed, 95% Wald CIs
#   - intercept_p (H0: α=0), slope_p1 (Wald H0: β=1)
#   - r2_marginal, r2_conditional (via MuMIn::r.squaredGLMM)
#   - rmse_lmm (from LMM residuals), rmse_raw (baseline vs mean(FP))
#   - is_singular
#   - zed_centered (flag), zed_mean (for back-transform/reporting)
#
# Centering:
#   - If center_predictor=True, ZED is grand-mean centered within each
#     (metric, state) block before fitting, so the intercept tests average
#     additive bias at a typical ZED level.
# ------------------------------------------------------------

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Dict

import numpy as np
import pandas as pd

# --- Point to your R install (Windows-friendly defaults) ---
R_HOME = os.environ.get("R_HOME", r"C:\\Program Files\\R\\R-4.5.1")
R_BIN64 = os.path.join(R_HOME, "bin", "x64")
if os.path.isdir(R_BIN64):
    os.environ["PATH"] = R_BIN64 + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(R_BIN64)  # type: ignore[attr-defined]
    except Exception:
        pass
else:
    R_BIN = os.path.join(R_HOME, "bin")
    if os.path.isdir(R_BIN):
        os.environ["PATH"] = R_BIN + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(R_BIN)  # type: ignore[attr-defined]
        except Exception:
            pass

# Optional: surface user R library
os.environ.setdefault("R_LIBS_USER", r"C:\\Users\\User\\AppData\\Local\\R\\win-library\\4.5")

# rpy2 imports
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError

# ---------------------------
# R setup
# ---------------------------
os.environ.setdefault("R_HOME", r"C:\\Program Files\\R\\R-4.5.1")
os.environ.setdefault("R_LIBS_USER", r"C:\\Users\\User\\AppData\\Local\\R\\win-library\\4.5")

_r = robjects.r
_r("""
ulib <- Sys.getenv('R_LIBS_USER')
if (!nzchar(ulib)) {
  ulib <- file.path(Sys.getenv('USERPROFILE'), 'AppData/Local/R/win-library/4.5')
}
dir.create(ulib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(ulib, .libPaths()))
""")

# Ensure required packages are available (includes MuMIn for R^2)
_r("""
needed <- c('lme4','lmerTest','broom.mixed','MuMIn')
to_install <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]
if (length(to_install)) {
  install.packages(to_install, repos = 'https://cloud.r-project.org', lib = .libPaths()[1])
}
""")

# Import packages (will raise if still missing)
utils = importr("utils")
lme4 = importr("lme4")
lmerTest = importr("lmerTest")
broom_mixed = importr("broom.mixed")
MuMIn = importr("MuMIn")

# ---------------------------
# Data helpers (Python side)
# ---------------------------

def _make_pairs_trial(
    df_trial: pd.DataFrame,
    metric: str,
    state: str,
    *,
    device_col: str = "device",
    participant_col: str = "participant name",
    value_col: str = "value",
    trial_col: str = "trial",
    state_col: str = "state",
    zed_label: str = "ZED_COM",
    fp_label: str = "Force_Plate",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Filter to (metric, state), pivot to paired FP/ZED per participant & trial.
    Returns columns: [participant, state, trial, FP, ZED]
    """
    df = df_trial[(df_trial["metric"] == metric) & (df_trial[state_col] == state)].copy()
    if df.empty:
        return pd.DataFrame(columns=[participant_col, state_col, trial_col, "FP", "ZED"])

    pivot = (
        df.pivot_table(
            index=[participant_col, state_col, trial_col],
            columns=device_col,
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    rename_map = {}
    if fp_label in pivot.columns:
        rename_map[fp_label] = "FP"
    if zed_label in pivot.columns:
        rename_map[zed_label] = "ZED"
    pivot = pivot.rename(columns=rename_map)

    pivot = pivot.dropna(subset=["FP", "ZED"])
    for col in ["FP", "ZED"]:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
    pivot = pivot.dropna(subset=["FP", "ZED"])

    return pivot[[participant_col, state_col, trial_col, "FP", "ZED"]].copy()


def _rmse_raw_only(paired_df: pd.DataFrame) -> float:
    """
    Baseline RMSE of FP vs mean(FP). Used just for a scale reference.
    """
    if paired_df.empty:
        return float("nan")
    y = paired_df["FP"].values
    return float(np.sqrt(np.mean((y - y.mean()) ** 2)))


# ---------------------------
# Core R-fitting function
# ---------------------------

def _fit_lmm_in_r(
    paired_df: pd.DataFrame,
    participant_col: str,
    include_ci: bool,
    alpha: float,
    include_p: bool,
    random_slope: bool,
    reml: bool,
    debug: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Fit random-intercept LMM in R:
      FP ~ ZED + (1|participant)  [or (1 + ZED | participant) if random_slope=True]
    Return fixed effects, CIs, p-values, RMSE, R^2, singular flag.
    """
    if paired_df.empty:
        return {
            "intercept": np.nan,
            "slope_zed": np.nan,
            "intercept_ci_lo": np.nan,
            "intercept_ci_hi": np.nan,
            "slope_ci_lo": np.nan,
            "slope_ci_hi": np.nan,
            "intercept_p": np.nan,
            "slope_p1": np.nan,
            "rmse_lmm": np.nan,
            "r2_marginal": np.nan,
            "r2_conditional": np.nan,
            "is_singular": np.nan,
        }

    df_r = paired_df[[participant_col, "FP", "ZED"]].rename(
        columns={participant_col: "participant"}
    ).copy()

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df_r)

    re_form = "(1 + ZED | participant)" if random_slope else "(1 | participant)"
    formula_str = f"FP ~ ZED + {re_form}"

    _r.assign("df_in", r_df)
    _r.assign("formula_str", formula_str)
    _r.assign("use_reml", bool(reml))
    _r.assign("want_ci", bool(include_ci))
    _r.assign("alpha_level", float(alpha))
    _r.assign("want_p", bool(include_p))

    r_code = """
    suppressPackageStartupMessages({
      library(lmerTest)
      library(broom.mixed)
      library(MuMIn)
    })

    fit <- lmer(as.formula(formula_str), data = df_in, REML = use_reml)

    if (want_ci) {
      tt <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE,
                              conf.level = 1 - alpha_level, conf.method = "Wald")
    } else {
      tt <- broom.mixed::tidy(fit, effects = "fixed", conf.int = FALSE)
      tt$conf.low  <- NA_real_
      tt$conf.high <- NA_real_
    }

    tt <- tt[tt$term %in% c("(Intercept)", "ZED"), , drop=FALSE]
    if (!"p.value" %in% names(tt)) tt$p.value <- NA_real_
    if (!"df" %in% names(tt))      tt$df      <- NA_real_

    est_intercept <- if (nrow(tt[tt$term=="(Intercept)",])==1) tt$estimate[tt$term=="(Intercept)"] else NA_real_
    ci_int_lo     <- if (nrow(tt[tt$term=="(Intercept)",])==1) tt$conf.low[tt$term=="(Intercept)"] else NA_real_
    ci_int_hi     <- if (nrow(tt[tt$term=="(Intercept)",])==1) tt$conf.high[tt$term=="(Intercept)"] else NA_real_
    p_int_0       <- if (nrow(tt[tt$term=="(Intercept)",])==1) tt$p.value[tt$term=="(Intercept)"] else NA_real_

    est_slope     <- if (nrow(tt[tt$term=="ZED",])==1) tt$estimate[tt$term=="ZED"] else NA_real_
    se_slope      <- if (nrow(tt[tt$term=="ZED",])==1) tt$std.error[tt$term=="ZED"] else NA_real_
    ci_slope_lo   <- if (nrow(tt[tt$term=="ZED",])==1) tt$conf.low[tt$term=="ZED"] else NA_real_
    ci_slope_hi   <- if (nrow(tt[tt$term=="ZED",])==1) tt$conf.high[tt$term=="ZED"] else NA_real_
    df_slope      <- if (nrow(tt[tt$term=="ZED",])==1) tt$df[tt$term=="ZED"] else NA_real_

    # Wald test vs 1 using Satterthwaite df when available
    p_slope_1 <- NA_real_
    if (is.finite(est_slope) && is.finite(se_slope) && se_slope > 0) {
      t1 <- (est_slope - 1.0) / se_slope
      if (is.finite(df_slope) && df_slope > 0) {
        p_slope_1 <- 2 * (1 - pt(abs(t1), df = df_slope))
      } else {
        p_slope_1 <- 2 * (1 - pnorm(abs(t1)))
      }
    }

    # LMM residual RMSE (response scale)
    rmse_lmm <- NA_real_
    resid_cond <- tryCatch(resid(fit), error = function(e) NULL)
    if (!is.null(resid_cond)) rmse_lmm <- sqrt(mean(resid_cond^2, na.rm = TRUE))

    # Marginal / Conditional R^2 (MuMIn)
    r2_marginal <- NA_real_
    r2_conditional <- NA_real_
    rs <- tryCatch(MuMIn::r.squaredGLMM(fit), error = function(e) NULL)
    if (!is.null(rs)) {
      r2_marginal    <- suppressWarnings(as.numeric(rs[1, "R2m"]))
      r2_conditional <- suppressWarnings(as.numeric(rs[1, "R2c"]))
    }

    is_singular <- tryCatch(lme4::isSingular(fit, tol = 1e-5), error = function(e) NA)

    out <- list(
      intercept       = est_intercept,
      slope_zed       = est_slope,
      intercept_ci_lo = ci_int_lo,
      intercept_ci_hi = ci_int_hi,
      slope_ci_lo     = ci_slope_lo,
      slope_ci_hi     = ci_slope_hi,
      intercept_p     = p_int_0,   # H0: alpha = 0
      slope_p1        = p_slope_1, # H0: beta  = 1
      rmse_lmm        = rmse_lmm,
      r2_marginal     = r2_marginal,
      r2_conditional  = r2_conditional,
      is_singular     = as.logical(is_singular)
    )
    out
    """

    try:
        res = _r(r_code)
    except RRuntimeError:
        # Return NA row on fit failure
        return {
            "intercept": np.nan,
            "slope_zed": np.nan,
            "intercept_ci_lo": np.nan,
            "intercept_ci_hi": np.nan,
            "slope_ci_lo": np.nan,
            "slope_ci_hi": np.nan,
            "intercept_p": np.nan,
            "slope_p1": np.nan,
            "rmse_lmm": np.nan,
            "r2_marginal": np.nan,
            "r2_conditional": np.nan,
            "is_singular": np.nan,
        }

    out = {k: (float(res.rx2(k)[0]) if len(res.rx2(k)) else np.nan) for k in res.names}
    # preserve logical for singular if possible
    try:
        is_sing_vec = res.rx2("is_singular")
        if len(is_sing_vec):
            out["is_singular"] = bool(is_sing_vec[0])
    except Exception:
        pass

    return out


# ---------------------------
# Public API
# ---------------------------

def run_lmm_table_all_r(
    df_trial: pd.DataFrame,
    *,
    states: Iterable[str] = ("open", "closed"),
    metrics: Optional[Iterable[str]] = None,
    device_col: str = "device",
    participant_col: str = "participant name",
    value_col: str = "value",
    trial_col: str = "trial",
    state_col: str = "state",
    zed_label: str = "ZED_COM",
    fp_label: str = "Force_Plate",
    random_slope: bool = False,
    reml: bool = True,
    include_ci: bool = False,
    alpha: float = 0.05,
    include_p: bool = True,
    center_predictor: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    For each (metric, state):
      1) Pair FP/ZED per participant×trial.
      2) Optionally center ZED so intercept reflects average bias at typical ZED.
      3) Fit R lmer model with random intercept (default).
      4) Collect fixed effects, CIs, intercept p, slope p vs 1, LMM RMSE, R^2.

    Returns columns:
      ['metric','state',
       'intercept','slope_zed',
       'intercept_ci_lo','intercept_ci_hi','slope_ci_lo','slope_ci_hi',
       'intercept_p','slope_p1',
       'r2_marginal','r2_conditional',
       'rmse_lmm','rmse_raw',
       'is_singular','zed_centered','zed_mean']
    """
    if metrics is None:
        metrics = list(df_trial["metric"].dropna().unique())

    rows: List[Dict] = []
    for metric in metrics:
        for state in states:
            paired = _make_pairs_trial(
                df_trial, metric, state,
                device_col=device_col,
                participant_col=participant_col,
                value_col=value_col,
                trial_col=trial_col,
                state_col=state_col,
                zed_label=zed_label,
                fp_label=fp_label,
                debug=debug,
            )

            zed_centered_flag = False
            zed_mean = np.nan
            if center_predictor and not paired.empty:
                zed_mean = float(paired["ZED"].mean())
                paired = paired.copy()
                paired["ZED"] = paired["ZED"] - zed_mean
                zed_centered_flag = True

            rmse_raw = _rmse_raw_only(paired)

            if paired.empty:
                rows.append({
                    "metric": metric, "state": state,
                    "intercept": np.nan, "slope_zed": np.nan,
                    "intercept_ci_lo": np.nan, "intercept_ci_hi": np.nan,
                    "slope_ci_lo": np.nan, "slope_ci_hi": np.nan,
                    "intercept_p": np.nan, "slope_p1": np.nan,
                    "r2_marginal": np.nan, "r2_conditional": np.nan,
                    "rmse_lmm": np.nan, "rmse_raw": rmse_raw,
                    "is_singular": np.nan,
                    "zed_centered": zed_centered_flag, "zed_mean": zed_mean,
                })
                continue

            fit_res = _fit_lmm_in_r(
                paired_df=paired,
                participant_col=participant_col,
                include_ci=include_ci,
                alpha=alpha,
                include_p=include_p,
                random_slope=random_slope,
                reml=reml,
                debug=debug,
            )

            row = {
                "metric": metric,
                "state": state,
                "intercept": fit_res.get("intercept", np.nan),
                "slope_zed": fit_res.get("slope_zed", np.nan),
                "intercept_ci_lo": fit_res.get("intercept_ci_lo", np.nan),
                "intercept_ci_hi": fit_res.get("intercept_ci_hi", np.nan),
                "slope_ci_lo": fit_res.get("slope_ci_lo", np.nan),
                "slope_ci_hi": fit_res.get("slope_ci_hi", np.nan),
                "intercept_p": fit_res.get("intercept_p", np.nan),
                "slope_p1": fit_res.get("slope_p1", np.nan),
                "r2_marginal": fit_res.get("r2_marginal", np.nan),
                "r2_conditional": fit_res.get("r2_conditional", np.nan),
                "rmse_lmm": fit_res.get("rmse_lmm", np.nan),
                "rmse_raw": rmse_raw,
                "is_singular": fit_res.get("is_singular", np.nan),
                "zed_centered": zed_centered_flag,
                "zed_mean": zed_mean,
            }
            rows.append(row)

    cols = [
        "metric","state",
        "intercept","slope_zed",
        "intercept_ci_lo","intercept_ci_hi",
        "slope_ci_lo","slope_ci_hi",
        "intercept_p","slope_p1",
        "r2_marginal","r2_conditional",
        "rmse_lmm","rmse_raw",
        "is_singular","zed_centered","zed_mean",
    ]
    out = pd.DataFrame(rows, columns=cols)
    return out


# ---------------------------
# Pretty table helper
# ---------------------------

def prettify_lmm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Round numeric values to 3 decimals
    - Combine CI columns into '[lo, hi]' strings
    - Order columns and sort by metric/state
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(3)

    def _fmt_ci(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f"[{lo:.3f}, {hi:.3f}]"

    if {"intercept_ci_lo", "intercept_ci_hi"}.issubset(df.columns):
        df["intercept_CI"] = [
            _fmt_ci(lo, hi) for lo, hi in zip(df["intercept_ci_lo"], df["intercept_ci_hi"])
        ]
        df.drop(columns=["intercept_ci_lo","intercept_ci_hi"], inplace=True)

    if {"slope_ci_lo", "slope_ci_hi"}.issubset(df.columns):
        df["slope_CI"] = [
            _fmt_ci(lo, hi) for lo, hi in zip(df["slope_ci_lo"], df["slope_ci_hi"])
        ]
        df.drop(columns=["slope_ci_lo","slope_ci_hi"], inplace=True)

    desired_order = [
        "metric","state",
        "intercept","intercept_CI","intercept_p",
        "slope_zed","slope_CI","slope_p1",
        "r2_marginal","r2_conditional",
        "rmse_lmm","rmse_raw",
        "is_singular","zed_centered","zed_mean",
    ]
    cols_in_df = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in cols_in_df]
    df = df[cols_in_df + remaining]

    sort_cols = [c for c in ["metric","state"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df
