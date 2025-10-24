# src/lmm_fatigue_learning.py
# ------------------------------------------------------------
# Linear Mixed Models in R (lme4/lmerTest) via rpy2
# Test fatigue/learning on Force_Plate data:
#   value ~ trial_c (+ I(trial_c^2)) + (1 | participant)   [or (1 + trial_c | participant)]
#
# Key outputs per (metric, state):
#   - slope_trial (beta_1), 95% Wald CI, p_slope (H0: beta_1 = 0)
#   - optional quad term beta_2 if include_quadratic=True
#   - rmse_lmm, r2_marginal, r2_conditional, is_singular
#   - direction = {'learning','fatigue','none'} based on slope & p
#
# Assumes df columns:
#   ['participant name','device','metric','state','trial','value']
#   Uses ONLY device == 'Force_Plate'
# ------------------------------------------------------------

from __future__ import annotations

import os
from typing import Iterable, Optional, List, Dict

import numpy as np
import pandas as pd

# --- Point to your R install (Windows-friendly defaults) ---
R_HOME = os.environ.get("R_HOME", r"C:\Program Files\R\R-4.5.1")
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

# Optional: expose user libs
os.environ.setdefault("R_LIBS_USER", r"C:\Users\User\AppData\Local\R\win-library\4.5")

# rpy2 imports
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError

# ---------------------------
# R setup
# ---------------------------
os.environ.setdefault("R_HOME", r"C:\Program Files\R\R-4.5.1")
os.environ.setdefault("R_LIBS_USER", r"C:\Users\User\AppData\Local\R\win-library\4.5")

_r = robjects.r
_r("""
ulib <- Sys.getenv('R_LIBS_USER')
if (!nzchar(ulib)) {
  ulib <- file.path(Sys.getenv('USERPROFILE'), 'AppData/Local/R/win-library/4.5')
}
dir.create(ulib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(ulib, .libPaths()))
""")

# Ensure required packages (MuMIn for R^2)
_r("""
needed <- c('lme4','lmerTest','broom.mixed','MuMIn','emmeans')
to_install <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]
if (length(to_install)) {
  install.packages(to_install, repos = 'https://cloud.r-project.org', lib = .libPaths()[1])
}
""")

# Import packages
lme4 = importr("lme4")
lmerTest = importr("lmerTest")
broom_mixed = importr("broom.mixed")
MuMIn = importr("MuMIn")
emmeans = importr("emmeans")

# ---------------------------
# Core: fit an LMM in R
# ---------------------------

def _fit_fatigue_lmm_in_r(
    df_py: pd.DataFrame,
    participant_col: str,
    include_ci: bool = True,
    alpha: float = 0.05,
    random_slope: bool = False,
    include_quadratic: bool = False,
    trial_as_factor: bool = False,
    reml: bool = True,
) -> Dict[str, float]:
    """
    df_py expects columns: [participant_col, 'trial_c', 'value']
    (and 'trial' if trial_as_factor=True for factor coding).
    Returns dict of fixed effects and diagnostics.
    """

    if df_py.empty:
        return {
            "slope_trial": np.nan, "slope_ci_lo": np.nan, "slope_ci_hi": np.nan, "p_slope": np.nan,
            "quad": np.nan, "quad_ci_lo": np.nan, "quad_ci_hi": np.nan, "p_quad": np.nan,
            "anova_df": np.nan, "anova_F": np.nan, "anova_p": np.nan,
            "rmse_lmm": np.nan, "r2_marginal": np.nan, "r2_conditional": np.nan,
            "is_singular": np.nan
        }

    # Convert to R
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(
            df_py.rename(columns={participant_col: "participant"})
        )

    # Build formula
    if trial_as_factor:
        # ANOVA-style model with trial as categorical (baseline test: do means differ by trial?)
        fixed = "as.factor(trial)"
        re_form = "(1 | participant)"  # random intercept; random slopes for factor are heavy
    else:
        fixed = "trial_c" + (" + I(trial_c^2)" if include_quadratic else "")
        re_form = "(1 + trial_c | participant)" if random_slope else "(1 | participant)"

    formula = f"value ~ {fixed} + {re_form}"

    _r.assign("df_in", r_df)
    _r.assign("formula_str", formula)
    _r.assign("use_reml", bool(reml))
    _r.assign("want_ci", bool(include_ci))
    _r.assign("alpha_level", float(alpha))
    _r.assign("trial_as_factor", bool(trial_as_factor))

    r_code = """
    suppressPackageStartupMessages({
      library(lmerTest); library(broom.mixed); library(MuMIn)
    })

    fit <- lmer(as.formula(formula_str), data = df_in, REML = use_reml)

    # Fixed effects table with/without CI
    if (want_ci) {
      tt <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE,
                              conf.level = 1 - alpha_level, conf.method = "Wald")
    } else {
      tt <- broom.mixed::tidy(fit, effects = "fixed", conf.int = FALSE)
      tt$conf.low <- NA_real_; tt$conf.high <- NA_real_
    }

    # Pull slope(s)
    get_row <- function(df, term) {
      if (term %in% df$term) df[df$term == term, , drop=FALSE] else data.frame()
    }

    slope_row <- get_row(tt, "trial_c")
    quad_row  <- get_row(tt, "I(trial_c^2)")

    est_slope <- if (nrow(slope_row)) slope_row$estimate else NA_real_
    ci_slope_lo <- if (nrow(slope_row)) slope_row$conf.low else NA_real_
    ci_slope_hi <- if (nrow(slope_row)) slope_row$conf.high else NA_real_
    p_slope <- if (nrow(slope_row) && "p.value" %in% names(slope_row)) slope_row$p.value else NA_real_

    est_quad <- if (nrow(quad_row)) quad_row$estimate else NA_real_
    ci_quad_lo <- if (nrow(quad_row)) quad_row$conf.low else NA_real_
    ci_quad_hi <- if (nrow(quad_row)) quad_row$conf.high else NA_real_
    p_quad <- if (nrow(quad_row) && "p.value" %in% names(quad_row)) quad_row$p.value else NA_real_

    # If trial is a factor: run ANOVA (type III via lmerTest) on trial
    an_df <- NA_real_; an_F <- NA_real_; an_p <- NA_real_
    if (trial_as_factor) {
      an <- tryCatch(anova(fit, ddf="Satterthwaite"), error = function(e) NULL)
      if (!is.null(an)) {
        rn <- rownames(an)
        ix <- which(grepl("as.factor\\(trial\\)", rn))
        if (length(ix) == 1) {
          an_df <- suppressWarnings(as.numeric(an$DenDF[ix]))
          an_F  <- suppressWarnings(as.numeric(an$`F value`[ix]))
          an_p  <- suppressWarnings(as.numeric(an$`Pr(>F)`[ix]))
        }
      }
    }

    # Residual RMSE
    rmse_lmm <- NA_real_
    resid_cond <- tryCatch(resid(fit), error=function(e) NULL)
    if (!is.null(resid_cond)) rmse_lmm <- sqrt(mean(resid_cond^2, na.rm = TRUE))

    # R^2
    r2_marginal <- NA_real_; r2_conditional <- NA_real_
    rs <- tryCatch(MuMIn::r.squaredGLMM(fit), error=function(e) NULL)
    if (!is.null(rs)) {
      r2_marginal    <- suppressWarnings(as.numeric(rs[1, "R2m"]))
      r2_conditional <- suppressWarnings(as.numeric(rs[1, "R2c"]))
    }

    is_singular <- tryCatch(lme4::isSingular(fit, tol=1e-5), error=function(e) NA)

    list(
      slope_trial     = est_slope,
      slope_ci_lo     = ci_slope_lo,
      slope_ci_hi     = ci_slope_hi,
      p_slope         = p_slope,
      quad            = est_quad,
      quad_ci_lo      = ci_quad_lo,
      quad_ci_hi      = ci_quad_hi,
      p_quad          = p_quad,
      anova_df        = an_df,
      anova_F         = an_F,
      anova_p         = an_p,
      rmse_lmm        = rmse_lmm,
      r2_marginal     = r2_marginal,
      r2_conditional  = r2_conditional,
      is_singular     = as.logical(is_singular)
    )
    """

    try:
        res = _r(r_code)
    except RRuntimeError:
        return {
            "slope_trial": np.nan, "slope_ci_lo": np.nan, "slope_ci_hi": np.nan, "p_slope": np.nan,
            "quad": np.nan, "quad_ci_lo": np.nan, "quad_ci_hi": np.nan, "p_quad": np.nan,
            "anova_df": np.nan, "anova_F": np.nan, "anova_p": np.nan,
            "rmse_lmm": np.nan, "r2_marginal": np.nan, "r2_conditional": np.nan,
            "is_singular": np.nan
        }

    out = {k: float(res.rx2(k)[0]) if len(res.rx2(k)) else np.nan for k in res.names}
    try:
        out["is_singular"] = bool(res.rx2("is_singular")[0])
    except Exception:
        pass
    return out

# ---------------------------
# Public API
# ---------------------------

def run_fatigue_learning_lmm_table(
    df: pd.DataFrame,
    *,
    states: Iterable[str] = ("open", "closed"),
    metrics: Optional[Iterable[str]] = None,
    participant_col: str = "participant name",
    device_col: str = "device",
    value_col: str = "value",
    trial_col: str = "trial",
    state_col: str = "state",
    include_ci: bool = True,
    alpha: float = 0.05,
    random_slope: bool = False,
    include_quadratic: bool = False,
    trial_as_factor: bool = False,
    reml: bool = True,
    center_trial: bool = True,
) -> pd.DataFrame:
    """
    Build LMMs per (metric, state) **for Force_Plate only** and test trend over trials.

    Returns tidy DataFrame with:
      ['metric','state','slope_trial','slope_ci_lo','slope_ci_hi','p_slope',
       'quad','quad_ci_lo','quad_ci_hi','p_quad',
       'anova_df','anova_F','anova_p',
       'rmse_lmm','r2_marginal','r2_conditional','is_singular',
       'direction']
    """
    df = df[df[device_col] == "Force_Plate"].copy()
    df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if metrics is None:
        metrics = list(df["metric"].dropna().unique())

    rows: List[Dict] = []
    for metric in metrics:
        for state in states:
            sub = df[(df["metric"] == metric) & (df[state_col].str.lower() == str(state).lower())].copy()
            if sub.empty:
                rows.append({
                    "metric": metric, "state": state,
                    "slope_trial": np.nan, "slope_ci_lo": np.nan, "slope_ci_hi": np.nan, "p_slope": np.nan,
                    "quad": np.nan, "quad_ci_lo": np.nan, "quad_ci_hi": np.nan, "p_quad": np.nan,
                    "anova_df": np.nan, "anova_F": np.nan, "anova_p": np.nan,
                    "rmse_lmm": np.nan, "r2_marginal": np.nan, "r2_conditional": np.nan, "is_singular": np.nan,
                    "direction": "none"
                })
                continue

            # center trial if requested
            if center_trial:
                sub["trial_c"] = sub[trial_col] - sub[trial_col].mean()
            else:
                sub["trial_c"] = sub[trial_col]

            # For factor model, keep raw trial too
            if trial_as_factor:
                sub["trial"] = sub[trial_col]

            df_fit = sub[[participant_col, "trial_c", value_col] + (["trial"] if trial_as_factor else [])].rename(
                columns={value_col: "value"}
            )

            res = _fit_fatigue_lmm_in_r(
                df_py=df_fit,
                participant_col=participant_col,
                include_ci=include_ci,
                alpha=alpha,
                random_slope=random_slope,
                include_quadratic=include_quadratic,
                trial_as_factor=trial_as_factor,
                reml=reml,
            )

            # Direction classification from linear slope
            direction = "none"
            if np.isfinite(res.get("p_slope", np.nan)) and res["p_slope"] <= alpha and np.isfinite(res.get("slope_trial", np.nan)):
                direction = "learning" if res["slope_trial"] > 0 else "fatigue"

            rows.append({
                "metric": metric,
                "state": state,
                **res,
                "direction": direction
            })

    cols = [
        "metric","state",
        "slope_trial","slope_ci_lo","slope_ci_hi","p_slope",
        "quad","quad_ci_lo","quad_ci_hi","p_quad",
        "anova_df","anova_F","anova_p",
        "rmse_lmm","r2_marginal","r2_conditional","is_singular",
        "direction"
    ]
    return pd.DataFrame(rows, columns=cols)

# ---------------------------
# Helpers
# ---------------------------

def prettify_fatigue_table(df: pd.DataFrame) -> pd.DataFrame:
    """Round numbers, fold CIs into strings, order/sort columns."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(3)

    def fmt_ci(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f"[{lo:.3f}, {hi:.3f}]"

    if {"slope_ci_lo","slope_ci_hi"}.issubset(df.columns):
        df["slope_CI"] = [fmt_ci(lo, hi) for lo, hi in zip(df["slope_ci_lo"], df["slope_ci_hi"])]
        df.drop(columns=["slope_ci_lo","slope_ci_hi"], inplace=True)

    if {"quad_ci_lo","quad_ci_hi"}.issubset(df.columns):
        df["quad_CI"] = [fmt_ci(lo, hi) for lo, hi in zip(df["quad_ci_lo"], df["quad_ci_hi"])]
        df.drop(columns=["quad_ci_lo","quad_ci_hi"], inplace=True)

    desired = [
        "metric","state",
        "slope_trial","slope_CI","p_slope",
        "quad","quad_CI","p_quad",
        "r2_marginal","r2_conditional",
        "rmse_lmm","is_singular",
        "direction",
        "anova_df","anova_F","anova_p",
    ]
    cols_in = [c for c in desired if c in df.columns]
    rest = [c for c in df.columns if c not in cols_in]
    df = df[cols_in + rest]

    if {"metric","state"}.issubset(df.columns):
        df = df.sort_values(["metric","state"]).reset_index(drop=True)
    return df


def add_fdr(df: pd.DataFrame, pcol: str = "p_slope", method: str = "fdr_bh") -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction for a p-value column."""
    from statsmodels.stats.multitest import multipletests
    df = df.copy()
    mask = df[pcol].notna()
    if mask.any():
        rej, p_adj, _, _ = multipletests(df.loc[mask, pcol].values, method=method)
        df.loc[mask, pcol + "_adj"] = p_adj
        df.loc[mask, "signif_adj"] = rej
    else:
        df[pcol + "_adj"] = np.nan
        df["signif_adj"] = False
    return df
