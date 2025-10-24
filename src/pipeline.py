import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay,
    fbeta_score, make_scorer
)
from sklearn.model_selection import LeaveOneOut, GridSearchCV
import os
import joblib


# -----------------------------
# Optional correlation filter (for LR)
# -----------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drop columns with pairwise |corr| >= threshold (greedy on upper triangle).
    Works on numpy arrays (no feature names needed).
    """
    def __init__(self, threshold=0.95):
        self.threshold = float(threshold)
        self.keep_indices_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]
        corr = np.corrcoef(X, rowvar=False)
        keep = []
        dropped = set()
        for i in range(n_feat):
            if i in dropped:
                continue
            keep.append(i)
            for j in range(i+1, n_feat):
                if j in dropped:
                    continue
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= self.threshold:
                    dropped.add(j)
        self.keep_indices_ = np.array(keep, dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


# -----------------------------
# Helper: pick scorer by name ("f1" or "f2")
# -----------------------------
def get_scorer(score_metric: str = "f1"):
    sm = (score_metric or "f1").strip().lower()
    if sm == "f2":
        return make_scorer(fbeta_score, beta=2)
    # default
    return "f1"


# -----------------------------
# Inner-tuned, outer-evaluated nested LOOCV
# -----------------------------
def nested_loocv_predict(X, y, pipeline, param_grid, scoring="f1", positive_label=1, random_state=42):
    """
    Outer: LeaveOneOut
      For each outer train, tune hyper-params with inner LeaveOneOut GridSearchCV,
      fit best estimator, predict the held-out sample.
    Returns:
      y_true, y_pred, y_proba, chosen_params_list
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    loo_outer = LeaveOneOut()
    y_true = []
    y_pred = []
    y_proba = []
    chosen_params = []

    for train_idx, test_idx in loo_outer.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Inner LOO for tuning
        inner = LeaveOneOut()
        gs = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            cv=inner,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_tr, y_tr)

        best_est = gs.best_estimator_
        chosen_params.append(gs.best_params_)

        # Predict test sample
        yhat = best_est.predict(X_te)[0]
        y_true.append(y_te[0])
        y_pred.append(yhat)

        # Proba if available
        if hasattr(best_est, "predict_proba"):
            yprob = best_est.predict_proba(X_te)[0, 1]
        elif hasattr(best_est, "decision_function"):
            # map decision function to (0,1) via sigmoid for PR plotting
            df_val = best_est.decision_function(X_te)[0]
            yprob = 1 / (1 + np.exp(-df_val))
        else:
            yprob = float(yhat)
        y_proba.append(yprob)

    return (np.array(y_true), np.array(y_pred), np.array(y_proba), chosen_params)


# -----------------------------
# Evaluation + plots
# -----------------------------
def evaluate_and_plot(process_name, model_name, y_true, y_pred, y_proba, out_dir="plots", show_f2=False):
    """
    Compute F1/precision/recall, conf mat, PR curve; save plots.
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0) if show_f2 else None

    print(f"\n[{process_name}] {model_name}")
    print(f"  F1: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}" + (f" | F2: {f2:.3f}" if show_f2 else ""))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax_cm, values_format='d', colorbar=False)
    ax_cm.set_title(f"{process_name} - {model_name}\nConfusion Matrix")
    fig_cm.tight_layout()
    cm_path = f"{out_dir}/{process_name.replace(' ','_')}_{model_name}_confmat.png"
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    # Precision-Recall
    precision, recall, thresh = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig_pr, ax_pr = plt.subplots(figsize=(5,4))
    ax_pr.plot(recall, precision, lw=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{process_name} - {model_name}\nPR curve (AP={ap:.3f})")
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    pr_path = f"{out_dir}/{process_name.replace(' ','_')}_{model_name}_pr.png"
    fig_pr.savefig(pr_path, dpi=150)
    plt.close(fig_pr)

    return {"f1": f1, "precision": prec, "recall": rec, "F2": f2, "AP": ap, "cm": cm,
            "confmat_path": cm_path, "pr_path": pr_path}


# -----------------------------
# Define both model pipelines + grids
# -----------------------------
def make_lr_pipeline_and_grid():
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        # CorrelationFilter may be used or skipped via param grid
        ("corr", CorrelationFilter(threshold=0.95)),
        ("clf", LogisticRegression(
            solver="saga", max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])

    Cs = np.logspace(-3, 3, 7)
    grid = []

    # With correlation filter (try a few thresholds)
    for thr in [0.85, 0.90, 0.95]:
        grid.append({
            "corr": [CorrelationFilter(threshold=thr)],
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__l1_ratio": [0.5],   # only used for elasticnet
            "clf__C": Cs
        })

    # Without correlation filter
    grid.append({
        "corr": ["passthrough"],
        "clf__penalty": ["l1", "l2", "elasticnet"],
        "clf__l1_ratio": [0.5],
        "clf__C": Cs
    })

    return pipe_lr, grid


def make_dt_pipeline_and_grid():
    pipe_dt = Pipeline([
        # No scaling needed; keep structure for consistency
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])

    grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [2, 3, 4, 5, 6],
        "clf__min_samples_leaf": [1, 2, 4]
    }
    return pipe_dt, grid


# -----------------------------
# Final fit on all data and save best model
# -----------------------------
def fit_full_and_save(X, y, pipeline, param_grid, scoring, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    inner = LeaveOneOut()
    gs = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        cv=inner,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )
    gs.fit(X, y)
    best_est = gs.best_estimator_
    joblib.dump(best_est, save_path)
    print(f"  Saved final tuned model → {save_path}")
    return best_est, gs.best_params_


# -----------------------------
# Run one process (one device × one state)
# -----------------------------
def run_process(process_name, df_proc, target_col="low stability", plots_dir="plots",
                score_metric: str = "f1", models_dir: str = None):
    """
    df_proc: DataFrame with numeric feature columns + binary target 'low stability'

    score_metric: "f1" (default) or "f2"
    models_dir: where to save trained models. If None, defaults to:
                <PROJECT_ROOT>/machine_learning/trained_models
    """
    # Separate X, y
    y = df_proc[target_col].astype(int).to_numpy()
    X = df_proc.drop(columns=[target_col]).to_numpy(dtype=float)

    # Make output dirs
    os.makedirs(plots_dir, exist_ok=True)
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    # Scorer
    scoring = get_scorer(score_metric)
    show_f2 = (str(score_metric).strip().lower() == "f2")

    results_summary = {}

    # --- Logistic Regression ---
    pipe_lr, grid_lr = make_lr_pipeline_and_grid()
    y_true, y_pred, y_proba, chosen = nested_loocv_predict(
        X, y, pipe_lr, grid_lr, scoring=scoring
    )
    res_lr = evaluate_and_plot(process_name, "LogReg", y_true, y_pred, y_proba,
                               out_dir=plots_dir, show_f2=show_f2)
    # summarize chosen params
    params_counter_lr = Counter(tuple(sorted(d.items())) for d in chosen)
    print("  Most chosen LR params (top 5):")
    for (params, cnt) in params_counter_lr.most_common(5):
        print(f"    {dict(params)} → {cnt} folds")
    results_summary["LogReg"] = res_lr

    # Save final tuned LR on all data
    lr_save = os.path.join(
        models_dir,
        f"{process_name.replace(' ','_')}_LogReg_{score_metric.lower()}.joblib"
    )
    _, best_lr_params = fit_full_and_save(X, y, pipe_lr, grid_lr, scoring, lr_save)

    # --- Decision Tree ---
    pipe_dt, grid_dt = make_dt_pipeline_and_grid()
    y_true, y_pred, y_proba, chosen = nested_loocv_predict(
        X, y, pipe_dt, grid_dt, scoring=scoring
    )
    res_dt = evaluate_and_plot(process_name, "DecisionTree", y_true, y_pred, y_proba,
                               out_dir=plots_dir, show_f2=show_f2)
    params_counter_dt = Counter(tuple(sorted(d.items())) for d in chosen)
    print("  Most chosen DT params (top 5):")
    for (params, cnt) in params_counter_dt.most_common(5):
        print(f"    {dict(params)} → {cnt} folds")
    results_summary["DecisionTree"] = res_dt

    # Save final tuned DT on all data
    dt_save = os.path.join(
        models_dir,
        f"{process_name.replace(' ','_')}_DecisionTree_{score_metric.lower()}.joblib"
    )
    _, best_dt_params = fit_full_and_save(X, y, pipe_dt, grid_dt, scoring, dt_save)

    return results_summary


def plot_feature_corr_matrix(df, fig_size=(10, 8)):
    # Select numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=fig_size)

    # Draw the heatmap with the mask
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
