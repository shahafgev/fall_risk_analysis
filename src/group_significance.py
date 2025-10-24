import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, sem


def test_significant_differences(df1, df2, alpha=0.05, exclude_metrics=None, metric_order=None):
    """
    Test significant differences between two groups 
    for each (metric, state, device) combination.

    Parameters:
    - df1: DataFrame of first group
    - df2: DataFrame of second group
    - alpha: significance level (default 0.05)
    - exclude_metrics: list of metric names to omit (case-insensitive)
    - metric_order: list specifying the row order (case-insensitive). 
                    Unspecified metrics are appended alphabetically.

    Returns:
    - DataFrame summarizing p-values and test used.
    """
    devices = ['Force_Plate', 'ZED_COM']
    state_order = ["open", "closed"]   # preferred order
    states = [s for s in state_order if s in set(df1['state']).union(set(df2['state']))]

    # --- Helpers ---
    def _norm(s):  # normalize for case-insensitive comparison
        return str(s).strip().lower()

    exclude_set = set(_norm(m) for m in (exclude_metrics or []))

    # Gather available metrics
    all_metrics = sorted(list(set(df1['metric']).union(set(df2['metric']))), key=lambda x: _norm(x))
    metrics = [m for m in all_metrics if _norm(m) not in exclude_set]

    # Apply custom order
    if metric_order:
        pref = [_norm(m) for m in metric_order]
        in_pref = [m for m in metrics if _norm(m) in pref]
        in_pref.sort(key=lambda m: pref.index(_norm(m)))
        not_in_pref = sorted([m for m in metrics if _norm(m) not in pref], key=lambda x: _norm(x))
        metrics = in_pref + not_in_pref

    results = []
    for metric in metrics:
        for state in states:
            row = {'Metric': metric, 'State': state}

            for device in devices:
                # Filter data
                df1_values = df1[
                    (df1['state'] == state) &
                    (df1['device'] == device) &
                    (df1['metric'] == metric)
                ]['average_value'].dropna()

                df2_values = df2[
                    (df2['state'] == state) &
                    (df2['device'] == device) &
                    (df2['metric'] == metric)
                ]['average_value'].dropna()

                # Skip if not enough samples
                if len(df1_values) < 3 or len(df2_values) < 3:
                    row[f'P-value {device}'] = np.nan
                    row[f'{device} test used'] = 'Too few samples'
                    continue

                # Normality
                _, p_df1_normality = shapiro(df1_values)
                _, p_df2_normality = shapiro(df2_values)
                normality = (p_df1_normality > alpha) and (p_df2_normality > alpha)

                if normality:
                    # Homogeneity of variance
                    _, p_levene = levene(df1_values, df2_values)
                    equal_var = p_levene > alpha

                    if equal_var:
                        _, pval = ttest_ind(df1_values, df2_values, equal_var=True)
                        test_used = 't-test'
                    else:
                        _, pval = ttest_ind(df1_values, df2_values, equal_var=False)
                        test_used = 'Welch'
                else:
                    _, pval = mannwhitneyu(df1_values, df2_values, alternative='two-sided')
                    test_used = 'Mann-Whitney'

                row[f'P-value {device}'] = pval
                row[f'{device} test used'] = test_used

            results.append(row)

    # Build final DataFrame
    results_df = pd.DataFrame(results)

    # Set the exact column order
    ordered_cols = [
        'Metric', 'State',  
        'P-value Force_Plate', 'P-value ZED_COM', 
        'Force_Plate test used', 'ZED_COM test used'
    ]
    results_df = results_df[ordered_cols]

    return results_df