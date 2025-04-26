import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, sem


def test_significant_differences(df1, df2, alpha=0.05):
    """
    Test significant differences between older adults and students 
    for each (state, metric, device) combination.

    Parameters:
    - df1: DataFrame of first group
    - df2: DataFrame of second group
    - alpha: significance level (default 0.05)

    Returns:
    - DataFrame summarizing p-values, significance, and test used (unordered).
    - Styled DataFrame with green cells for significant True values (optional for export).
    """
    value_col = 'average_value' if 'average_value' in df1.columns else 'value'
    devices = ['Force_Plate', 'ZED_COM']
    states = sorted(list(set(df1['state']).union(set(df2['state']))))
    metrics = sorted(list(set(df1['metric']).union(set(df2['metric']))))

    results = []

    for state in states:
        for metric in metrics:
            row = {'State': state, 'Metric': metric}

            for device in devices:
                # Filter data
                df1_values = df1[
                    (df1['state'] == state) &
                    (df1['device'] == device) &
                    (df1['metric'] == metric)
                ][value_col].dropna()

                df2_values = df2[
                    (df2['state'] == state) &
                    (df2['device'] == device) &
                    (df2['metric'] == metric)
                ][value_col].dropna()

                # Skip if not enough samples
                if len(df1_values) < 3 or len(df2_values) < 3:
                    row[f'P-value {device}'] = np.nan
                    row[f'{device} significant'] = np.nan
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
                        stat, pval = ttest_ind(df1_values, df2_values, equal_var=True)
                        test_used = 't-test'
                    else:
                        stat, pval = ttest_ind(df1_values, df2_values, equal_var=False)
                        test_used = 'Welch'
                else:
                    stat, pval = mannwhitneyu(df1_values, df2_values, alternative='two-sided')
                    test_used = 'Mann-Whitney'

                row[f'P-value {device}'] = pval
                row[f'{device} significant'] = pval < alpha
                row[f'{device} test used'] = test_used

            results.append(row)

    # Build final DataFrame
    results_df = pd.DataFrame(results)

    # Set the exact column order
    ordered_cols = [
        'State', 'Metric', 
        'P-value Force_Plate', 'P-value ZED_COM', 
        'Force_Plate significant', 'ZED_COM significant',
        'Force_Plate test used', 'ZED_COM test used'
    ]
    results_df = results_df[ordered_cols]

    return results_df


def style_significance(df):
    """
    Style the significance columns: color green if True.
    """
    def highlight_significant(val):
        color = 'background-color: green' if val is True else ''
        return color

    styled = df.style.applymap(highlight_significant, subset=['Force_Plate significant', 'ZED_COM significant'])
    return styled


def siginificance_dataframe(df1, df2, alpha=0.05):
    df = test_significant_differences(df1, df2, alpha)
    styled_df = style_significance(df)
    return styled_df


def plot_metrics_with_significance_side_by_side(state, device, df1, df2, group1_label, group2_label, alpha=0.05, add_error_bars=True):
    """
    Plots two subplots side-by-side: one for all metrics except 'Path length',
    and one just for 'Path length', keeping consistent bar spacing and width.

    Parameters:
    - df1, df2: DataFrames for group1 and group2
    - alpha: significance level (default 0.05)
    - state, device: 'open'/'closed' and 'Force_Plate'/'ZED_COM'
    - add_error_bars: whether to show SEM error bars
    """
    results_df = test_significant_differences(df1, df2, alpha)
    value_col = 'average_value' if 'average_value' in df1.columns else 'value'

    # Split metrics
    all_metrics = sorted(list(set(df1['metric']).union(set(df2['metric']))))
    path_metric = [m for m in all_metrics if m.lower() == 'path length']
    other_metrics = [m for m in all_metrics if m.lower() != 'path length']
    metrics = other_metrics + path_metric

    def collect_data(metric_list):
        means_df1, means_df2, sems_df1, sems_df2, sigs = [], [], [], [], []
        for metric in metric_list:
            df1_vals = df1[(df1['state'] == state) & (df1['device'] == device) & (df1['metric'] == metric)][value_col].dropna()
            df2_vals = df2[(df2['state'] == state) & (df2['device'] == device) & (df2['metric'] == metric)][value_col].dropna()
            means_df1.append(df1_vals.mean())
            means_df2.append(df2_vals.mean())
            sems_df1.append(sem(df1_vals) if len(df1_vals) > 1 else 0)
            sems_df2.append(sem(df2_vals) if len(df2_vals) > 1 else 0)
            sig_row = results_df[(results_df['State'] == state) & (results_df['Metric'] == metric)]
            sig = sig_row.iloc[0][f'{device} significant'] if not sig_row.empty else False
            sigs.append(sig)
        return means_df1, means_df2, sems_df1, sems_df2, sigs

    # Get all values
    mean_df1, mean_df2, sem_df1, sem_df2, sigs = collect_data(metrics)
    width = 0.35
    x_all = np.arange(len(metrics))

    # Split values
    split_idx = len(other_metrics)
    x_left = x_all[:split_idx]
    x_right = x_all[split_idx:]

    # Start plotting
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6), sharey=False, gridspec_kw={'width_ratios': [len(other_metrics), 1]})

    # Left subplot (all except path length)
    bars_df1_l = ax_left.bar(x_left - width/2, mean_df1[:split_idx], width,
                            yerr=sem_df1[:split_idx] if add_error_bars else None, capsize=5, label=group1_label)
    bars_df2_l = ax_left.bar(x_left + width/2, mean_df2[:split_idx], width,
                            yerr=sem_df2[:split_idx] if add_error_bars else None, capsize=5, label=group2_label)

    for i, sig in enumerate(sigs[:split_idx]):
        if sig:
            max_height = max(bars_df1_l[i].get_height(), bars_df2_l[i].get_height())
            ax_left.text(x_left[i], max_height + 0.05 * max_height, '*', ha='center', va='bottom', fontsize=14, color='red')

    ax_left.set_title(f'Metrics (excluding Path Length)')
    ax_left.set_ylabel('Average Value')
    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels(other_metrics, rotation=45, ha='right')
    ax_left.legend()

    # Right subplot (Path Length only)
    if x_right.size > 0:
        bars_df1_r = ax_right.bar(x_right - width/2, mean_df1[split_idx:], width,
                                 yerr=sem_df1[split_idx:] if add_error_bars else None, capsize=5)
        bars_df2_r = ax_right.bar(x_right + width/2, mean_df2[split_idx:], width,
                                 yerr=sem_df2[split_idx:] if add_error_bars else None, capsize=5)
        for i, sig in enumerate(sigs[split_idx:]):
            if sig:
                max_height = max(bars_df1_r[i].get_height(), bars_df2_r[i].get_height())
                ax_right.text(x_right[i], max_height + 0.05 * max_height, '*', ha='center', va='bottom', fontsize=14, color='red')

        ax_right.set_title('Path Length')
        ax_right.set_xticks(x_right)
        ax_right.set_xticklabels(path_metric, rotation=0)

    # Final layout
    fig.suptitle(f'State: {state.capitalize()}, Device: {device}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()