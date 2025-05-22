import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def concordance_correlation_coefficient(x, y):
    """Manual implementation of CCC between two arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    rho = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))

    ccc = (2 * rho * np.sqrt(var_x) * np.sqrt(var_y)) / (
        var_x + var_y + (mean_x - mean_y) ** 2
    )
    return ccc


def calculate_ccc_by_metric_and_state(df, trial_level=False):
    """
    Calculate CCC between ZED_COM and Force_Plate for each (metric, state) pair.

    Parameters:
        df (pd.DataFrame): Trial-level or already-averaged data.
        trial_level (bool): If True, uses 'value'; otherwise, 'average_value'.

    Returns:
        pd.DataFrame with columns ['Metric', 'State', 'CCC']
    """
    results = []

    value_col = 'value' if trial_level else 'average_value'
    index_cols = ['participant name', 'trial'] if trial_level else ['participant name']

    for (metric, state), group in df.groupby(['metric', 'state']):
        # Pivot to wide format
        pivot_df = group.pivot_table(
            index=index_cols,
            columns='device',
            values=value_col
        ).dropna()

        if {'ZED_COM', 'Force_Plate'}.issubset(pivot_df.columns):
            ccc_value = concordance_correlation_coefficient(
                pivot_df['ZED_COM'],
                pivot_df['Force_Plate']
            )
            results.append({
                'Metric': metric,
                'State': state,
                'CCC': ccc_value
            })

    return pd.DataFrame(results)


def plot_ccc_results(ccc_df, title='CCC by Metric and State'):
    """
    Plot CCC values grouped by metric and state.

    Parameters:
        ccc_df (pd.DataFrame): DataFrame with ['Metric', 'State', 'CCC']
        title (str): Title of the plot
    """
    plt.figure(figsize=(12, 5))
    sns.barplot(data=ccc_df, x='Metric', y='CCC', hue='State', dodge=True)

    plt.title(title)
    plt.ylabel('Concordance Correlation Coefficient (CCC)')
    plt.xlabel('Metric')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
