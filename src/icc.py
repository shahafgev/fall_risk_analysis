import pandas as pd
from pingouin import intraclass_corr
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_icc_by_metric_and_state(df, trial_level=False):
    """
    Calculate ICC between ZED_COM and Force_Plate for each (metric, state) pair.

    Parameters:
        df (pd.DataFrame): Trial-level or already-averaged data.
        trial_level (bool): If True, data is at the trial level (ICC(A,1));
                            if False, data is pre-averaged (ICC(A,k)).

    Returns:
        pd.DataFrame with ['Metric', 'State', 'ICC']
    """
    results = []

    value_col = 'value' if trial_level else 'average_value'
    index_cols = ['participant name', 'trial'] if trial_level else ['participant name']

    for (metric, state), group in df.groupby(['metric', 'state']):
        pivot_df = group.pivot_table(
            index=index_cols,
            columns='device',
            values=value_col
        ).dropna()

        if {'ZED_COM', 'Force_Plate'}.issubset(pivot_df.columns):
            long_df = pivot_df.reset_index().melt(
                id_vars=index_cols,
                value_vars=['ZED_COM', 'Force_Plate'],
                var_name='Device',
                value_name='Value'
            )
            long_df['metric'] = metric
            long_df['state'] = state

            icc_result = intraclass_corr(
                data=long_df,
                targets='participant name',
                raters='Device',
                ratings='Value',
                nan_policy='omit'
            )

            # Correct mapping:
            # trial_level = True → ICC2 (A,1)
            # trial_level = False → ICC3 (A,k)
            icc_type = 'ICC2' if trial_level else 'ICC3'
            icc_value = icc_result.loc[icc_result['Type'] == icc_type, 'ICC'].values[0]

            results.append({
                'Metric': metric,
                'State': state,
                'ICC': icc_value
            })

    return pd.DataFrame(results)


def plot_icc_results(icc_df, title='ICC by Metric and State'):
    """
    Plot ICC values grouped by metric and state.

    Parameters:
        icc_df (pd.DataFrame): DataFrame with columns ['Metric', 'State', 'ICC']
        title (str): Title of the plot
    """
    plt.figure(figsize=(12, 5))
    sns.barplot(data=icc_df, x='Metric', y='ICC', hue='State', dodge=True)


    plt.title(title)
    plt.ylabel('Intraclass Correlation Coefficient (ICC)')
    plt.xlabel('Metric')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()