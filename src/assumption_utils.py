import pandas as pd
import os
import sys


notebooks_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
sys.path.insert(0, notebooks_path)


def load_assumption_summary(state, metric, average_level, source_name):
    """
    Loads the assumption summary from the appropriate CSV and returns
    the row matching the specified state and metric.

    Parameters:
    - state: 'open' or 'closed'
    - metric: metric name (e.g. 'AP Range')
    - average_level: bool, True for average-level, False for trial-level
    - source_name: one of ['oa', 'st'] (older adults or students)

    Returns:
    - dict with normality, heteroscedasticity, and summary
    """
    level = "average" if average_level else "trial"
    filename = f"assumptions/{source_name}_assumptions.csv"
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Assumption file not found: {filename}")

    df = pd.read_csv(filename)
    row = df[(df['state'] == state) & (df['metric'] == metric)]

    if row.empty:
        return {"normality": "Not found", "heteroscedasticity": "Not found", "summary": "Not found"}
    
    return row.iloc[0][['normality', 'heteroscedasticity', 'summary']].to_dict()
