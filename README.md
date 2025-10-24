# Fall Stability Analysis

This project analyzes postural stability data collected from older adults and students using multiple devices (Force Plate, ZED 2i, Smartphones).

## Project Structure

```
fall_risk_analysis/
├── notebooks/
│   ├── additional_info_bias.ipynb
│   ├── signals.ipynb
│   ├── fatigue_learning.ipynb
│   ├── model.ipynb
│   ├── device_comparision/
│   │   ├── rmse.ipynb
│   │   ├── pearson_correlation.ipynb
│   │   ├── check_assumption.ipynb
│   │   ├── ccc.ipynb
│   │   ├── bland_altman.ipynb
│   │   └── assumptions/
│   │       ├── st_average_assumptions.csv
│   │       ├── st_trial_assumptions.csv
│   │       ├── oa_average_assumptions.csv
│   │       └── oa_trial_assumptions.csv
│   ├── group_tests/
│   │   ├── low_vs_high_stability.ipynb
│   │   └── students_vs_older.ipynb
│   └── exploration/
│       ├── participant_visualizer.ipynb
│       ├── phones_data_visualize.ipynb
│       └── zed_fps_check.ipynb
├── results/
│   ├── models/
│   ├── plots/
│   └── tables/
├── src/
│   ├── aggregation.py
│   ├── assumption_utils.py
│   ├── bland_altman.py
│   ├── ccc.py
│   ├── create_datasets.py
│   ├── group_significance.py
│   ├── pearson_correlation.py
│   ├── preprocessing.py
│   └── rmse.py

├── README.md
├── requirements.txt
```

