# Fall Stability Analysis

This project analyzes postural stability data collected from older adults and students using multiple devices (Force Plate, ZED 2i, Smartphones).

## Project Structure

'''
- **src/**: Core analysis scripts and utilities:
  - `preprocessing.py`: Data cleaning and preparation
  - `aggregation.py`: Data aggregation functions
  - `assumption_utils.py`: Assumption checks for statistical tests
  - `bland_altman.py`, `ccc.py`, `pearson_correlation.py`, `rmse.py`: Device comparison metrics
  - `group_significance.py`: Group analysis and statistical testing
  - `create_datasets.py`: Dataset creation scripts
- **notebooks/**: Jupyter notebooks for analysis and visualization:
  - `exploration/`: Data exploration and visualization
  - `device_comparision/`: Device agreement and statistical comparison (Bland–Altman, CCC, Pearson, RMSE, assumptions)
  - `group_tests/`: Group significance tests (e.g., students vs. older adults)
  - `fatigue_learning.ipynb`: Fatigue and learning effects
  - `model.ipynb`: Classification and clustering models
  - `signals.ipynb`: Signal processing and feature extraction
- **results/**: Output from analyses:
  - `models/`: Saved models
  - `plots/`: Generated plots
  - `tables/`: Summary tables and results
'''

