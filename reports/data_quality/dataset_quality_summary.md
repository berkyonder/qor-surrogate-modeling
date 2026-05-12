# Dataset Quality Summary

## Raw Dataset
- Rows: 864
- Columns: 21

## Clean Dataset
- Rows: 864
- Columns: 17

## Missing Values
- Total missing values: 0

## Duplicate Analysis
- rows: 864
- columns: 21
- full_duplicate_rows: 0
- duplicate_configurations: 0
- duplicate_metric_vectors: 360

## Constant Columns
- experimental_setup
- dsps

## Main Targets
### total_area
count      864.000000
mean      5910.313657
std       2225.814085
min       1118.000000
25%       5329.000000
50%       6167.000000
75%       7016.000000
max      12280.000000
Name: total_area, dtype: float64

### control_steps
count    864.000000
mean      70.162037
std       47.896921
min        5.000000
25%       37.000000
50%       58.000000
75%       95.250000
max      258.000000
Name: control_steps, dtype: float64

### frequency_mhz
count    864.000000
mean      85.967189
std       16.859660
min       66.666667
25%       66.887273
50%      100.085479
75%      100.365330
max      106.760046
Name: frequency_mhz, dtype: float64

## Outlier Note
Outliers are reported but not automatically removed. In this project, large values may represent valid high-complexity hardware designs rather than data errors.

## Modeling Note
The modeling dataset is one-hot encoded and ready for baseline regression models. Targets should be selected in the training script, e.g., total_area or control_steps.