# Dataset Quality Summary

## Raw Dataset
- Rows: 865
- Columns: 73

## Clean Dataset
- Rows: 865
- Columns: 67

## OpenROAD Modeling Dataset
- Rows: 18
- Columns: 79
- This dataset contains only configurations with successfully parsed OpenROAD final QoR labels.

## Missing Values
- Total missing values: 30161

## Duplicate Analysis
- rows: 865
- columns: 73
- full_duplicate_rows: 0
- duplicate_configurations: 1
- duplicate_metric_vectors: 361

## Constant Columns
- experimental_setup
- dsps
- num_while_loops

## Main Bambu Targets
### total_area
count      865.000000
mean      5910.852023
std       2224.581976
min       1118.000000
25%       5330.000000
50%       6172.000000
75%       7016.000000
max      12280.000000
Name: total_area, dtype: float64

### control_steps
count    865.000000
mean      70.123699
std       47.882473
min        5.000000
25%       37.000000
50%       58.000000
75%       95.000000
max      258.000000
Name: control_steps, dtype: float64

### frequency_mhz
count    865.000000
mean      85.983834
std       16.857011
min       66.666667
25%       66.887273
50%      100.170958
75%      100.365330
max      106.760046
Name: frequency_mhz, dtype: float64

## Main OpenROAD Targets
### openroad_synth_chip_area
count       18.000000
mean     42523.336333
std      14719.355560
min      11447.842000
25%      37039.901500
50%      44335.683000
75%      49241.055500
max      75947.788000
Name: openroad_synth_chip_area, dtype: float64

### openroad_critical_path_delay
count    18.000000
mean      4.407739
std       1.344837
min       2.077100
25%       3.054925
50%       5.258450
75%       5.427850
max       5.810800
Name: openroad_critical_path_delay, dtype: float64

### openroad_total_power
count    18.000000
mean      0.029402
std       0.069383
min       0.002190
25%       0.005203
50%       0.005700
75%       0.010242
max       0.295000
Name: openroad_total_power, dtype: float64

## Outlier Note
Outliers are reported but not automatically removed. In this project, large values may represent valid high-complexity hardware designs rather than data errors.

## Modeling Note
The general modeling dataset is one-hot encoded for the full Bambu/HLS dataset. The OpenROAD modeling dataset is a labeled subset for final ASIC QoR prediction.