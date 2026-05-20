# Final OpenROAD QoR Modeling Summary

## Dataset
- OpenROAD-labeled rows: 18
- Columns: 79
- OpenROAD clock constraints:
  - 2.0: 1 rows
  - 10.0: 17 rows

## Main Direct-Target Results

These results use direct OpenROAD targets, not normalized target ratios.
For small group folds, MAE and relative MAE are emphasized over R2.

| Target | Preferred experiment | Evaluation | MAE | RMSE | Relative MAE | R2 | Model | Feature set |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Final ASIC area | rigorous_direct_target | groupkfold_by_benchmark | 6637.1103 | 6637.1103 | 0.1534 | 0.0000 | elasticnet_a1.0_l10.8 | raw |
| Final critical path delay | rigorous_direct_target | groupkfold_by_benchmark | 0.7632 | 0.7633 | 0.1871 | -1429.4650 | elasticnet_a0.01_l10.2 | ratios |
| Final total power | ablation_direct_target | groupkfold_by_benchmark | 0.0157 | 0.0177 | 0.3770 | -2.4530 | elasticnet_a0.001_l10.2 | yosys_only |

## Best Group-Aware Results Across All Experiments

| Target | Experiment | MAE | RMSE | Relative MAE | R2 | Model | Feature set | Notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| Final ASIC area | rigorous_direct_target | 6637.1103 | 6637.1103 | 0.1534 | 0.0000 | elasticnet_a1.0_l10.8 | raw | direct target |
| Final critical path delay | rigorous_direct_target | 0.7632 | 0.7633 | 0.1871 | -1429.4650 | elasticnet_a0.01_l10.2 | ratios | direct target |
| Final total power | ablation_direct_target | 0.0157 | 0.0177 | 0.3770 | -2.4530 | elasticnet_a0.001_l10.2 | yosys_only | direct target |

## Best Random-Split Results Across All Experiments

Random split is useful as an optimistic estimate, but it is less rigorous than group-aware validation.

| Target | Experiment | MAE | RMSE | Relative MAE | R2 | Model |
|---|---:|---:|---:|---:|---:|---|
| Final ASIC area | eda_aware_normalized_target | 3048.0266 | 3419.5953 | 0.0587 | 0.9514 | gpr_matern |
| Final critical path delay | feature_engineering | 0.4784 | 0.5564 | N/A | 0.7839 | gradient_boosting |
| Final total power | rigorous_direct_target | 0.0175 | 0.0240 | 0.5404 | 0.3687 | elasticnet_a0.1_l10.5 |

## Interpretation

- Area is the most predictable final OpenROAD QoR target.
- Timing prediction is moderately successful, especially with structural HLS/Yosys features and regularized models.
- Power remains exploratory because it is sensitive to clock constraints and the dataset contains limited clock-condition diversity.
- Group-aware validation is treated as the most realistic test because it evaluates generalization to unseen benchmarks.
- R2 is unstable for leave-one-benchmark/group folds with very small test sets, so MAE and relative MAE are emphasized.