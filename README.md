# QoR Surrogate Modeling for Early HLS-to-ASIC Decision Making

Predicting final hardware Quality of Results (QoR), such as **ASIC area**, **critical-path delay**, and optional **power**, from early High-Level Synthesis (HLS), static source-code, and synthesis-level features using machine learning.

This project uses **Bambu HLS**, **PolyBench/C**, **Yosys**, and **OpenROAD-flow-scripts** with the **Nangate45** platform to build an initial HLS-to-ASIC surrogate modeling workflow.

```text
C source code
    ↓
Static source-code features
    ↓
Bambu HLS metrics
    ↓
Yosys synthesis statistics
    ↓
OpenROAD final ASIC QoR labels
    ↓
Machine-learning surrogate models
```

The objective is to study whether early-stage design information contains predictive signal for final physical-design QoR, reducing the need to run full OpenROAD place-and-route for every design-space point.

## Current Status

### Implemented

- Bambu + PolyBench benchmark automation
- Bambu log parsing into structured early HLS metrics
- Static C source-code feature extraction using regex-based heuristics
- Yosys synthesis report parsing
- OpenROAD final report parsing
- OpenROAD physical/timing/electrical effect extraction
- Dataset preprocessing, validation, and visualization
- Baseline ML models
- Feature-engineered ML models
- Direct-target modeling with regularization and feature selection
- EDA-aware exploratory modeling
- Feature-group ablation studies
- Random split, k-fold cross-validation, and group-aware unseen-benchmark evaluation
- Final consolidated modeling summary and target-distribution summaries

## Final Modeling Setup

### Input Features

The final surrogate model uses early-stage features available before the expensive OpenROAD backend run.

| Feature Group | Examples |
|---|---|
| HLS configuration features | `clock_period`, `dataset_size`, `mem_policy`, `dsp_coeff`, `opt_level` |
| Static source-code features | loop counts, source lines, arithmetic operation counts, array accesses |
| Bambu/HLS metrics | `control_steps`, `states`, `area_est`, `mux_area`, `registers`, `flipflops` |
| Yosys synthesis metrics | cell counts, wire counts, wire bits, module counts |
| Derived feature-engineering inputs | optional ratios computed only from pre-OpenROAD code/HLS/Yosys features |

### Target Labels

The main final labels are parsed from OpenROAD reports.

| Target | Meaning | Role |
|---|---|---|
| `openroad_synth_chip_area` | Final synthesized ASIC chip/cell area | Main target |
| `openroad_critical_path_delay` | Final OpenROAD critical-path delay | Main timing target |
| `openroad_total_power` | Final estimated total power | Optional / exploratory target |

Additional physical/timing closure metrics are parsed for analysis.

| Physical Effect Metric | Description |
|---|---|
| `openroad_wns` | Worst negative slack |
| `openroad_tns` | Total negative slack |
| `openroad_worst_slack` | Worst reported slack |
| `openroad_setup_violation_count` | Number of setup violations |
| `openroad_hold_violation_count` | Number of hold violations |
| `openroad_max_slew_violation_count` | Number of max slew violations |
| `openroad_max_cap_violation_count` | Number of max capacitance violations |

These metrics address backend physical/timing closure effects. The flow also extracts approximate routed wirelength from final DEF files, route-guide indicators, and route-DRC indicators from OpenROAD routing reports.

## Dataset Overview

### Full Early HLS Dataset

The early HLS dataset contains **864 design points** generated from Bambu/PolyBench design-space exploration.

```text
18 PolyBench benchmarks
× 2 dataset sizes
× 2 clock periods
× 2 memory allocation policies
× 2 DSP allocation coefficients
× 3 optimization levels
= 864 configurations
```

### OpenROAD-Labeled Backend Subset

Full OpenROAD place-and-route is significantly more expensive than early HLS/Yosys extraction. Therefore, a smaller backend-labeled subset was generated for final ASIC QoR prediction.

| Dataset | Rows | Description |
|---|---:|---|
| Full Bambu/HLS dataset | 864 | Early-stage design-space points |
| Yosys-labeled subset | 79 | Designs with parsed Yosys synthesis statistics |
| OpenROAD-labeled subset | 18 | Final ASIC QoR labels from OpenROAD |
| Benchmark-diverse OpenROAD runs | 17 | One selected backend run per benchmark at 10 ns |
| Additional aggressive-clock run | 1 | One extra `orclk2` run for clock-sensitivity exploration |

### OpenROAD Target Distributions

| Target | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Final ASIC area | 42,523.3 | 44,335.7 | 11,447.8 | 75,947.8 |
| Critical-path delay | 4.4077 ns | 5.2585 ns | 2.0771 ns | 5.8108 ns |
| Total power | 0.0294 W | 0.0057 W | 0.00219 W | 0.295 W |

The power target is highly skewed due to the aggressive-clock outlier. For this reason, power prediction is treated as exploratory.

## Main Results

The main reported results use direct OpenROAD targets, not normalized target ratios.

The most relevant evaluation is group-aware validation by benchmark, because it tests generalization to unseen benchmarks. Since group folds are small, MAE and relative MAE are emphasized over R2.

| Target | Evaluation | MAE | Relative MAE | Interpretation |
|---|---|---:|---:|---|
| Final ASIC area | GroupKFold by benchmark | 6637.11 | 15.34% | Lowest relative error among evaluated targets |
| Critical-path delay | GroupKFold by benchmark | 0.763 ns | 18.71% | Moderately predictive timing result |
| Total power | GroupKFold by benchmark | 0.0157 W | 37.70% | Exploratory due to target skew and clock sensitivity |

### Interpretation

- Among the evaluated targets, final ASIC area showed the lowest relative error.
- Critical-path delay showed moderate predictability from structural HLS/Yosys features.
- Power prediction was less stable due to clock sensitivity and skewed target distribution.
- Group-aware validation is treated as the most realistic evaluation because it tests unseen-benchmark generalization.
- The results indicate that early HLS, static source-code, and Yosys synthesis features contain predictive information for final OpenROAD ASIC QoR.
- The results should be interpreted as a feasibility study because the OpenROAD-labeled subset contains 18 backend runs.

## Modeling Experiments

The project includes several modeling stages.

### 1) Baseline Models

Initial models trained on direct OpenROAD targets:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 2) Feature-Engineered Models

Additional experiments using:

- SelectKBest
- PCA
- Ridge Regression
- Gaussian Process Regression
- Random Forest
- Gradient Boosting

### 3) Direct-Target Regularized Models

The main direct-target experiments use:

- Elastic Net
- Ridge Regression
- PLS Regression
- Gaussian Process Regression
- Robust feature selection
- Optional derived ratios from pre-OpenROAD features
- Group-aware validation

### 4) EDA-Aware Exploratory Models

Exploratory models tested normalized/residual targets such as:

- `final_area / area_est`
- `critical_path_delay / OpenROAD input clock constraint`
- `total_power / area_est`

These are not used as the main results, because the main project conclusion is based on direct OpenROAD targets. They are retained as exploratory analysis.

### 5) Ablation Study

The ablation study compares feature groups.

| Feature Set | Purpose |
|---|---|
| `code_only` | Static source-code features only |
| `hls_only` | Bambu/HLS metrics only |
| `yosys_only` | Yosys synthesis statistics only |
| `hls_plus_yosys` | Combined HLS and synthesis features |
| `ratios_only` | Derived ratios from pre-OpenROAD features |
| `all_direct_features` | All valid pre-OpenROAD features |

Ablation results suggest:

- Area benefits from combined early-stage features.
- Timing can be predicted meaningfully from HLS structural metrics.
- Power is most associated with synthesis-level/Yosys features, but remains the least stable target.

## Workflow

### Full Processing and Modeling Pipeline

```bash
python scripts/extract_static_features.py
python scripts/parse_yosys_reports.py
python scripts/parse_openroad_reports.py
python scripts/preprocess_dataset.py

python scripts/train_baseline.py
python scripts/train_feature_engineered_models.py
python scripts/train_rigorous_feature_models.py
python scripts/train_eda_aware_models.py
python scripts/train_ablation_models.py

python scripts/create_final_modeling_summary.py
python scripts/summarize_target_distributions.py
python scripts/summarize_openroad_physical_effects.py
python scripts/plot_final_openroad_results.py
```

### Rebuild Raw Dataset from Bambu Logs

```powershell
Get-ChildItem data/raw_reports/*_clk*_mem*_dsp*_opt*_log.txt | ForEach-Object {
    python scripts/parse_bambu_log.py $_.FullName -o data/extracted_metrics/early_hls_metrics_raw.csv
}
```

## Main Data Files

| File | Purpose |
|---|---|
| `data/extracted_metrics/early_hls_metrics_raw.csv` | Raw Bambu metrics parsed from logs |
| `data/extracted_metrics/static_code_features.csv` | Static C source-code features |
| `data/extracted_metrics/yosys_qor_targets.csv` | Parsed Yosys synthesis statistics |
| `data/extracted_metrics/openroad_qor_targets.csv` | Parsed OpenROAD final QoR and physical metrics |
| `data/processed/early_hls_metrics_clean.csv` | Cleaned HLS/static/Yosys/OpenROAD merged dataset |
| `data/processed/modeling_dataset.csv` | General one-hot encoded modeling dataset |
| `data/processed/openroad_modeling_dataset.csv` | OpenROAD-labeled modeling dataset |

## Reports and Outputs

| Directory / File | Contents |
|---|---|
| `reports/data_quality/` | Dataset quality reports, missing values, correlations, group summaries |
| `reports/figures/` | Dataset visualizations |
| `reports/modeling/openroad_all_model_results.csv` | Baseline model results |
| `reports/modeling/openroad_feature_engineering_results.csv` | Feature engineering experiment results |
| `reports/modeling/openroad_rigorous_feature_results.csv` | Direct-target regularized model results |
| `reports/modeling/openroad_eda_aware_results.csv` | Exploratory EDA-aware normalized-target results |
| `reports/modeling/openroad_ablation_results.csv` | Feature-group ablation study results |
| `reports/modeling/final_openroad_modeling_summary.md` | Final consolidated modeling summary |
| `reports/modeling/openroad_target_distribution_summary.md` | Target distribution and scale summary |
| `reports/modeling/openroad_physical_effects_summary.md` | Physical/timing closure effect summary |
| `data/extracted_metrics/openroad_physical_metrics.csv` | Parsed DEF wirelength, route-guide, and route-DRC physical metrics |
| `reports/modeling/openroad_wirelength_drc_summary.md` | Summary of routed wirelength and route-DRC indicators |
| `reports/modeling/openroad_physical_correlation_summary.md` | Correlation analysis between physical effects and final QoR |
| `reports/modeling/final_plots/` | Final target distribution plots |

## Physical Effects Summary

The OpenROAD parser currently inspects the following physical/timing closure effects:

- WNS
- TNS
- Worst slack
- Minimum feasible clock period
- Fmax
- Critical-path slack
- Setup violation count
- Hold violation count
- Max slew violation count
- Max capacitance violation count

Current OpenROAD physical-effect summary:

| Metric | Total Violations | Max per Design |
|---|---:|---:|
| Setup violations | 0 | 0 |
| Hold violations | 0 | 0 |
| Max slew violations | 0 | 0 |
| Max capacitance violations | 3 | 2 |

The physical-effect analysis was extended with routed wirelength and routing/DRC indicators extracted from OpenROAD outputs. All 18 OpenROAD runs include final DEF, route guide, and route DRC reports. These metrics are not used as main model inputs because they are post-routing quantities, but they help explain final QoR behavior. Routed wirelength and routing-complexity indicators show strong correlation with final area and moderate association with critical-path delay, making them plausible auxiliary prediction targets.

Physical-effect correlation highlights:

| Physical metric | Related QoR metric | Pearson correlation |
|---|---|---:|
| Routed net count | Final ASIC area | 0.9978 |
| Route-guide nonempty lines | Final ASIC area | 0.9974 |
| Route-guide segment-like lines | Final ASIC area | 0.9968 |
| Approximate routed wirelength | Final ASIC area | 0.9315 |
| DEF coordinate pair count | Critical-path delay | 0.7149 |
| Route-guide segment-like lines | Critical-path delay | 0.7083 |

## Key Takeaways

- Early HLS, static code, and Yosys synthesis features contain measurable predictive information for final OpenROAD ASIC QoR.
- Final area prediction achieved approximately **15.34% relative MAE** under unseen-benchmark validation.
- Critical-path delay prediction achieved approximately **18.71% relative MAE** under unseen-benchmark validation.
- Power prediction remains exploratory because it is more sensitive to clock constraints and physical implementation details.
- The OpenROAD-labeled dataset is small due to backend runtime cost; additional backend-labeled samples are expected to improve robustness.
- The project provides an initial HLS-to-ASIC surrogate modeling workflow and evaluates its feasibility on a small OpenROAD-labeled subset.

## Documentation

For detailed setup instructions and toolchain overview, see [docs/initial_setup.md](docs/initial_setup.md).

## Notes

- The full early-stage Bambu/HLS dataset contains 864 design-space points.
- The final ASIC/OpenROAD-labeled subset contains 18 backend runs.
- Direct OpenROAD targets are used as the main modeling targets.
- Normalized-target experiments are included only as exploratory EDA-aware analysis.
- Some configurations can produce identical or similar early QoR metrics; these are preserved because they represent meaningful design-space observations.
- Group-aware validation is emphasized because it better reflects generalization to unseen benchmarks.
