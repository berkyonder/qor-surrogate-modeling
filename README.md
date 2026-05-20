# QoR Surrogate Modeling for Early HLS-to-ASIC Decision Making

Predicting final hardware Quality of Results (QoR), such as **ASIC area**, **critical-path delay**, and optional **power**, from early High-Level Synthesis (HLS), static source-code, and synthesis-level features using machine learning.

This project uses **Bambu HLS** and **PolyBench/C** benchmarks to build a structured dataset containing HLS configuration parameters, static C source-code features, intermediate Bambu HLS metrics, and QoR-related target metrics.

## Current Status

### Implemented

- Bambu + PolyBench benchmark automation
- Bambu log parsing into structured early HLS metrics
- Static C source-code feature extraction using regex-based heuristics
- Yosys synthesis report parsing
- OpenROAD final report parsing
- OpenROAD physical/timing/electrical effect extraction
- Dataset preprocessing, validation, and visualization
- Baseline regression models (Linear Regression, Random Forest, Gradient Boosting)
- Random split, 5-fold cross-validation, and leave-one-benchmark-out evaluation methods
- Feature importance analysis and interpretation

### Modeling Targets

- `total_area` — predicted hardware area
- `control_steps` — predicted control flow complexity

Further target metrics and backend-level evaluation will be clarified with the project supervisor.

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

### Main Data Files

| File | Purpose |
|------|---------|
| `data/extracted_metrics/early_hls_metrics_raw.csv` | Raw Bambu metrics parsed from logs |
| `data/extracted_metrics/static_code_features.csv` | Static C code features (loops, operations, etc.) |
| `data/processed/early_hls_metrics_clean.csv` | Cleaned dataset after preprocessing |
| `data/processed/modeling_dataset.csv` | One-hot encoded dataset ready for modeling |

### Feature Groups

The dataset includes:

- **HLS Configuration Features**: `dataset_size`, `clock_period`, `mem_policy`, `dsp_coeff`, `opt_level`
- **Static Code Features**: loop counts, source lines, array accesses, arithmetic operation counts
- **Intermediate HLS Metrics**: `control_steps`, `frequency_mhz`, `states`, `modules_instantiated`, `flipflops`, `area_est`, `mux_area`, `registers`

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

Future work can add explicit routed wirelength, congestion, and detailed DRC parsing from DEF/routing reports.

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
