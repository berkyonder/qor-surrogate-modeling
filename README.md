# QoR Surrogate Modeling for Early HLS Decision Making

Predicting hardware Quality of Results (QoR), such as area- and latency-related metrics, from early High-Level Synthesis (HLS) information using machine learning.

This project uses **Bambu HLS** and **PolyBench/C** benchmarks to build a structured dataset containing HLS configuration parameters, static C source-code features, intermediate Bambu HLS metrics, and QoR-related target metrics.

## Current Status

### Implemented

- Bambu + PolyBench benchmark automation
- Bambu log parsing into structured metrics
- Static C source-code feature extraction via regex heuristics
- Dataset preprocessing, validation, and visualization
- Baseline regression models (Linear Regression, Random Forest, Gradient Boosting)
- Random split, 5-fold cross-validation, and leave-one-benchmark-out evaluation methods
- Feature importance analysis and interpretation

### Modeling Targets

- `total_area` — predicted hardware area
- `control_steps` — predicted control flow complexity

Further target metrics and backend-level evaluation will be clarified with the project supervisor.

## Dataset Overview

The main dataset contains **864 design points** generated from:

```
18 PolyBench benchmarks
× 2 dataset sizes (MINI, LARGE)
× 2 clock periods (5ns, 10ns)
× 2 memory allocation policies (bram, dram)
× 2 DSP allocation coefficients (0, 8)
× 3 optimization levels (0, 1, 2)
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

### Full Modeling Pipeline

```powershell
python scripts/extract_static_features.py
python scripts/preprocess_dataset.py
python scripts/visualize_dataset.py
python scripts/train_baseline.py
python scripts/summarize_modeling_results.py
python scripts/visualize_modeling_results.py
```

### Rebuild Raw Dataset from Bambu Logs

```powershell
Get-ChildItem data/raw_reports/*_clk*_mem*_dsp*_opt*_log.txt | ForEach-Object {
    python scripts/parse_bambu_log.py $_.FullName -o data/extracted_metrics/early_hls_metrics_raw.csv
}
```

## Models and Evaluation

### Baseline Regressors

- **Linear Regression** — baseline interpretability
- **Random Forest Regressor** — ensemble non-linearity and feature importance
- **Gradient Boosting Regressor** — boosted iterative refinement

### Evaluation Methods

| Method | Description |
|--------|-------------|
| **Random Split** | 80/20 train/test split |
| **5-Fold CV** | 5-fold cross-validation with mean ± std metrics |
| **Leave-One-Benchmark-Out** | Generalization across unseen benchmarks |

### Feature Modes

The modeling pipeline evaluates two feature configurations:

- **Full Mode**: uses all available early HLS/static features except the target
- **No-Leakage Mode**: removes near-direct proxy features such as `area_est`, `mux_area`, or `states`

## Outputs

Generated outputs are organized as follows:

| Directory | Contents |
|-----------|----------|
| `reports/data_quality/` | Dataset quality reports and validation plots |
| `reports/figures/` | Dataset and modeling visualizations |
| `reports/modeling/` | Model predictions, feature importance, and evaluation metrics |

## Documentation

For detailed setup instructions and toolchain overview, see [docs/initial_setup.md](docs/initial_setup.md).

## Notes

- The current workflow demonstrates dataset generation, preprocessing, visualization, and baseline surrogate modeling using Bambu-level HLS metrics.
- Some configurations can produce identical QoR metrics; these are kept because they represent meaningful design-space observations.
- Feature importance analysis prioritizes no-leakage models to ensure only non-trivial predictive signals are captured.
