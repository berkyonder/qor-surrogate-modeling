#!/usr/bin/env python3
"""
analyze_openroad_physical_correlations.py

Side analysis for physical effects requested by the advisor.

This script correlates extracted OpenROAD physical metrics:
- approximate routed wirelength from final DEF
- route guide indicators
- route DRC report indicators

with:
- final OpenROAD QoR targets
- early HLS/Yosys/static input features

Outputs:
- reports/modeling/openroad_physical_correlation_summary.csv
- reports/modeling/openroad_physical_correlation_summary.md
"""

from pathlib import Path
import pandas as pd
import numpy as np


MODELING_DATASET = Path("data/processed/openroad_modeling_dataset.csv")
PHYSICAL_METRICS = Path("data/extracted_metrics/openroad_physical_metrics.csv")

OUT_CSV = Path("reports/modeling/openroad_physical_correlation_summary.csv")
OUT_MD = Path("reports/modeling/openroad_physical_correlation_summary.md")

PHYSICAL_COLUMNS = [
    "openroad_def_routed_wirelength_microns",
    "openroad_def_coordinate_pair_count",
    "openroad_def_routed_net_count",
    "openroad_route_guide_nonempty_lines",
    "openroad_route_guide_segment_like_lines",
    "openroad_route_drc_violation_keyword_count",
]

FINAL_QOR_COLUMNS = [
    "openroad_synth_chip_area",
    "openroad_critical_path_delay",
    "openroad_total_power",
    "openroad_wns",
    "openroad_tns",
    "openroad_worst_slack",
    "openroad_setup_violation_count",
    "openroad_hold_violation_count",
    "openroad_max_slew_violation_count",
    "openroad_max_cap_violation_count",
]

EARLY_FEATURE_HINTS = [
    "area_est",
    "mux_area",
    "total_area",
    "registers",
    "flipflops",
    "control_steps",
    "states",
    "modules_instantiated",
    "performance_conflicts",
    "num_array_accesses",
    "num_arithmetic_ops",
    "num_mul_ops",
    "num_total_loops",
    "max_loop_depth",
    "yosys_num_cells",
    "yosys_total_cells",
    "yosys_num_wires",
    "yosys_total_wires",
    "yosys_num_wire_bits",
    "yosys_total_wire_bits",
]


def safe_corr(x: pd.Series, y: pd.Series):
    pair = pd.concat([x, y], axis=1).dropna()

    if len(pair) < 3:
        return np.nan

    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan

    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not MODELING_DATASET.exists():
        raise FileNotFoundError(f"Missing modeling dataset: {MODELING_DATASET}")

    if not PHYSICAL_METRICS.exists():
        raise FileNotFoundError(f"Missing physical metrics: {PHYSICAL_METRICS}")

    model_df = pd.read_csv(MODELING_DATASET)
    phys_df = pd.read_csv(PHYSICAL_METRICS)

    if "config_name" not in phys_df.columns:
        raise ValueError("openroad_physical_metrics.csv must contain config_name")

    # openroad_modeling_dataset may not keep config_name after preprocessing.
    # So use openroad_qor_targets as the bridge if needed.
    qor_path = Path("data/extracted_metrics/openroad_qor_targets.csv")
    if "config_name" in model_df.columns:
        merged = model_df.merge(phys_df, on="config_name", how="left")
    elif qor_path.exists():
        qor_df = pd.read_csv(qor_path)
        key_cols = [
            "benchmark",
            "dataset_size",
            "clock_period",
            "mem_policy",
            "dsp_coeff",
            "opt_level",
            "openroad_clock_constraint",
            "config_name",
        ]
        key_cols = [c for c in key_cols if c in qor_df.columns]

        # Reconstruct benchmark from one-hot if needed is hard, so physical correlations
        # can be computed directly from openroad_qor_targets + physical metrics.
        merged = qor_df.merge(phys_df, on="config_name", how="left")
    else:
        merged = phys_df.copy()

    available_physical = [c for c in PHYSICAL_COLUMNS if c in merged.columns]
    available_qor = [c for c in FINAL_QOR_COLUMNS if c in merged.columns]
    available_early = [c for c in EARLY_FEATURE_HINTS if c in merged.columns]

    rows = []

    for phys_col in available_physical:
        for other_col in available_qor + available_early:
            if phys_col == other_col:
                continue

            if not pd.api.types.is_numeric_dtype(merged[phys_col]):
                continue

            if not pd.api.types.is_numeric_dtype(merged[other_col]):
                continue

            corr = safe_corr(merged[phys_col], merged[other_col])

            rows.append({
                "physical_metric": phys_col,
                "compared_metric": other_col,
                "metric_group": "final_qor" if other_col in available_qor else "early_feature",
                "pearson_corr": corr,
                "abs_pearson_corr": abs(corr) if pd.notna(corr) else np.nan,
                "n": pd.concat([merged[phys_col], merged[other_col]], axis=1).dropna().shape[0],
            })

    corr_df = pd.DataFrame(rows)

    if not corr_df.empty:
        corr_df = corr_df.sort_values("abs_pearson_corr", ascending=False)

    corr_df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# OpenROAD Physical-Effect Correlation Summary\n")
    lines.append(f"- Rows analyzed: {len(merged)}")
    lines.append(f"- Physical metrics analyzed: {len(available_physical)}")
    lines.append("")

    lines.append("## Physical metrics")
    for c in available_physical:
        lines.append(f"- `{c}`")
    lines.append("")

    if not corr_df.empty:
        lines.append("## Strongest correlations with final QoR metrics")
        top_qor = corr_df[corr_df["metric_group"] == "final_qor"].dropna(subset=["pearson_corr"]).head(10)

        if top_qor.empty:
            lines.append("- No non-constant final-QoR correlations available.")
        else:
            lines.append("")
            lines.append("| Physical metric | QoR metric | Pearson correlation |")
            lines.append("|---|---|---:|")
            for _, row in top_qor.iterrows():
                lines.append(
                    f"| `{row['physical_metric']}` | `{row['compared_metric']}` | {row['pearson_corr']:.4f} |"
                )
        lines.append("")

        lines.append("## Strongest correlations with early-stage features")
        top_early = corr_df[corr_df["metric_group"] == "early_feature"].dropna(subset=["pearson_corr"]).head(10)

        if top_early.empty:
            lines.append("- No non-constant early-feature correlations available.")
        else:
            lines.append("")
            lines.append("| Physical metric | Early feature | Pearson correlation |")
            lines.append("|---|---|---:|")
            for _, row in top_early.iterrows():
                lines.append(
                    f"| `{row['physical_metric']}` | `{row['compared_metric']}` | {row['pearson_corr']:.4f} |"
                )
        lines.append("")

    lines.append("## Interpretation note")
    lines.append(
        "This is an auxiliary physical-effect analysis. The main surrogate targets remain "
        "final OpenROAD area and critical-path delay. Wirelength, route-guide indicators, "
        "and DRC report indicators are inspected to evaluate whether physical implementation "
        "effects are related to early-stage features and final QoR."
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== OpenROAD physical correlation analysis complete ===")
    print(f"CSV: {OUT_CSV}")
    print(f"Markdown: {OUT_MD}")

    if not corr_df.empty:
        print("\nTop correlations:")
        print(corr_df.head(15))


if __name__ == "__main__":
    main()