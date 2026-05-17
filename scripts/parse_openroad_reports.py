#!/usr/bin/env python3
"""
preprocess_dataset.py

Preprocessing and data-quality reporting for the QoR surrogate modeling dataset.

Inputs:
- data/extracted_metrics/early_hls_metrics_raw.csv
- data/extracted_metrics/static_code_features.csv
- data/extracted_metrics/yosys_qor_targets.csv
- data/extracted_metrics/openroad_qor_targets.csv

Outputs:
- data/processed/early_hls_metrics_clean.csv
- data/processed/modeling_dataset.csv
- data/processed/openroad_modeling_dataset.csv
- reports/data_quality/*.csv
- reports/data_quality/dataset_quality_summary.md
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/extracted_metrics/early_hls_metrics_raw.csv")
STATIC_FEATURES_PATH = Path("data/extracted_metrics/static_code_features.csv")
YOSYS_FEATURES_PATH = Path("data/extracted_metrics/yosys_qor_targets.csv")
OPENROAD_TARGETS_PATH = Path("data/extracted_metrics/openroad_qor_targets.csv")

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT_DIR = Path("reports/data_quality")

MERGE_KEYS = [
    "benchmark",
    "dataset_size",
    "clock_period",
    "mem_policy",
    "dsp_coeff",
    "opt_level",
]

ID_COLUMNS = [
    "design_name",
    "log_file",
    "yosys_report_file",
    "config_name",
]

CONSTANT_OR_UNUSED_COLUMNS = [
    "dsps",
    "experimental_setup",
]

CATEGORICAL_COLUMNS = [
    "benchmark",
    "dataset_size",
    "mem_policy",
    "opt_level",
]

NUMERIC_COLUMNS = [
    "clock_period",
    "dsp_coeff",
    "control_steps",
    "min_slack",
    "frequency_mhz",
    "states",
    "modules_instantiated",
    "performance_conflicts",
    "flipflops",
    "area_est",
    "mux_area",
    "total_area",
    "registers",
]

OPENROAD_OUTPUT_COLUMNS = [
    "openroad_tns",
    "openroad_wns",
    "openroad_worst_slack",
    "openroad_clock_period_min",
    "openroad_fmax",
    "openroad_critical_path_delay",
    "openroad_critical_path_slack",
    "openroad_setup_violation_count",
    "openroad_hold_violation_count",
    "openroad_max_slew_violation_count",
    "openroad_max_cap_violation_count",
    "openroad_total_power",
    "openroad_synth_chip_area",
]

OPENROAD_TARGETS = [
    "openroad_synth_chip_area",
    "openroad_critical_path_delay",
    "openroad_total_power",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in CATEGORICAL_COLUMNS + ID_COLUMNS + ["experimental_setup"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in NUMERIC_COLUMNS + ["dsps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_merge_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_cols = ["benchmark", "dataset_size", "mem_policy", "opt_level"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "clock_period" in df.columns:
        df["clock_period"] = pd.to_numeric(df["clock_period"], errors="coerce").astype("Int64")

    if "dsp_coeff" in df.columns:
        df["dsp_coeff"] = pd.to_numeric(df["dsp_coeff"], errors="coerce").astype(float)

    return df


def make_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": df.isna().mean().values * 100,
    }).sort_values("missing_count", ascending=False)


def make_duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    config_cols = [c for c in MERGE_KEYS if c in df.columns]

    metric_cols = [
        "control_steps",
        "min_slack",
        "frequency_mhz",
        "states",
        "modules_instantiated",
        "performance_conflicts",
        "flipflops",
        "area_est",
        "mux_area",
        "total_area",
        "registers",
        "dsps",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    report = {
        "rows": len(df),
        "columns": df.shape[1],
        "full_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_configurations": int(df.duplicated(subset=config_cols).sum()) if config_cols else 0,
        "duplicate_metric_vectors": int(df.duplicated(subset=metric_cols).sum()) if metric_cols else 0,
    }

    return pd.DataFrame([report])


def make_constant_column_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "unique_values": df[col].nunique(dropna=False),
            "is_constant": df[col].nunique(dropna=False) <= 1,
        })

    return pd.DataFrame(rows).sort_values(["is_constant", "unique_values"], ascending=[False, True])


def make_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= 1:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_percent": mask.mean() * 100,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def make_group_summaries(df: pd.DataFrame, report_dir: Path) -> None:
    group_cols = [
        "benchmark",
        "dataset_size",
        "clock_period",
        "mem_policy",
        "dsp_coeff",
        "opt_level",
    ]

    target_cols = [
        "total_area",
        "control_steps",
        "frequency_mhz",
        "registers",
        "flipflops",
        "modules_instantiated",
        "performance_conflicts",
        "openroad_synth_chip_area",
        "openroad_critical_path_delay",
        "openroad_total_power",
    ]

    target_cols = [c for c in target_cols if c in df.columns]

    for col in group_cols:
        if col not in df.columns or not target_cols:
            continue

        summary = df.groupby(col)[target_cols].agg(["mean", "std", "min", "max"])
        summary.to_csv(report_dir / f"group_summary_by_{col}.csv")


def make_correlation_reports(df: pd.DataFrame, report_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return

    corr = numeric_df.corr()
    corr.to_csv(report_dir / "correlation_matrix.csv")

    targets = [
        "total_area",
        "control_steps",
        "frequency_mhz",
        "openroad_synth_chip_area",
        "openroad_critical_path_delay",
        "openroad_total_power",
    ]

    for target in targets:
        if target in corr.columns:
            target_corr = (
                corr[target]
                .dropna()
                .sort_values(ascending=False)
                .reset_index()
            )
            target_corr.columns = ["feature", f"correlation_with_{target}"]
            target_corr.to_csv(report_dir / f"correlation_with_{target}.csv", index=False)


def merge_static_features(df: pd.DataFrame) -> pd.DataFrame:
    if not STATIC_FEATURES_PATH.exists():
        print("\n=== Static features not found ===")
        print(f"Skipping static merge: {STATIC_FEATURES_PATH}")
        return df

    static_df = pd.read_csv(STATIC_FEATURES_PATH)
    static_df.columns = static_df.columns.str.strip()

    if "source_file" in static_df.columns:
        static_df = static_df.drop(columns=["source_file"])

    merged = df.merge(static_df, on="benchmark", how="left")

    print("\n=== Static features merged ===")
    print(f"Static feature file: {STATIC_FEATURES_PATH}")
    print(f"Dataset shape after merge: {merged.shape}")

    return merged


def merge_yosys_features(df: pd.DataFrame) -> pd.DataFrame:
    if not YOSYS_FEATURES_PATH.exists():
        print("\n=== Yosys features not found ===")
        print(f"Skipping Yosys merge: {YOSYS_FEATURES_PATH}")
        return df

    yosys_df = pd.read_csv(YOSYS_FEATURES_PATH)
    yosys_df.columns = yosys_df.columns.str.strip()
    yosys_df = normalize_merge_key_types(yosys_df)

    keep_cols = [
        *MERGE_KEYS,
        "yosys_selected_module",
        "yosys_num_wires",
        "yosys_num_wire_bits",
        "yosys_num_pub_wires",
        "yosys_num_pub_wire_bits",
        "yosys_num_memories",
        "yosys_num_memory_bits",
        "yosys_num_processes",
        "yosys_num_cells",
        "yosys_total_modules",
        "yosys_total_wires",
        "yosys_total_wire_bits",
        "yosys_total_pub_wires",
        "yosys_total_pub_wire_bits",
        "yosys_total_memories",
        "yosys_total_memory_bits",
        "yosys_total_processes",
        "yosys_total_cells",
        "yosys_report_file",
    ]

    keep_cols = [c for c in keep_cols if c in yosys_df.columns]
    yosys_df = yosys_df[keep_cols].drop_duplicates(subset=MERGE_KEYS)

    merged = df.merge(yosys_df, on=MERGE_KEYS, how="left")

    print("\n=== Yosys features merged ===")
    print(f"Yosys feature file: {YOSYS_FEATURES_PATH}")
    print(f"Dataset shape after merge: {merged.shape}")
    print(f"Rows with Yosys data: {merged['yosys_total_cells'].notna().sum() if 'yosys_total_cells' in merged.columns else 0}")

    return merged


def merge_openroad_targets(df: pd.DataFrame) -> pd.DataFrame:
    if not OPENROAD_TARGETS_PATH.exists():
        print("\n=== OpenROAD targets not found ===")
        print(f"Skipping OpenROAD merge: {OPENROAD_TARGETS_PATH}")
        return df

    openroad_df = pd.read_csv(OPENROAD_TARGETS_PATH)
    openroad_df.columns = openroad_df.columns.str.strip()
    openroad_df = normalize_merge_key_types(openroad_df)

    keep_cols = [
        *MERGE_KEYS,
        "config_name",
        "openroad_clock_constraint",
        "config_parse_status",
        "openroad_run_status",
        "openroad_parse_status",
        *OPENROAD_OUTPUT_COLUMNS,
    ]

    keep_cols = [c for c in keep_cols if c in openroad_df.columns]
    openroad_df = openroad_df[keep_cols]

    # Keep only successfully parsed OpenROAD rows.
    if "openroad_parse_status" in openroad_df.columns:
        openroad_df = openroad_df[openroad_df["openroad_parse_status"] == "ok"].copy()

    merged = df.merge(openroad_df, on=MERGE_KEYS, how="left")

    print("\n=== OpenROAD targets merged ===")
    print(f"OpenROAD target file: {OPENROAD_TARGETS_PATH}")
    print(f"Dataset shape after merge: {merged.shape}")
    print(f"Rows with OpenROAD labels: {merged['openroad_critical_path_delay'].notna().sum() if 'openroad_critical_path_delay' in merged.columns else 0}")

    return merged


def create_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()

    clean = clean.drop_duplicates().reset_index(drop=True)

    drop_cols = ID_COLUMNS + CONSTANT_OR_UNUSED_COLUMNS
    clean = clean.drop(columns=[c for c in drop_cols if c in clean.columns])

    return clean


def create_modeling_dataset(clean: pd.DataFrame) -> pd.DataFrame:
    modeling = clean.copy()

    categorical_existing = [c for c in CATEGORICAL_COLUMNS if c in modeling.columns]

    modeling = pd.get_dummies(
        modeling,
        columns=categorical_existing,
        drop_first=False,
        dtype=int,
    )

    return modeling


def create_openroad_modeling_dataset(clean: pd.DataFrame) -> pd.DataFrame:
    required = [
        "openroad_synth_chip_area",
        "openroad_critical_path_delay",
    ]

    missing = [c for c in required if c not in clean.columns]
    if missing:
        return pd.DataFrame()

    openroad_df = clean.dropna(subset=required).copy()

    # OpenROAD status/config metadata should not be model features.
    drop_cols = [
        "config_name",
        "config_parse_status",
        "openroad_run_status",
        "openroad_parse_status",
    ]
    openroad_df = openroad_df.drop(columns=[c for c in drop_cols if c in openroad_df.columns])

    categorical_existing = [c for c in CATEGORICAL_COLUMNS if c in openroad_df.columns]

    openroad_df = pd.get_dummies(
        openroad_df,
        columns=categorical_existing,
        drop_first=False,
        dtype=int,
    )

    return openroad_df


def write_summary(
    df: pd.DataFrame,
    clean: pd.DataFrame,
    openroad_modeling: pd.DataFrame,
    report_dir: Path,
    missing_report: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    constant_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Dataset Quality Summary\n")

    lines.append("## Raw Dataset")
    lines.append(f"- Rows: {df.shape[0]}")
    lines.append(f"- Columns: {df.shape[1]}\n")

    lines.append("## Clean Dataset")
    lines.append(f"- Rows: {clean.shape[0]}")
    lines.append(f"- Columns: {clean.shape[1]}\n")

    if not openroad_modeling.empty:
        lines.append("## OpenROAD Modeling Dataset")
        lines.append(f"- Rows: {openroad_modeling.shape[0]}")
        lines.append(f"- Columns: {openroad_modeling.shape[1]}")
        lines.append(
            "- This dataset contains only configurations with successfully parsed OpenROAD final QoR labels.\n"
        )

    lines.append("## Missing Values")
    lines.append(f"- Total missing values: {int(df.isna().sum().sum())}\n")

    lines.append("## Duplicate Analysis")
    for col, value in duplicate_report.iloc[0].items():
        lines.append(f"- {col}: {value}")
    lines.append("")

    lines.append("## Constant Columns")
    constants = constant_report[constant_report["is_constant"]]["column"].tolist()
    if constants:
        for col in constants:
            lines.append(f"- {col}")
    else:
        lines.append("- No constant columns detected.")
    lines.append("")

    lines.append("## Main Bambu Targets")
    for target in ["total_area", "control_steps", "frequency_mhz"]:
        if target in df.columns:
            lines.append(f"### {target}")
            lines.append(str(df[target].describe()))
            lines.append("")

    lines.append("## Main OpenROAD Targets")
    for target in OPENROAD_TARGETS:
        if target in df.columns:
            lines.append(f"### {target}")
            lines.append(str(df[target].dropna().describe()))
            lines.append("")

    lines.append("## Outlier Note")
    lines.append(
        "Outliers are reported but not automatically removed. In this project, "
        "large values may represent valid high-complexity hardware designs rather than data errors."
    )
    lines.append("")

    lines.append("## Modeling Note")
    lines.append(
        "The general modeling dataset is one-hot encoded for the full Bambu/HLS dataset. "
        "The OpenROAD modeling dataset is a labeled subset for final ASIC QoR prediction."
    )

    (report_dir / "dataset_quality_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    df = normalize_merge_key_types(df)

    df = merge_static_features(df)
    df = normalize_merge_key_types(df)

    df = merge_yosys_features(df)
    df = normalize_merge_key_types(df)

    df = merge_openroad_targets(df)
    df = normalize_merge_key_types(df)

    missing_report = make_missing_report(df)
    duplicate_report = make_duplicate_report(df)
    constant_report = make_constant_column_report(df)
    outlier_report = make_outlier_report(df)

    missing_report.to_csv(args.report_dir / "missing_values_report.csv", index=False)
    duplicate_report.to_csv(args.report_dir / "duplicate_report.csv", index=False)
    constant_report.to_csv(args.report_dir / "constant_columns_report.csv", index=False)
    outlier_report.to_csv(args.report_dir / "outlier_report_iqr.csv", index=False)

    make_group_summaries(df, args.report_dir)
    make_correlation_reports(df, args.report_dir)

    clean = create_clean_dataset(df)
    modeling = create_modeling_dataset(clean)
    openroad_modeling = create_openroad_modeling_dataset(clean)

    clean.to_csv(args.output_dir / "early_hls_metrics_clean.csv", index=False)
    modeling.to_csv(args.output_dir / "modeling_dataset.csv", index=False)

    if not openroad_modeling.empty:
        openroad_modeling.to_csv(
            args.output_dir / "openroad_modeling_dataset.csv",
            index=False,
        )

    write_summary(
        df=df,
        clean=clean,
        openroad_modeling=openroad_modeling,
        report_dir=args.report_dir,
        missing_report=missing_report,
        duplicate_report=duplicate_report,
        constant_report=constant_report,
        outlier_report=outlier_report,
    )

    print("\n=== Preprocessing complete ===")
    print(f"Input dataset: {args.input}")
    print(f"Clean dataset: {args.output_dir / 'early_hls_metrics_clean.csv'}")
    print(f"Modeling dataset: {args.output_dir / 'modeling_dataset.csv'}")

    if not openroad_modeling.empty:
        print(f"OpenROAD modeling dataset: {args.output_dir / 'openroad_modeling_dataset.csv'}")
    else:
        print("OpenROAD modeling dataset: not generated yet")

    print(f"Reports: {args.report_dir}")


if __name__ == "__main__":
    main()