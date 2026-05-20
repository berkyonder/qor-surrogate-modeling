#!/usr/bin/env python3
"""
summarize_modeling_results.py

Summarize OpenROAD final ASIC QoR modeling results.

Inputs:
- reports/modeling/openroad_random_split_results.csv
- reports/modeling/openroad_kfold_results.csv
- reports/modeling/openroad_leave_one_benchmark_out_results.csv
- reports/modeling/openroad_all_model_results.csv

Outputs:
- Console summary
- reports/modeling/openroad_modeling_summary.txt
"""

from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("reports/modeling")
SUMMARY_PATH = RESULTS_DIR / "openroad_modeling_summary.txt"

TARGET_NAMES = {
    "openroad_synth_chip_area": "Final ASIC area",
    "openroad_critical_path_delay": "Final critical path delay",
    "openroad_total_power": "Final total power",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def target_label(target: str) -> str:
    return TARGET_NAMES.get(target, target)


def format_float(value, digits=6):
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def summarize_random_split(df: pd.DataFrame) -> str:
    lines = []

    lines.append("=" * 100)
    lines.append("OPENROAD RANDOM SPLIT RESULTS")
    lines.append("=" * 100)
    lines.append("")

    if df.empty:
        lines.append("No random split results available.")
        lines.append("")
        return "\n".join(lines)

    for metric in ["mae", "rmse", "r2"]:
        lines.append(f"### {metric.upper()} ###")

        pivot = df.pivot_table(
            index="target",
            columns="model",
            values=metric,
            aggfunc="mean",
        )

        if not pivot.empty:
            pivot.index = [target_label(t) for t in pivot.index]
            lines.append(str(pivot))
        else:
            lines.append("No data.")

        lines.append("")

    return "\n".join(lines)


def summarize_kfold(df: pd.DataFrame) -> str:
    lines = []

    lines.append("=" * 100)
    lines.append("OPENROAD K-FOLD CROSS-VALIDATION RESULTS")
    lines.append("=" * 100)
    lines.append("")

    if df.empty:
        lines.append("No k-fold results available.")
        lines.append("")
        return "\n".join(lines)

    for metric in ["mae", "rmse", "r2"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        lines.append(f"### {metric.upper()} mean ± std ###")

        temp = df.copy()
        temp["formatted"] = temp.apply(
            lambda row: (
                f"{format_float(row[mean_col], 4)} ± "
                f"{format_float(row[std_col], 4)}"
                if std_col in temp.columns
                else format_float(row[mean_col], 4)
            ),
            axis=1,
        )

        pivot = temp.pivot_table(
            index="target",
            columns="model",
            values="formatted",
            aggfunc="first",
        )

        if not pivot.empty:
            pivot.index = [target_label(t) for t in pivot.index]
            lines.append(str(pivot))
        else:
            lines.append("No data.")

        lines.append("")

    return "\n".join(lines)


def summarize_lobo(df: pd.DataFrame) -> str:
    lines = []

    lines.append("=" * 100)
    lines.append("OPENROAD LEAVE-ONE-BENCHMARK-OUT RESULTS")
    lines.append("=" * 100)
    lines.append("")

    if df.empty:
        lines.append("No leave-one-benchmark-out results available.")
        lines.append("")
        return "\n".join(lines)

    for metric in ["mae", "rmse", "r2"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        lines.append(f"### {metric.upper()} mean ± std ###")

        temp = df.copy()
        temp["formatted"] = temp.apply(
            lambda row: (
                f"{format_float(row[mean_col], 4)} ± "
                f"{format_float(row[std_col], 4)}"
                if std_col in temp.columns
                else format_float(row[mean_col], 4)
            ),
            axis=1,
        )

        pivot = temp.pivot_table(
            index="target",
            columns="model",
            values="formatted",
            aggfunc="first",
        )

        if not pivot.empty:
            pivot.index = [target_label(t) for t in pivot.index]
            lines.append(str(pivot))
        else:
            lines.append("No data.")

        lines.append("")

    return "\n".join(lines)


def summarize_best_models(random_df: pd.DataFrame, kfold_df: pd.DataFrame, lobo_df: pd.DataFrame) -> str:
    lines = []

    lines.append("=" * 100)
    lines.append("BEST OPENROAD MODELS BY TARGET & EVALUATION METHOD")
    lines.append("=" * 100)
    lines.append("")

    sections = [
        ("RANDOM SPLIT", random_df, "r2", "mae", "rmse"),
        ("K-FOLD", kfold_df, "r2_mean", "mae_mean", "rmse_mean"),
        ("LEAVE-ONE-BENCHMARK-OUT", lobo_df, "r2_mean", "mae_mean", "rmse_mean"),
    ]

    for section_name, df, r2_col, mae_col, rmse_col in sections:
        lines.append(f"--- {section_name} ---")
        lines.append("")

        if df.empty or r2_col not in df.columns:
            lines.append("No results available.")
            lines.append("")
            continue

        for target in sorted(df["target"].dropna().unique()):
            target_df = df[df["target"] == target].copy()

            if target_df.empty:
                continue

            # Prefer highest R2. If R2 is all NaN, use lowest MAE.
            if target_df[r2_col].notna().any():
                best = target_df.sort_values(r2_col, ascending=False).iloc[0]
            else:
                best = target_df.sort_values(mae_col, ascending=True).iloc[0]

            lines.append(f"Target: {target_label(target)}")
            lines.append(f"  Raw target column: {target}")
            lines.append(f"  Model: {best['model']}")

            if r2_col in best:
                lines.append(f"  R²: {format_float(best[r2_col], 6)}")
            if mae_col in best:
                lines.append(f"  MAE: {format_float(best[mae_col], 6)}")
            if rmse_col in best:
                lines.append(f"  RMSE: {format_float(best[rmse_col], 6)}")

            if "rows" in best:
                lines.append(f"  Rows: {int(best['rows'])}")
            if "features" in best:
                lines.append(f"  Features: {int(best['features'])}")

            lines.append("")

    return "\n".join(lines)


def summarize_dataset_context() -> str:
    lines = []

    dataset_path = Path("data/processed/openroad_modeling_dataset.csv")
    target_path = Path("data/extracted_metrics/openroad_qor_targets.csv")

    lines.append("=" * 100)
    lines.append("OPENROAD DATASET CONTEXT")
    lines.append("=" * 100)
    lines.append("")

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        lines.append(f"OpenROAD modeling dataset: {dataset_path}")
        lines.append(f"Rows: {df.shape[0]}")
        lines.append(f"Columns: {df.shape[1]}")

        target_cols = [
            "openroad_synth_chip_area",
            "openroad_critical_path_delay",
            "openroad_total_power",
        ]

        for col in target_cols:
            if col in df.columns:
                lines.append("")
                lines.append(f"{target_label(col)} ({col})")
                lines.append(str(df[col].describe()))

        if "openroad_clock_constraint" in df.columns:
            lines.append("")
            lines.append("OpenROAD clock constraints:")
            lines.append(str(df["openroad_clock_constraint"].value_counts().sort_index()))

    else:
        lines.append(f"OpenROAD modeling dataset not found: {dataset_path}")

    lines.append("")

    if target_path.exists():
        target_df = pd.read_csv(target_path)
        lines.append(f"Parsed OpenROAD target file: {target_path}")
        lines.append(f"Rows: {target_df.shape[0]}")

        if "openroad_parse_status" in target_df.columns:
            lines.append("Parse status:")
            lines.append(str(target_df["openroad_parse_status"].value_counts()))

    lines.append("")

    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    random_df = load_csv(RESULTS_DIR / "openroad_random_split_results.csv")
    kfold_df = load_csv(RESULTS_DIR / "openroad_kfold_results.csv")
    lobo_df = load_csv(RESULTS_DIR / "openroad_leave_one_benchmark_out_results.csv")

    sections = [
        summarize_dataset_context(),
        summarize_random_split(random_df),
        summarize_kfold(kfold_df),
        summarize_lobo(lobo_df),
        summarize_best_models(random_df, kfold_df, lobo_df),
        "=" * 100 + "\nSUMMARY COMPLETE\n" + "=" * 100,
    ]

    summary = "\n".join(sections)

    print(summary)

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"\nSaved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()