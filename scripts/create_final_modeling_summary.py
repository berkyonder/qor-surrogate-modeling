#!/usr/bin/env python3
"""
create_final_modeling_summary.py

Consolidates final OpenROAD QoR modeling results from:
- baseline models
- feature-engineered models
- rigorous feature models
- EDA-aware models
- ablation study

Outputs:
- reports/modeling/final_openroad_modeling_summary.csv
- reports/modeling/final_openroad_modeling_summary.md
"""

from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("reports/modeling")

OUTPUT_CSV = RESULTS_DIR / "final_openroad_modeling_summary.csv"
OUTPUT_MD = RESULTS_DIR / "final_openroad_modeling_summary.md"

TARGET_LABELS = {
    "openroad_synth_chip_area": "Final ASIC area",
    "openroad_critical_path_delay": "Final critical path delay",
    "openroad_total_power": "Final total power",
}

TARGET_ORDER = [
    "openroad_synth_chip_area",
    "openroad_critical_path_delay",
    "openroad_total_power",
]


def load_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def standardize_baseline() -> pd.DataFrame:
    path = RESULTS_DIR / "openroad_all_model_results.csv"
    df = load_if_exists(path)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "experiment": "baseline",
            "evaluation": row.get("evaluation"),
            "target": row.get("target"),
            "model": row.get("model"),
            "feature_set": "all_direct_features",
            "reduction": "none",
            "scaler": "default",
            "benchmark_onehot": None,
            "mae": row.get("mae"),
            "rmse": row.get("rmse"),
            "r2": row.get("r2"),
            "relative_mae": None,
            "rows": row.get("rows"),
            "features": row.get("features"),
        })

    return pd.DataFrame(rows)


def standardize_feature_engineering() -> pd.DataFrame:
    path = RESULTS_DIR / "openroad_feature_engineering_results.csv"
    df = load_if_exists(path)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "experiment": "feature_engineering",
            "evaluation": row.get("evaluation"),
            "target": row.get("target"),
            "model": row.get("model"),
            "feature_set": row.get("reduction"),
            "reduction": row.get("reduction"),
            "scaler": "standard",
            "benchmark_onehot": None,
            "mae": row.get("mae"),
            "rmse": row.get("rmse"),
            "r2": row.get("r2"),
            "relative_mae": None,
            "rows": row.get("rows"),
            "features": row.get("features_before"),
        })

    return pd.DataFrame(rows)


def standardize_rigorous() -> pd.DataFrame:
    path = RESULTS_DIR / "openroad_rigorous_feature_results.csv"
    df = load_if_exists(path)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "experiment": "rigorous_direct_target",
            "evaluation": row.get("evaluation"),
            "target": row.get("target"),
            "model": row.get("model"),
            "feature_set": row.get("feature_mode"),
            "reduction": row.get("reduction"),
            "scaler": "standard_or_log",
            "benchmark_onehot": row.get("include_benchmark_onehot"),
            "mae": row.get("mae"),
            "rmse": row.get("rmse"),
            "r2": row.get("r2"),
            "relative_mae": row.get("relative_mae"),
            "rows": row.get("rows"),
            "features": row.get("features"),
        })

    return pd.DataFrame(rows)


def standardize_eda_aware() -> pd.DataFrame:
    path = RESULTS_DIR / "openroad_eda_aware_results.csv"
    df = load_if_exists(path)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "experiment": "eda_aware_normalized_target",
            "evaluation": row.get("evaluation"),
            "target": row.get("original_target"),
            "normalized_target": row.get("normalized_target"),
            "model": row.get("model"),
            "feature_set": "semantic_ratios_normalized_target",
            "reduction": row.get("reduction"),
            "scaler": row.get("scaler"),
            "benchmark_onehot": row.get("include_benchmark_onehot"),
            "mae": row.get("mae_original"),
            "rmse": row.get("rmse_original"),
            "r2": row.get("r2_original"),
            "relative_mae": row.get("relative_mae_original"),
            "rows": row.get("rows"),
            "features": row.get("features"),
        })

    return pd.DataFrame(rows)


def standardize_ablation() -> pd.DataFrame:
    path = RESULTS_DIR / "openroad_ablation_results.csv"
    df = load_if_exists(path)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "experiment": "ablation_direct_target",
            "evaluation": row.get("evaluation"),
            "target": row.get("target"),
            "model": row.get("model"),
            "feature_set": row.get("feature_set"),
            "reduction": row.get("reduction"),
            "scaler": row.get("scaler"),
            "benchmark_onehot": row.get("include_benchmark_onehot"),
            "mae": row.get("mae"),
            "rmse": row.get("rmse"),
            "r2": row.get("r2"),
            "relative_mae": row.get("relative_mae"),
            "rows": row.get("rows"),
            "features": row.get("features"),
        })

    return pd.DataFrame(rows)


def best_by_relative_mae(df: pd.DataFrame, evaluation: str, target: str) -> pd.DataFrame:
    sub = df[
        (df["evaluation"] == evaluation)
        & (df["target"] == target)
        & (df["relative_mae"].notna())
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    return sub.sort_values("relative_mae", ascending=True).head(1)


def best_by_r2(df: pd.DataFrame, evaluation: str, target: str) -> pd.DataFrame:
    sub = df[
        (df["evaluation"] == evaluation)
        & (df["target"] == target)
        & (df["r2"].notna())
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    return sub.sort_values("r2", ascending=False).head(1)


def write_markdown(all_df: pd.DataFrame):
    lines = []

    lines.append("# Final OpenROAD QoR Modeling Summary\n")

    dataset_path = Path("data/processed/openroad_modeling_dataset.csv")
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        lines.append("## Dataset")
        lines.append(f"- OpenROAD-labeled rows: {dataset.shape[0]}")
        lines.append(f"- Columns: {dataset.shape[1]}")
        if "openroad_clock_constraint" in dataset.columns:
            lines.append("- OpenROAD clock constraints:")
            for value, count in dataset["openroad_clock_constraint"].value_counts().sort_index().items():
                lines.append(f"  - {value}: {count} rows")
        lines.append("")

    lines.append("## Main Direct-Target Results")
    lines.append("")
    lines.append("These results use direct OpenROAD targets, not normalized target ratios.")
    lines.append("For small group folds, MAE and relative MAE are emphasized over R2.\n")

    preferred = {
        "openroad_synth_chip_area": {
            "experiment": "rigorous_direct_target",
            "evaluation": "groupkfold_by_benchmark",
        },
        "openroad_critical_path_delay": {
            "experiment": "rigorous_direct_target",
            "evaluation": "groupkfold_by_benchmark",
        },
        "openroad_total_power": {
            "experiment": "ablation_direct_target",
            "evaluation": "groupkfold_by_benchmark",
        },
    }

    lines.append("| Target | Preferred experiment | Evaluation | MAE | RMSE | Relative MAE | R2 | Model | Feature set |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

    for target in TARGET_ORDER:
        cfg = preferred[target]
        sub = all_df[
            (all_df["target"] == target)
            & (all_df["experiment"] == cfg["experiment"])
            & (all_df["evaluation"] == cfg["evaluation"])
            & (all_df["relative_mae"].notna())
        ].copy()

        if sub.empty:
            continue

        best = sub.sort_values("relative_mae", ascending=True).iloc[0]

        lines.append(
            f"| {TARGET_LABELS[target]} | "
            f"{best['experiment']} | "
            f"{best['evaluation']} | "
            f"{best['mae']:.4f} | "
            f"{best['rmse']:.4f} | "
            f"{best['relative_mae']:.4f} | "
            f"{best['r2']:.4f} | "
            f"{best['model']} | "
            f"{best['feature_set']} |"
        )

    lines.append("")
    lines.append("## Best Group-Aware Results Across All Experiments")
    lines.append("")
    lines.append("| Target | Experiment | MAE | RMSE | Relative MAE | R2 | Model | Feature set | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")

    for target in TARGET_ORDER:
        best = best_by_relative_mae(all_df, "groupkfold_by_benchmark", target)
        if best.empty:
            continue
        row = best.iloc[0]

        note = "direct target"
        if row["experiment"] == "eda_aware_normalized_target":
            note = "normalized target exploratory"

        lines.append(
            f"| {TARGET_LABELS[target]} | "
            f"{row['experiment']} | "
            f"{row['mae']:.4f} | "
            f"{row['rmse']:.4f} | "
            f"{row['relative_mae']:.4f} | "
            f"{row['r2']:.4f} | "
            f"{row['model']} | "
            f"{row['feature_set']} | "
            f"{note} |"
        )

    lines.append("")
    lines.append("## Best Random-Split Results Across All Experiments")
    lines.append("")
    lines.append("Random split is useful as an optimistic estimate, but it is less rigorous than group-aware validation.\n")
    lines.append("| Target | Experiment | MAE | RMSE | Relative MAE | R2 | Model |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for target in TARGET_ORDER:
        best = best_by_r2(all_df, "random_split", target)
        if best.empty:
            continue
        row = best.iloc[0]

        rel = row["relative_mae"]
        rel_text = f"{rel:.4f}" if pd.notna(rel) else "N/A"

        lines.append(
            f"| {TARGET_LABELS[target]} | "
            f"{row['experiment']} | "
            f"{row['mae']:.4f} | "
            f"{row['rmse']:.4f} | "
            f"{rel_text} | "
            f"{row['r2']:.4f} | "
            f"{row['model']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Area is the most predictable final OpenROAD QoR target.")
    lines.append("- Timing prediction is moderately successful, especially with structural HLS/Yosys features and regularized models.")
    lines.append("- Power remains exploratory because it is sensitive to clock constraints and the dataset contains limited clock-condition diversity.")
    lines.append("- Group-aware validation is treated as the most realistic test because it evaluates generalization to unseen benchmarks.")
    lines.append("- R2 is unstable for leave-one-benchmark/group folds with very small test sets, so MAE and relative MAE are emphasized.")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dfs = [
        standardize_baseline(),
        standardize_feature_engineering(),
        standardize_rigorous(),
        standardize_eda_aware(),
        standardize_ablation(),
    ]

    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        raise RuntimeError("No modeling result files found.")

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(OUTPUT_CSV, index=False)

    write_markdown(all_df)

    print("\n=== Final OpenROAD modeling summary created ===")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Markdown: {OUTPUT_MD}")

    print("\nMain group-aware direct-target results:")
    md = OUTPUT_MD.read_text(encoding="utf-8")
    print(md[:3000])


if __name__ == "__main__":
    main()