#!/usr/bin/env python3
"""
visualize_modeling_results.py

Create plots for OpenROAD final ASIC QoR modeling results.

Inputs:
- reports/modeling/openroad_random_split_results.csv
- reports/modeling/openroad_kfold_results.csv
- reports/modeling/openroad_leave_one_benchmark_out_results.csv
- reports/modeling/openroad_feature_importance_*.csv

Outputs:
- reports/modeling/plots/*.png
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("reports/modeling")
PLOTS_DIR = RESULTS_DIR / "plots"

TARGET_NAMES = {
    "openroad_synth_chip_area": "Final ASIC area",
    "openroad_critical_path_delay": "Critical path delay",
    "openroad_total_power": "Total power",
}


def target_label(target: str) -> str:
    return TARGET_NAMES.get(target, target)


def safe_load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def save_bar_plot(df: pd.DataFrame, value_col: str, title: str, output_path: Path):
    if df.empty or value_col not in df.columns:
        print(f"Skipping plot, missing data/value column: {output_path}")
        return

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[value_col])

    if plot_df.empty:
        print(f"Skipping plot, no non-null values: {output_path}")
        return

    plot_df["target_label"] = plot_df["target"].map(target_label)
    plot_df["label"] = plot_df["target_label"] + "\n" + plot_df["model"]

    plot_df = plot_df.sort_values(["target_label", "model"])

    plt.figure(figsize=(max(8, len(plot_df) * 0.75), 5))
    plt.bar(plot_df["label"], plot_df[value_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_random_split_results():
    df = safe_load(RESULTS_DIR / "openroad_random_split_results.csv")

    if df.empty:
        return

    save_bar_plot(
        df,
        "mae",
        "OpenROAD Random Split MAE",
        PLOTS_DIR / "openroad_random_split_mae.png",
    )

    save_bar_plot(
        df,
        "rmse",
        "OpenROAD Random Split RMSE",
        PLOTS_DIR / "openroad_random_split_rmse.png",
    )

    save_bar_plot(
        df,
        "r2",
        "OpenROAD Random Split R²",
        PLOTS_DIR / "openroad_random_split_r2.png",
    )


def plot_kfold_results():
    df = safe_load(RESULTS_DIR / "openroad_kfold_results.csv")

    if df.empty:
        return

    save_bar_plot(
        df,
        "mae_mean",
        "OpenROAD K-Fold MAE Mean",
        PLOTS_DIR / "openroad_kfold_mae_mean.png",
    )

    save_bar_plot(
        df,
        "rmse_mean",
        "OpenROAD K-Fold RMSE Mean",
        PLOTS_DIR / "openroad_kfold_rmse_mean.png",
    )

    save_bar_plot(
        df,
        "r2_mean",
        "OpenROAD K-Fold R² Mean",
        PLOTS_DIR / "openroad_kfold_r2_mean.png",
    )


def plot_lobo_results():
    df = safe_load(RESULTS_DIR / "openroad_leave_one_benchmark_out_results.csv")

    if df.empty:
        return

    save_bar_plot(
        df,
        "mae_mean",
        "OpenROAD Leave-One-Benchmark-Out MAE Mean",
        PLOTS_DIR / "openroad_lobo_mae_mean.png",
    )

    save_bar_plot(
        df,
        "rmse_mean",
        "OpenROAD Leave-One-Benchmark-Out RMSE Mean",
        PLOTS_DIR / "openroad_lobo_rmse_mean.png",
    )

    save_bar_plot(
        df,
        "r2_mean",
        "OpenROAD Leave-One-Benchmark-Out R² Mean",
        PLOTS_DIR / "openroad_lobo_r2_mean.png",
    )


def clean_target_from_importance_filename(path: Path) -> tuple[str, str]:
    """
    File pattern:
    openroad_feature_importance_<target>_<model>.csv

    Targets contain underscores, so parse by known model names.
    """
    stem = path.stem
    prefix = "openroad_feature_importance_"

    if not stem.startswith(prefix):
        return stem, "unknown"

    rest = stem[len(prefix):]

    known_models = [
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    ]

    for model in known_models:
        suffix = "_" + model
        if rest.endswith(suffix):
            target = rest[:-len(suffix)]
            return target, model

    return rest, "unknown"


def plot_feature_importance(top_n: int = 15):
    importance_files = sorted(RESULTS_DIR.glob("openroad_feature_importance_*.csv"))

    if not importance_files:
        print("No OpenROAD feature importance files found.")
        return

    for path in importance_files:
        df = pd.read_csv(path)

        if df.empty or "feature" not in df.columns or "importance" not in df.columns:
            continue

        target, model = clean_target_from_importance_filename(path)

        top_df = df.sort_values("importance", ascending=False).head(top_n)
        top_df = top_df.sort_values("importance", ascending=True)

        plt.figure(figsize=(8, max(4, len(top_df) * 0.35)))
        plt.barh(top_df["feature"], top_df["importance"])
        plt.xlabel("Importance")
        plt.title(f"Feature importance: {target_label(target)} ({model})")
        plt.tight_layout()

        safe_target = re.sub(r"[^A-Za-z0-9_]+", "_", target)
        safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model)
        out_path = PLOTS_DIR / f"feature_importance_{safe_target}_{safe_model}.png"

        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


def plot_dataset_overview():
    dataset_path = Path("data/processed/openroad_modeling_dataset.csv")

    if not dataset_path.exists():
        print(f"Skipping dataset overview, missing: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    if "openroad_clock_constraint" in df.columns:
        counts = df["openroad_clock_constraint"].value_counts().sort_index()

        plt.figure(figsize=(6, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("OpenROAD clock constraint (ns)")
        plt.ylabel("Number of rows")
        plt.title("OpenROAD-labeled rows by clock constraint")
        plt.tight_layout()

        out_path = PLOTS_DIR / "openroad_rows_by_clock_constraint.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")

    target_cols = [
        "openroad_synth_chip_area",
        "openroad_critical_path_delay",
        "openroad_total_power",
    ]

    for col in target_cols:
        if col not in df.columns:
            continue

        values = df[col].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=min(10, max(3, len(values))))
        plt.xlabel(target_label(col))
        plt.ylabel("Count")
        plt.title(f"Distribution: {target_label(col)}")
        plt.tight_layout()

        out_path = PLOTS_DIR / f"distribution_{col}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Creating OpenROAD modeling plots ===")

    plot_dataset_overview()
    plot_random_split_results()
    plot_kfold_results()
    plot_lobo_results()
    plot_feature_importance(top_n=15)

    print("\n=== Visualization complete ===")
    print(f"Plots written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
