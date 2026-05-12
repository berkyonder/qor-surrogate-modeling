#!/usr/bin/env python3
"""
visualize_dataset.py

Creates visualization figures for the Bambu early-HLS QoR dataset.

Inputs:
- data/extracted_metrics/early_hls_metrics_raw.csv
- data/processed/early_hls_metrics_clean.csv

Outputs:
- reports/figures/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


RAW_DATASET = Path("data/extracted_metrics/early_hls_metrics_raw.csv")
CLEAN_DATASET = Path("data/processed/early_hls_metrics_clean.csv")
FIGURE_DIR = Path("reports/figures")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    for col in df.columns:
        if col in [
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
            "dsps",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_histogram(df: pd.DataFrame, column: str, filename: str, title: str) -> None:
    if column not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def save_boxplot_by_category(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    filename: str,
    title: str,
) -> None:
    if value_col not in df.columns or category_col not in df.columns:
        return

    categories = sorted(df[category_col].dropna().unique())
    data = [df[df[category_col] == cat][value_col].dropna() for cat in categories]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=categories)
    plt.title(title)
    plt.xlabel(category_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    filename: str,
    title: str,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, filename: str, title: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(11, 9))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image)
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def save_group_mean_barplot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    filename: str,
    title: str,
) -> None:
    if group_col not in df.columns or value_col not in df.columns:
        return

    grouped = df.groupby(group_col)[value_col].mean().sort_values()

    plt.figure(figsize=(10, 5))
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(f"Mean {value_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def generate_figures(df: pd.DataFrame, prefix: str) -> None:
    save_correlation_heatmap(
        df,
        f"{prefix}_correlation_heatmap.png",
        f"{prefix}: Correlation Heatmap",
    )

    save_histogram(
        df,
        "total_area",
        f"{prefix}_total_area_distribution.png",
        f"{prefix}: Distribution of total_area",
    )

    save_histogram(
        df,
        "control_steps",
        f"{prefix}_control_steps_distribution.png",
        f"{prefix}: Distribution of control_steps",
    )

    save_histogram(
        df,
        "frequency_mhz",
        f"{prefix}_frequency_mhz_distribution.png",
        f"{prefix}: Distribution of frequency_mhz",
    )

    save_boxplot_by_category(
        df,
        "total_area",
        "opt_level",
        f"{prefix}_total_area_by_opt_level.png",
        f"{prefix}: total_area by opt_level",
    )

    save_boxplot_by_category(
        df,
        "control_steps",
        "opt_level",
        f"{prefix}_control_steps_by_opt_level.png",
        f"{prefix}: control_steps by opt_level",
    )

    save_boxplot_by_category(
        df,
        "total_area",
        "mem_policy",
        f"{prefix}_total_area_by_mem_policy.png",
        f"{prefix}: total_area by mem_policy",
    )

    save_boxplot_by_category(
        df,
        "control_steps",
        "mem_policy",
        f"{prefix}_control_steps_by_mem_policy.png",
        f"{prefix}: control_steps by mem_policy",
    )

    save_boxplot_by_category(
        df,
        "total_area",
        "dataset_size",
        f"{prefix}_total_area_by_dataset_size.png",
        f"{prefix}: total_area by dataset_size",
    )

    save_boxplot_by_category(
        df,
        "control_steps",
        "dataset_size",
        f"{prefix}_control_steps_by_dataset_size.png",
        f"{prefix}: control_steps by dataset_size",
    )

    save_scatter(
        df,
        "area_est",
        "total_area",
        f"{prefix}_area_est_vs_total_area.png",
        f"{prefix}: area_est vs total_area",
    )

    save_scatter(
        df,
        "states",
        "control_steps",
        f"{prefix}_states_vs_control_steps.png",
        f"{prefix}: states vs control_steps",
    )

    save_scatter(
        df,
        "modules_instantiated",
        "total_area",
        f"{prefix}_modules_vs_total_area.png",
        f"{prefix}: modules_instantiated vs total_area",
    )

    save_group_mean_barplot(
        df,
        "benchmark",
        "total_area",
        f"{prefix}_mean_total_area_by_benchmark.png",
        f"{prefix}: Mean total_area by benchmark",
    )

    save_group_mean_barplot(
        df,
        "benchmark",
        "control_steps",
        f"{prefix}_mean_control_steps_by_benchmark.png",
        f"{prefix}: Mean control_steps by benchmark",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(RAW_DATASET)
    clean_df = load_dataset(CLEAN_DATASET)

    generate_figures(raw_df, "raw")
    generate_figures(clean_df, "clean")

    print("\n=== Visualization complete ===")
    print(f"Figures written to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
