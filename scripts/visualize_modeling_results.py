#!/usr/bin/env python3
"""
visualize_modeling_results.py

Creates plots for baseline model evaluation and feature importance.

Inputs:
- reports/modeling/random_split_results.csv
- reports/modeling/kfold_results.csv
- reports/modeling/leave_one_benchmark_out_results.csv
- reports/modeling/feature_importance_*.csv

Outputs:
- reports/figures/modeling_*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELING_DIR = Path("reports/modeling")
FIGURE_DIR = Path("reports/figures")


def save_barplot(df, x_col, y_col, group_col, title, filename):
    """
    Simple grouped bar plot using pandas pivot.
    """
    if df.empty:
        return

    pivot = df.pivot_table(
        values=y_col,
        index=x_col,
        columns=group_col,
        aggfunc="first",
    )

    plt.figure(figsize=(10, 5))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def plot_random_split_results():
    path = MODELING_DIR / "random_split_results.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    for target in df["target"].unique():
        for feature_mode in df["feature_mode"].unique():
            subset = df[
                (df["target"] == target)
                & (df["feature_mode"] == feature_mode)
            ]

            save_barplot(
                subset,
                x_col="model",
                y_col="r2",
                group_col="feature_mode",
                title=f"Random Split R² - {target} ({feature_mode})",
                filename=f"modeling_random_split_r2_{target}_{feature_mode}.png",
            )

            save_barplot(
                subset,
                x_col="model",
                y_col="mae",
                group_col="feature_mode",
                title=f"Random Split MAE - {target} ({feature_mode})",
                filename=f"modeling_random_split_mae_{target}_{feature_mode}.png",
            )


def plot_kfold_results():
    path = MODELING_DIR / "kfold_results.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    for target in df["target"].unique():
        for feature_mode in df["feature_mode"].unique():
            subset = df[
                (df["target"] == target)
                & (df["feature_mode"] == feature_mode)
            ]

            plt.figure(figsize=(9, 5))
            plt.bar(subset["model"], subset["r2_mean"], yerr=subset["r2_std"], capsize=4)
            plt.title(f"5-Fold CV R² - {target} ({feature_mode})")
            plt.xlabel("model")
            plt.ylabel("R² mean")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"modeling_kfold_r2_{target}_{feature_mode}.png", dpi=300)
            plt.close()

            plt.figure(figsize=(9, 5))
            plt.bar(subset["model"], subset["mae_mean"], yerr=subset["mae_std"], capsize=4)
            plt.title(f"5-Fold CV MAE - {target} ({feature_mode})")
            plt.xlabel("model")
            plt.ylabel("MAE mean")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"modeling_kfold_mae_{target}_{feature_mode}.png", dpi=300)
            plt.close()


def plot_lobo_results():
    path = MODELING_DIR / "leave_one_benchmark_out_results.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    for target in df["target"].unique():
        for feature_mode in df["feature_mode"].unique():
            subset = df[
                (df["target"] == target)
                & (df["feature_mode"] == feature_mode)
            ]

            plt.figure(figsize=(9, 5))
            plt.bar(subset["model"], subset["mae_mean"], yerr=subset["mae_std"], capsize=4)
            plt.title(f"Leave-One-Benchmark-Out MAE - {target} ({feature_mode})")
            plt.xlabel("model")
            plt.ylabel("MAE mean")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"modeling_lobo_mae_{target}_{feature_mode}.png", dpi=300)
            plt.close()


def plot_feature_importance(path: Path, output_name: str, top_n: int = 15):
    if not path.exists():
        print(f"Feature importance file not found: {path}")
        return

    df = pd.read_csv(path)

    if df.empty or "feature" not in df.columns or "importance" not in df.columns:
        return

    df = df.sort_values("importance", ascending=False).head(top_n)
    df = df.iloc[::-1]  # horizontal barplot from low to high

    plt.figure(figsize=(9, 6))
    plt.barh(df["feature"], df["importance"])
    plt.title(output_name.replace("_", " ").replace(".png", ""))
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=300)
    plt.close()


def plot_selected_feature_importances():
    selected_files = [
        (
            MODELING_DIR / "feature_importance_total_area_no_leakage_random_forest.csv",
            "feature_importance_total_area_no_leakage_random_forest.png",
        ),
        (
            MODELING_DIR / "feature_importance_total_area_no_leakage_gradient_boosting.csv",
            "feature_importance_total_area_no_leakage_gradient_boosting.png",
        ),
        (
            MODELING_DIR / "feature_importance_control_steps_no_leakage_random_forest.csv",
            "feature_importance_control_steps_no_leakage_random_forest.png",
        ),
        (
            MODELING_DIR / "feature_importance_control_steps_no_leakage_gradient_boosting.csv",
            "feature_importance_control_steps_no_leakage_gradient_boosting.png",
        ),
    ]

    for path, output_name in selected_files:
        plot_feature_importance(path, output_name)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_random_split_results()
    plot_kfold_results()
    plot_lobo_results()
    plot_selected_feature_importances()

    print("\n=== Modeling visualization complete ===")
    print(f"Figures written to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()