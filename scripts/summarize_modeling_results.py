#!/usr/bin/env python3
"""
summarize_modeling_results.py

Display formatted summary tables of baseline model results.
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("reports/modeling")


def display_random_split():
    """Display random split results in a formatted table."""
    df = pd.read_csv(RESULTS_DIR / "random_split_results.csv")
    
    print("\n" + "="*100)
    print("RANDOM SPLIT RESULTS (80/20 train/test)")
    print("="*100)
    
    # Pivot for better readability
    pivot_cols = ["target", "feature_mode", "model"]
    metric_cols = ["mae", "rmse", "r2"]
    
    for metric in metric_cols:
        print(f"\n### {metric.upper()} ###")
        pivot = df.pivot_table(
            values=metric,
            index=["target", "feature_mode"],
            columns="model",
            aggfunc="first"
        )
        print(pivot.to_string())


def display_kfold():
    """Display 5-fold cross-validation results."""
    df = pd.read_csv(RESULTS_DIR / "kfold_results.csv")
    
    print("\n" + "="*100)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*100)
    
    for metric in ["mae", "rmse", "r2"]:
        print(f"\n### {metric.upper()} (mean ± std) ###")
        
        results = []
        for _, row in df.iterrows():
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            results.append({
                "target": row["target"],
                "feature_mode": row["feature_mode"],
                "model": row["model"],
                f"{metric}": f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
            })
        
        results_df = pd.DataFrame(results)
        pivot = results_df.pivot_table(
            values=metric,
            index=["target", "feature_mode"],
            columns="model",
            aggfunc="first"
        )
        print(pivot.to_string())


def display_lobo():
    """Display leave-one-benchmark-out results."""
    df = pd.read_csv(RESULTS_DIR / "leave_one_benchmark_out_results.csv")
    
    print("\n" + "="*100)
    print("LEAVE-ONE-BENCHMARK-OUT RESULTS")
    print("="*100)
    
    for metric in ["mae", "rmse", "r2"]:
        print(f"\n### {metric.upper()} (mean ± std) ###")
        
        results = []
        for _, row in df.iterrows():
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            results.append({
                "target": row["target"],
                "feature_mode": row["feature_mode"],
                "model": row["model"],
                f"{metric}": f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
            })
        
        results_df = pd.DataFrame(results)
        pivot = results_df.pivot_table(
            values=metric,
            index=["target", "feature_mode"],
            columns="model",
            aggfunc="first"
        )
        print(pivot.to_string())


def display_best_models():
    """Display best performing models by target and evaluation method."""
    print("\n" + "=" * 100)
    print("BEST MODELS BY TARGET & EVALUATION METHOD")
    print("=" * 100)

    all_results = pd.read_csv(RESULTS_DIR / "all_model_results.csv")

    for eval_method in all_results["evaluation"].unique():
        print(f"\n--- {eval_method.upper()} ---")
        subset = all_results[all_results["evaluation"] == eval_method]

        for target in subset["target"].unique():
            target_subset = subset[subset["target"] == target].copy()

            # Random split uses r2. K-fold and LOBO use r2_mean.
            if eval_method == "random_split":
                score_col = "r2"
                mae_col = "mae"
                rmse_col = "rmse"
                std_suffix = False
            else:
                score_col = "r2_mean"
                mae_col = "mae_mean"
                rmse_col = "rmse_mean"
                std_suffix = True

            # Drop rows where score is missing.
            target_subset = target_subset.dropna(subset=[score_col])

            if target_subset.empty:
                print(f"\n  Target: {target}")
                print("    No valid rows found.")
                continue

            best_idx = target_subset[score_col].idxmax()
            best_row = target_subset.loc[best_idx]

            print(f"\n  Target: {target}")
            print(f"    Feature Mode: {best_row['feature_mode']}")
            print(f"    Model: {best_row['model']}")

            if std_suffix:
                print(f"    R²: {best_row['r2_mean']:.6f} ± {best_row['r2_std']:.6f}")
                print(f"    MAE: {best_row['mae_mean']:.4f} ± {best_row['mae_std']:.4f}")
                print(f"    RMSE: {best_row['rmse_mean']:.4f} ± {best_row['rmse_std']:.4f}")
            else:
                print(f"    R²: {best_row['r2']:.6f}")
                print(f"    MAE: {best_row['mae']:.4f}")
                print(f"    RMSE: {best_row['rmse']:.4f}")


def main():
    display_random_split()
    display_kfold()
    display_lobo()
    display_best_models()
    
    print("\n" + "="*100)
    print("SUMMARY COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
