#!/usr/bin/env python3
"""
train_baseline.py

Final baseline regression models for OpenROAD ASIC QoR prediction.

This script trains models to predict final OpenROAD QoR targets from:
- Bambu/HLS metrics
- static C-code features
- Yosys synthesis metrics
- OpenROAD clock constraint

Targets:
- openroad_synth_chip_area
- openroad_critical_path_delay
- openroad_total_power

Evaluation:
- random train/test split
- k-fold cross-validation
- leave-one-benchmark-out, when enough benchmarks exist

Outputs:
- reports/modeling/openroad_random_split_results.csv
- reports/modeling/openroad_kfold_results.csv
- reports/modeling/openroad_leave_one_benchmark_out_results.csv
- reports/modeling/openroad_all_model_results.csv
- reports/modeling/openroad_feature_importance_<target>_<model>.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUTPUT_DIR = Path("reports/modeling")

TARGETS = [
    "openroad_synth_chip_area",
    "openroad_critical_path_delay",
    "openroad_total_power",
]

# OpenROAD output columns must not be used as input features.
# Only openroad_clock_constraint is allowed as a feature, because it is known before backend execution.
OPENROAD_LEAKAGE_COLUMNS = [
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

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }


def get_models():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }


def get_feature_columns(df: pd.DataFrame, target: str):
    drop_cols = set(OPENROAD_LEAKAGE_COLUMNS)

    # The selected target is already in OPENROAD_LEAKAGE_COLUMNS, but keep this explicit.
    drop_cols.add(target)

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
    ]

    # Keep only numeric columns. Preprocessing already one-hot encodes categorical columns.
    numeric_feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    return numeric_feature_cols


def make_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def save_feature_importance(pipeline, feature_cols, target, model_name):
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_path = OUTPUT_DIR / f"openroad_feature_importance_{target}_{model_name}.csv"
    importance_df.to_csv(out_path, index=False)


def prepare_target_df(df: pd.DataFrame, target: str):
    target_df = df.dropna(subset=[target]).copy()

    feature_cols = get_feature_columns(target_df, target)

    # Drop rows where all features are missing.
    target_df = target_df.dropna(subset=feature_cols, how="all").copy()

    X = target_df[feature_cols]
    y = target_df[target]

    return target_df, X, y, feature_cols


def random_split_evaluation(df: pd.DataFrame):
    rows = []

    for target in TARGETS:
        print(f"\n=== Target: {target} ===")
        target_df, X, y, feature_cols = prepare_target_df(df, target)

        print(f"Rows available: {len(target_df)}")
        print(f"Feature columns: {len(feature_cols)}")

        if len(target_df) < 4:
            print(f"Skipping random split for {target}: need at least 4 rows.")
            continue

        print("Running random train/test split...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )

        for model_name, model in get_models().items():
            print(f"  Training {model_name}...")

            pipeline = make_pipeline(model)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = regression_metrics(y_test, y_pred)

            rows.append({
                "evaluation": "random_split",
                "target": target,
                "model": model_name,
                "rows": len(target_df),
                "features": len(feature_cols),
                **metrics,
            })

            save_feature_importance(
                pipeline,
                feature_cols,
                target,
                model_name,
            )

    return pd.DataFrame(rows)


def kfold_evaluation(df: pd.DataFrame, max_splits=5):
    rows = []

    for target in TARGETS:
        print(f"\n=== K-fold for target: {target} ===")
        target_df, X, y, feature_cols = prepare_target_df(df, target)

        n_rows = len(target_df)

        if n_rows < 4:
            print(f"Skipping k-fold for {target}: need at least 4 rows.")
            continue

        n_splits = min(max_splits, n_rows)

        # Avoid folds with single samples where possible.
        if n_splits > 3 and n_rows < 10:
            n_splits = 3

        print(f"Running {n_splits}-fold CV...")
        print(f"Rows available: {n_rows}")
        print(f"Feature columns: {len(feature_cols)}")

        kfold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for model_name, model in get_models().items():
            print(f"  Training {model_name}...")

            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
                print(f"    Fold {fold_idx}/{n_splits}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline = make_pipeline(model)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                metrics = regression_metrics(y_test, y_pred)
                metrics["fold"] = fold_idx
                fold_metrics.append(metrics)

            fold_df = pd.DataFrame(fold_metrics)

            rows.append({
                "evaluation": "kfold",
                "target": target,
                "model": model_name,
                "folds": n_splits,
                "rows": n_rows,
                "features": len(feature_cols),
                "mae_mean": fold_df["mae"].mean(),
                "mae_std": fold_df["mae"].std(),
                "rmse_mean": fold_df["rmse"].mean(),
                "rmse_std": fold_df["rmse"].std(),
                "r2_mean": fold_df["r2"].mean(),
                "r2_std": fold_df["r2"].std(),
            })

    return pd.DataFrame(rows)


def leave_one_benchmark_out_evaluation(df: pd.DataFrame):
    rows = []

    benchmark_cols = [c for c in df.columns if c.startswith("benchmark_")]

    if not benchmark_cols:
        print("\nSkipping leave-one-benchmark-out: benchmark one-hot columns not found.")
        return pd.DataFrame(rows)

    df = df.copy()
    df["_benchmark_group"] = df[benchmark_cols].idxmax(axis=1).str.replace("benchmark_", "", regex=False)

    unique_benchmarks = df["_benchmark_group"].nunique()

    if unique_benchmarks < 2:
        print("\nSkipping leave-one-benchmark-out: need at least 2 benchmarks.")
        return pd.DataFrame(rows)

    for target in TARGETS:
        print(f"\n=== Leave-one-benchmark-out for target: {target} ===")
        target_df, X, y, feature_cols = prepare_target_df(df, target)

        groups = target_df["_benchmark_group"]
        unique_benchmarks = groups.nunique()

        if unique_benchmarks < 2:
            print(f"Skipping LOBO for {target}: need at least 2 benchmarks.")
            continue

        print(f"Running LOBO across {unique_benchmarks} benchmarks...")
        print(f"Rows available: {len(target_df)}")
        print(f"Feature columns: {len(feature_cols)}")

        group_kfold = GroupKFold(n_splits=unique_benchmarks)

        for model_name, model in get_models().items():
            print(f"  Training {model_name}...")

            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), start=1):
                heldout_benchmark = groups.iloc[test_idx].iloc[0]
                print(f"    Fold {fold_idx}/{unique_benchmarks}, held out: {heldout_benchmark}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline = make_pipeline(model)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                metrics = regression_metrics(y_test, y_pred)

                fold_metrics.append({
                    "heldout_benchmark": heldout_benchmark,
                    **metrics,
                })

            fold_df = pd.DataFrame(fold_metrics)

            detail_path = (
                OUTPUT_DIR
                / f"openroad_leave_one_benchmark_out_detail_{target}_{model_name}.csv"
            )
            fold_df.to_csv(detail_path, index=False)

            rows.append({
                "evaluation": "leave_one_benchmark_out",
                "target": target,
                "model": model_name,
                "benchmarks": unique_benchmarks,
                "rows": len(target_df),
                "features": len(feature_cols),
                "mae_mean": fold_df["mae"].mean(),
                "mae_std": fold_df["mae"].std(),
                "rmse_mean": fold_df["rmse"].mean(),
                "rmse_std": fold_df["rmse"].std(),
                "r2_mean": fold_df["r2"].mean(),
                "r2_std": fold_df["r2"].std(),
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)

    print("\n=== Loaded OpenROAD modeling dataset ===")
    print(f"Dataset: {DATASET}")
    print(f"Shape: {df.shape}")

    if len(df) < 4:
        print("\nWARNING: Very few OpenROAD-labeled rows are available.")
        print("The script is ready, but meaningful model evaluation requires more OpenROAD runs.")

    print("\nAvailable OpenROAD targets:")
    for target in TARGETS:
        if target in df.columns:
            print(f"- {target}: {df[target].notna().sum()} non-null rows")
        else:
            print(f"- {target}: MISSING")

    random_results = random_split_evaluation(df)
    kfold_results = kfold_evaluation(df)
    lobo_results = leave_one_benchmark_out_evaluation(df)

    random_results.to_csv(OUTPUT_DIR / "openroad_random_split_results.csv", index=False)
    kfold_results.to_csv(OUTPUT_DIR / "openroad_kfold_results.csv", index=False)
    lobo_results.to_csv(OUTPUT_DIR / "openroad_leave_one_benchmark_out_results.csv", index=False)

    all_results = pd.concat(
        [
            random_results,
            kfold_results,
            lobo_results,
        ],
        ignore_index=True,
        sort=False,
    )

    all_results.to_csv(OUTPUT_DIR / "openroad_all_model_results.csv", index=False)

    print("\n=== OpenROAD modeling complete ===")
    print(f"Results written to: {OUTPUT_DIR}")

    if not random_results.empty:
        print("\nRandom split results:")
        print(random_results)

    if not kfold_results.empty:
        print("\nK-fold results:")
        print(kfold_results)

    if not lobo_results.empty:
        print("\nLeave-one-benchmark-out results:")
        print(lobo_results)


if __name__ == "__main__":
    main()
    