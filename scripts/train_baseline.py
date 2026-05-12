#!/usr/bin/env python3
"""
train_baseline.py

Baseline regression models for QoR surrogate modeling.

This script trains models to predict:
- total_area
- control_steps

Evaluation modes:
- random train/test split
- 5-fold cross-validation
- leave-one-benchmark-out evaluation

Feature modes:
- full: use all available features except target
- no_leakage: remove near-direct proxy features for each target

Outputs:
- reports/modeling/random_split_results.csv
- reports/modeling/kfold_results.csv
- reports/modeling/leave_one_benchmark_out_results.csv
- reports/modeling/feature_importance_<target>_<feature_mode>_<model>.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET = Path("data/processed/early_hls_metrics_clean.csv")
OUTPUT_DIR = Path("reports/modeling")

TARGETS = ["total_area", "control_steps"]

CATEGORICAL_COLUMNS = [
    "benchmark",
    "dataset_size",
    "mem_policy",
    "opt_level",
]

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
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


def get_feature_columns(df, target, feature_mode):
    drop_cols = [target]

    # Avoid using the other direct target as a feature only if it is not the current target?
    # We keep it for now because current project models early-HLS metrics jointly.
    # Specific leakage handling is below.

    if feature_mode == "no_leakage":
        if target == "total_area":
            drop_cols += ["area_est", "mux_area"]
        elif target == "control_steps":
            drop_cols += ["states"]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def make_pipeline(model, feature_cols, df):
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_encoded_feature_names(pipeline, feature_cols):
    preprocessor = pipeline.named_steps["preprocessor"]

    categorical = [
        c for c in CATEGORICAL_COLUMNS
        if c in feature_cols
    ]
    numeric = [
        c for c in feature_cols
        if c not in categorical
    ]

    feature_names = []

    if categorical:
        encoder = preprocessor.named_transformers_["categorical"]
        encoded_names = encoder.get_feature_names_out(categorical).tolist()
        feature_names.extend(encoded_names)

    feature_names.extend(numeric)

    return feature_names


def save_feature_importance(pipeline, feature_cols, target, feature_mode, model_name):
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = get_encoded_feature_names(pipeline, feature_cols)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_path = OUTPUT_DIR / f"feature_importance_{target}_{feature_mode}_{model_name}.csv"
    importance_df.to_csv(out_path, index=False)


def random_split_evaluation(df):
    rows = []

    for target in TARGETS:
        for feature_mode in ["full", "no_leakage"]:
            feature_cols = get_feature_columns(df, target, feature_mode)

            X = df[feature_cols]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )

            for model_name, model in get_models().items():
                pipeline = make_pipeline(model, feature_cols, df)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                metrics = regression_metrics(y_test, y_pred)

                rows.append({
                    "evaluation": "random_split",
                    "target": target,
                    "feature_mode": feature_mode,
                    "model": model_name,
                    **metrics,
                })

                save_feature_importance(
                    pipeline,
                    feature_cols,
                    target,
                    feature_mode,
                    model_name,
                )

    return pd.DataFrame(rows)


def kfold_evaluation(df, n_splits=5):
    rows = []

    for target in TARGETS:
        for feature_mode in ["full", "no_leakage"]:
            feature_cols = get_feature_columns(df, target, feature_mode)

            X = df[feature_cols]
            y = df[target]

            kfold = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            for model_name, model in get_models().items():
                fold_metrics = []

                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    pipeline = make_pipeline(model, feature_cols, df)
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    metrics = regression_metrics(y_test, y_pred)
                    metrics["fold"] = fold_idx
                    fold_metrics.append(metrics)

                fold_df = pd.DataFrame(fold_metrics)

                rows.append({
                    "evaluation": "kfold",
                    "target": target,
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "folds": n_splits,
                    "mae_mean": fold_df["mae"].mean(),
                    "mae_std": fold_df["mae"].std(),
                    "rmse_mean": fold_df["rmse"].mean(),
                    "rmse_std": fold_df["rmse"].std(),
                    "r2_mean": fold_df["r2"].mean(),
                    "r2_std": fold_df["r2"].std(),
                })

    return pd.DataFrame(rows)


def leave_one_benchmark_out_evaluation(df):
    rows = []

    groups = df["benchmark"]

    for target in TARGETS:
        for feature_mode in ["full", "no_leakage"]:
            feature_cols = get_feature_columns(df, target, feature_mode)

            X = df[feature_cols]
            y = df[target]

            group_kfold = GroupKFold(n_splits=df["benchmark"].nunique())

            for model_name, model in get_models().items():
                fold_metrics = []

                for train_idx, test_idx in group_kfold.split(X, y, groups):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    heldout_benchmark = X_test["benchmark"].iloc[0]

                    pipeline = make_pipeline(model, feature_cols, df)
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    metrics = regression_metrics(y_test, y_pred)

                    fold_metrics.append({
                        "heldout_benchmark": heldout_benchmark,
                        **metrics,
                    })

                fold_df = pd.DataFrame(fold_metrics)

                # Save detailed per-benchmark results.
                detail_path = (
                    OUTPUT_DIR
                    / f"leave_one_benchmark_out_detail_{target}_{feature_mode}_{model_name}.csv"
                )
                fold_df.to_csv(detail_path, index=False)

                rows.append({
                    "evaluation": "leave_one_benchmark_out",
                    "target": target,
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "benchmarks": df["benchmark"].nunique(),
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

    print("\n=== Loaded dataset ===")
    print(df.shape)

    random_results = random_split_evaluation(df)
    kfold_results = kfold_evaluation(df)
    lobo_results = leave_one_benchmark_out_evaluation(df)

    random_results.to_csv(OUTPUT_DIR / "random_split_results.csv", index=False)
    kfold_results.to_csv(OUTPUT_DIR / "kfold_results.csv", index=False)
    lobo_results.to_csv(OUTPUT_DIR / "leave_one_benchmark_out_results.csv", index=False)

    all_results = pd.concat(
        [
            random_results,
            kfold_results,
            lobo_results,
        ],
        ignore_index=True,
        sort=False,
    )

    all_results.to_csv(OUTPUT_DIR / "all_model_results.csv", index=False)

    print("\n=== Modeling complete ===")
    print(f"Results written to: {OUTPUT_DIR}")
    print("\nRandom split results:")
    print(random_results)
    print("\nK-fold results:")
    print(kfold_results)
    print("\nLeave-one-benchmark-out results:")
    print(lobo_results)


if __name__ == "__main__":
    main()
