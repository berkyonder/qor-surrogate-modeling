#!/usr/bin/env python3
"""
train_feature_engineered_models.py

Feature-reduced OpenROAD QoR modeling experiments.

This script compares:
- all features
- SelectKBest feature selection
- PCA dimensionality reduction
- Ridge / Random Forest / Gradient Boosting / Gaussian Process

Dataset:
- data/processed/openroad_modeling_dataset.csv

Targets:
- openroad_synth_chip_area
- openroad_critical_path_delay
- openroad_total_power

Output:
- reports/modeling/openroad_feature_engineering_results.csv
- reports/modeling/openroad_selected_features_<target>_<experiment>.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUTPUT_DIR = Path("reports/modeling")
OUTPUT_CSV = OUTPUT_DIR / "openroad_feature_engineering_results.csv"

RANDOM_STATE = 42

TARGETS = [
    "openroad_synth_chip_area",
    "openroad_critical_path_delay",
    "openroad_total_power",
]

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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }


def get_feature_columns(df: pd.DataFrame, target: str):
    drop_cols = set(OPENROAD_LEAKAGE_COLUMNS)
    drop_cols.add(target)

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    return feature_cols


def get_models():
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=1.0, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
    )

    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_depth=2,
        ),
        "gaussian_process": GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(reduction_name: str, model):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if reduction_name.startswith("select_k_"):
        k = int(reduction_name.replace("select_k_", ""))
        steps.append(("feature_reduction", SelectKBest(score_func=f_regression, k=k)))

    elif reduction_name.startswith("pca_"):
        n_components = int(reduction_name.replace("pca_", ""))
        steps.append(("feature_reduction", PCA(n_components=n_components, random_state=RANDOM_STATE)))

    elif reduction_name == "all_features":
        pass

    else:
        raise ValueError(f"Unknown reduction strategy: {reduction_name}")

    steps.append(("model", model))

    return Pipeline(steps)


def save_selected_features(pipeline, feature_cols, target, reduction_name):
    if "feature_reduction" not in pipeline.named_steps:
        return

    reducer = pipeline.named_steps["feature_reduction"]

    if not isinstance(reducer, SelectKBest):
        return

    mask = reducer.get_support()
    scores = reducer.scores_

    selected = pd.DataFrame({
        "feature": feature_cols,
        "score": scores,
        "selected": mask,
    }).sort_values("score", ascending=False)

    out_path = OUTPUT_DIR / f"openroad_selected_features_{target}_{reduction_name}.csv"
    selected.to_csv(out_path, index=False)


def evaluate_random_split(df, target, feature_cols, reduction_name, model_name, model):
    target_df = df.dropna(subset=[target]).copy()
    X = target_df[feature_cols]
    y = target_df[target]

    if len(target_df) < 6:
        return None

    # Make k not exceed available feature count/sample count.
    adjusted_reduction = reduction_name
    if reduction_name.startswith("select_k_"):
        k = int(reduction_name.replace("select_k_", ""))
        k = min(k, len(feature_cols))
        adjusted_reduction = f"select_k_{k}"

    if reduction_name.startswith("pca_"):
        n = int(reduction_name.replace("pca_", ""))
        # Train size is 75%; keep PCA components safely below expected train rows.
        max_components = max(1, min(n, len(feature_cols), int(len(target_df) * 0.75) - 1))
        adjusted_reduction = f"pca_{max_components}"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    pipe = make_pipeline(adjusted_reduction, model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    row = {
        "evaluation": "random_split",
        "target": target,
        "reduction": adjusted_reduction,
        "model": model_name,
        "rows": len(target_df),
        "features_before": len(feature_cols),
        "test_rows": len(y_test),
        **metrics(y_test, pred),
    }

    save_selected_features(pipe, feature_cols, target, adjusted_reduction)

    return row


def evaluate_kfold(df, target, feature_cols, reduction_name, model_name, model):
    target_df = df.dropna(subset=[target]).copy()
    X = target_df[feature_cols]
    y = target_df[target]

    n_rows = len(target_df)
    if n_rows < 6:
        return None

    n_splits = min(5, n_rows)
    if n_rows < 10:
        n_splits = 3

    fold_rows = []

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        adjusted_reduction = reduction_name

        if reduction_name.startswith("select_k_"):
            k = int(reduction_name.replace("select_k_", ""))
            k = min(k, len(feature_cols))
            adjusted_reduction = f"select_k_{k}"

        if reduction_name.startswith("pca_"):
            n = int(reduction_name.replace("pca_", ""))
            max_components = max(1, min(n, len(feature_cols), len(train_idx) - 1))
            adjusted_reduction = f"pca_{max_components}"

        pipe = make_pipeline(adjusted_reduction, model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        fold_metric = metrics(y_test, pred)
        fold_metric["fold"] = fold_idx
        fold_rows.append(fold_metric)

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "kfold",
        "target": target,
        "reduction": reduction_name,
        "model": model_name,
        "rows": n_rows,
        "features_before": len(feature_cols),
        "folds": n_splits,
        "mae": fold_df["mae"].mean(),
        "mae_std": fold_df["mae"].std(),
        "rmse": fold_df["rmse"].mean(),
        "rmse_std": fold_df["rmse"].std(),
        "r2": fold_df["r2"].mean(),
        "r2_std": fold_df["r2"].std(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)

    print("\n=== Feature-engineered OpenROAD modeling ===")
    print(f"Dataset: {DATASET}")
    print(f"Shape: {df.shape}")

    reductions = [
        "all_features",
        "select_k_5",
        "select_k_10",
        "select_k_15",
        "pca_3",
        "pca_5",
        "pca_10",
    ]

    models = get_models()
    rows = []

    for target in TARGETS:
        print(f"\n==============================")
        print(f"Target: {target}")
        print(f"==============================")

        feature_cols = get_feature_columns(df, target)
        target_rows = df[target].notna().sum()

        print(f"Rows: {target_rows}")
        print(f"Features before reduction: {len(feature_cols)}")

        for reduction in reductions:
            for model_name, model in models.items():
                print(f"Running: {reduction} + {model_name}")

                try:
                    random_row = evaluate_random_split(
                        df=df,
                        target=target,
                        feature_cols=feature_cols,
                        reduction_name=reduction,
                        model_name=model_name,
                        model=model,
                    )

                    if random_row is not None:
                        rows.append(random_row)

                    kfold_row = evaluate_kfold(
                        df=df,
                        target=target,
                        feature_cols=feature_cols,
                        reduction_name=reduction,
                        model_name=model_name,
                        model=model,
                    )

                    if kfold_row is not None:
                        rows.append(kfold_row)

                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    rows.append({
                        "evaluation": "error",
                        "target": target,
                        "reduction": reduction,
                        "model": model_name,
                        "error": str(exc),
                    })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Feature-engineered modeling complete ===")
    print(f"Output: {OUTPUT_CSV}")

    if not result_df.empty:
        print("\nBest random split by target, using highest R2:")
        random_df = result_df[result_df["evaluation"] == "random_split"].copy()

        if not random_df.empty:
            for target in TARGETS:
                tdf = random_df[random_df["target"] == target].copy()

                if tdf.empty:
                    continue

                if tdf["r2"].notna().any():
                    best = tdf.sort_values("r2", ascending=False).iloc[0]
                else:
                    best = tdf.sort_values("mae", ascending=True).iloc[0]

                print(
                    f"{target}: {best['reduction']} + {best['model']} | "
                    f"R2={best['r2']:.4f}, MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}"
                )

        print("\nBest k-fold by target, using lowest MAE:")
        kfold_df = result_df[result_df["evaluation"] == "kfold"].copy()

        if not kfold_df.empty:
            for target in TARGETS:
                tdf = kfold_df[kfold_df["target"] == target].copy()

                if tdf.empty:
                    continue

                best = tdf.sort_values("mae", ascending=True).iloc[0]

                print(
                    f"{target}: {best['reduction']} + {best['model']} | "
                    f"R2={best['r2']:.4f}, MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}"
                )


if __name__ == "__main__":
    main()
    