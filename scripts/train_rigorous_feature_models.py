#!/usr/bin/env python3
"""
train_rigorous_feature_models.py

Rigorous feature-engineering experiments for OpenROAD final ASIC QoR prediction.

Main goals:
- reduce feature/sample imbalance
- test engineered ratios and log transforms
- evaluate with group-aware validation by benchmark
- compare Ridge, ElasticNet, PLS, PCA, SelectKBest, and Gaussian Process

Dataset:
- data/processed/openroad_modeling_dataset.csv

Output:
- reports/modeling/openroad_rigorous_feature_results.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUTPUT_DIR = Path("reports/modeling")
OUTPUT_CSV = OUTPUT_DIR / "openroad_rigorous_feature_results.csv"

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

BENCHMARK_PREFIX = "benchmark_"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    target_mean = float(np.mean(np.abs(y_true)))

    return {
        "mae": mae,
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "relative_mae": mae / target_mean if target_mean != 0 else np.nan,
    }


def infer_benchmark_group(df: pd.DataFrame) -> pd.Series:
    benchmark_cols = [c for c in df.columns if c.startswith(BENCHMARK_PREFIX)]

    if not benchmark_cols:
        return pd.Series(["unknown"] * len(df), index=df.index)

    return (
        df[benchmark_cols]
        .idxmax(axis=1)
        .str.replace(BENCHMARK_PREFIX, "", regex=False)
    )


def safe_divide(a, b):
    return a / b.replace(0, np.nan)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has(cols):
        return all(c in df.columns for c in cols)

    if has(["num_arithmetic_ops", "num_array_accesses"]):
        df["ratio_compute_to_memory"] = safe_divide(
            df["num_arithmetic_ops"], df["num_array_accesses"]
        )

    if has(["num_mul_ops", "num_arithmetic_ops"]):
        df["ratio_mul_density"] = safe_divide(
            df["num_mul_ops"], df["num_arithmetic_ops"]
        )

    if has(["num_if_statements", "source_lines"]):
        df["ratio_control_density"] = safe_divide(
            df["num_if_statements"], df["source_lines"]
        )

    if has(["flipflops", "yosys_total_cells"]):
        df["ratio_ff_per_cell"] = safe_divide(
            df["flipflops"], df["yosys_total_cells"]
        )

    if has(["registers", "yosys_total_cells"]):
        df["ratio_registers_per_cell"] = safe_divide(
            df["registers"], df["yosys_total_cells"]
        )

    if has(["yosys_total_wire_bits", "yosys_total_cells"]):
        df["ratio_wire_bits_per_cell"] = safe_divide(
            df["yosys_total_wire_bits"], df["yosys_total_cells"]
        )

    if has(["yosys_total_wires", "yosys_total_cells"]):
        df["ratio_wires_per_cell"] = safe_divide(
            df["yosys_total_wires"], df["yosys_total_cells"]
        )

    if has(["mux_area", "total_area"]):
        df["ratio_mux_area_share"] = safe_divide(
            df["mux_area"], df["total_area"]
        )

    if has(["area_est", "control_steps"]):
        df["ratio_area_per_control_step"] = safe_divide(
            df["area_est"], df["control_steps"]
        )

    return df


def get_base_feature_columns(df: pd.DataFrame, target: str, include_benchmark_onehot: bool):
    drop_cols = set(OPENROAD_LEAKAGE_COLUMNS)
    drop_cols.add(target)

    feature_cols = []
    for col in df.columns:
        if col in drop_cols:
            continue

        if not include_benchmark_onehot and col.startswith(BENCHMARK_PREFIX):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def apply_feature_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()

    if "ratios" in mode:
        out = add_ratio_features(out)

    return out


def log1p_transform_array(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Make values nonnegative before log1p. Most hardware features are nonnegative,
    # but this protects min_slack or similar columns.
    min_vals = np.min(x, axis=0)
    shift = np.where(min_vals < 0, -min_vals, 0)
    return np.log1p(x + shift)


def make_kernel(kernel_name: str):
    if kernel_name == "matern":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(length_scale=1.0, nu=1.5, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    if kernel_name == "rbf":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=1.0, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    if kernel_name == "rq":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    raise ValueError(f"Unknown kernel: {kernel_name}")


def get_model_specs():
    specs = []

    specs.append(("ridge_a0.1", Ridge(alpha=0.1)))
    specs.append(("ridge_a1", Ridge(alpha=1.0)))
    specs.append(("ridge_a10", Ridge(alpha=10.0)))

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            specs.append((
                f"elasticnet_a{alpha}_l1{l1_ratio}",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=RANDOM_STATE,
                    max_iter=50000,
                )
            ))

    for n_components in [2, 3, 5]:
        specs.append((f"pls_{n_components}", PLSRegression(n_components=n_components)))

    for kernel_name in ["matern", "rbf", "rq"]:
        specs.append((
            f"gpr_{kernel_name}",
            GaussianProcessRegressor(
                kernel=make_kernel(kernel_name),
                normalize_y=True,
                alpha=1e-6,
                random_state=RANDOM_STATE,
            )
        ))

    return specs


def make_pipeline(
    model,
    reduction: str,
    n_train_rows: int,
    n_features: int,
    use_log_transform: bool,
):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_log_transform:
        steps.append(("log1p", FunctionTransformer(log1p_transform_array, validate=False)))

    steps.append(("scaler", StandardScaler()))

    if reduction.startswith("select_k_"):
        k = int(reduction.replace("select_k_", ""))
        k = max(1, min(k, n_features, n_train_rows - 1))
        steps.append(("feature_reduction", SelectKBest(score_func=f_regression, k=k)))

    elif reduction.startswith("pca_"):
        n_components = int(reduction.replace("pca_", ""))
        n_components = max(1, min(n_components, n_features, n_train_rows - 1))
        steps.append(("feature_reduction", PCA(n_components=n_components, random_state=RANDOM_STATE)))

    elif reduction == "none":
        pass

    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_random_split(df, target, feature_cols, model_name, model, reduction, use_log_transform):
    target_df = df.dropna(subset=[target]).copy()

    if len(target_df) < 6:
        return None

    X = target_df[feature_cols]
    y = target_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    pipe = make_pipeline(
        model=model,
        reduction=reduction,
        n_train_rows=len(X_train),
        n_features=len(feature_cols),
        use_log_transform=use_log_transform,
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    pred = np.asarray(pred).reshape(-1)

    return {
        "evaluation": "random_split",
        "target": target,
        "model": model_name,
        "reduction": reduction,
        "use_log_transform": use_log_transform,
        "rows": len(target_df),
        "features": len(feature_cols),
        "test_rows": len(y_test),
        **compute_metrics(y_test, pred),
    }


def evaluate_kfold(df, target, feature_cols, model_name, model, reduction, use_log_transform):
    target_df = df.dropna(subset=[target]).copy()

    if len(target_df) < 6:
        return None

    X = target_df[feature_cols]
    y = target_df[target]

    n_splits = min(5, len(target_df))
    if len(target_df) < 10:
        n_splits = 3

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipeline(
            model=model,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
            use_log_transform=use_log_transform,
        )

        pipe.fit(X_train, y_train)
        pred = np.asarray(pipe.predict(X_test)).reshape(-1)

        fold_rows.append(compute_metrics(y_test, pred))

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "kfold",
        "target": target,
        "model": model_name,
        "reduction": reduction,
        "use_log_transform": use_log_transform,
        "rows": len(target_df),
        "features": len(feature_cols),
        "folds": n_splits,
        "mae": fold_df["mae"].mean(),
        "mae_std": fold_df["mae"].std(),
        "rmse": fold_df["rmse"].mean(),
        "rmse_std": fold_df["rmse"].std(),
        "r2": fold_df["r2"].mean(),
        "r2_std": fold_df["r2"].std(),
        "relative_mae": fold_df["relative_mae"].mean(),
        "relative_mae_std": fold_df["relative_mae"].std(),
    }


def evaluate_groupkfold(df, target, feature_cols, model_name, model, reduction, use_log_transform):
    target_df = df.dropna(subset=[target]).copy()
    target_df["_benchmark_group"] = infer_benchmark_group(target_df)

    unique_groups = target_df["_benchmark_group"].nunique()

    if unique_groups < 2:
        return None

    X = target_df[feature_cols]
    y = target_df[target]
    groups = target_df["_benchmark_group"]

    splitter = GroupKFold(n_splits=unique_groups)
    fold_rows = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipeline(
            model=model,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
            use_log_transform=use_log_transform,
        )

        pipe.fit(X_train, y_train)
        pred = np.asarray(pipe.predict(X_test)).reshape(-1)

        row = compute_metrics(y_test, pred)
        row["heldout_group"] = groups.iloc[test_idx].iloc[0]
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "groupkfold_by_benchmark",
        "target": target,
        "model": model_name,
        "reduction": reduction,
        "use_log_transform": use_log_transform,
        "rows": len(target_df),
        "features": len(feature_cols),
        "groups": unique_groups,
        "mae": fold_df["mae"].mean(),
        "mae_std": fold_df["mae"].std(),
        "rmse": fold_df["rmse"].mean(),
        "rmse_std": fold_df["rmse"].std(),
        "r2": fold_df["r2"].mean(),
        "r2_std": fold_df["r2"].std(),
        "relative_mae": fold_df["relative_mae"].mean(),
        "relative_mae_std": fold_df["relative_mae"].std(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    base_df = pd.read_csv(DATASET)

    print("\n=== Rigorous OpenROAD feature modeling ===")
    print(f"Dataset: {DATASET}")
    print(f"Shape: {base_df.shape}")

    feature_modes = [
        "raw",
        "ratios",
    ]

    benchmark_modes = [
        True,
        False,
    ]

    reductions = [
        "none",
        "select_k_5",
        "select_k_10",
        "pca_3",
        "pca_5",
    ]

    model_specs = get_model_specs()

    rows = []

    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"Target: {target}")
        print("=" * 80)

        for feature_mode in feature_modes:
            df = apply_feature_mode(base_df, feature_mode)

            for include_benchmark_onehot in benchmark_modes:
                feature_cols = get_base_feature_columns(
                    df=df,
                    target=target,
                    include_benchmark_onehot=include_benchmark_onehot,
                )

                if not feature_cols:
                    continue

                for use_log_transform in [False, True]:
                    for reduction in reductions:
                        for model_name, model in model_specs:
                            # PLS is already a dimensionality-reduction regressor.
                            # Avoid stacking PCA/SelectKBest before PLS for now.
                            if model_name.startswith("pls_") and reduction != "none":
                                continue

                            print(
                                f"{target} | mode={feature_mode} | "
                                f"benchmark_onehot={include_benchmark_onehot} | "
                                f"log={use_log_transform} | reduction={reduction} | model={model_name}"
                            )

                            common = {
                                "feature_mode": feature_mode,
                                "include_benchmark_onehot": include_benchmark_onehot,
                            }

                            for eval_name, eval_func in [
                                ("random_split", evaluate_random_split),
                                ("kfold", evaluate_kfold),
                                ("groupkfold", evaluate_groupkfold),
                            ]:
                                try:
                                    result = eval_func(
                                        df=df,
                                        target=target,
                                        feature_cols=feature_cols,
                                        model_name=model_name,
                                        model=model,
                                        reduction=reduction,
                                        use_log_transform=use_log_transform,
                                    )

                                    if result is not None:
                                        result.update(common)
                                        rows.append(result)

                                except Exception as exc:
                                    rows.append({
                                        "evaluation": eval_name,
                                        "target": target,
                                        "feature_mode": feature_mode,
                                        "include_benchmark_onehot": include_benchmark_onehot,
                                        "use_log_transform": use_log_transform,
                                        "reduction": reduction,
                                        "model": model_name,
                                        "error": str(exc),
                                    })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Rigorous modeling complete ===")
    print(f"Output: {OUTPUT_CSV}")

    for evaluation in ["random_split", "kfold", "groupkfold_by_benchmark"]:
        eval_df = results[results["evaluation"] == evaluation].copy()

        if eval_df.empty:
            continue

        print("\n" + "=" * 80)
        print(f"Best results for {evaluation}")
        print("=" * 80)

        for target in TARGETS:
            tdf = eval_df[eval_df["target"] == target].copy()
            tdf = tdf.dropna(subset=["mae"])

            if tdf.empty:
                continue

            # For tiny data, relative MAE is usually more robust than R2.
            best = tdf.sort_values("relative_mae", ascending=True).iloc[0]

            print(
                f"{target}: "
                f"rel_MAE={best['relative_mae']:.4f}, "
                f"MAE={best['mae']:.4f}, "
                f"RMSE={best['rmse']:.4f}, "
                f"R2={best['r2']:.4f}, "
                f"mode={best['feature_mode']}, "
                f"bench_onehot={best['include_benchmark_onehot']}, "
                f"log={best['use_log_transform']}, "
                f"reduction={best['reduction']}, "
                f"model={best['model']}"
            )


if __name__ == "__main__":
    main()
    