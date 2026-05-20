#!/usr/bin/env python3
"""
train_eda_aware_models.py

EDA-aware modeling experiments for OpenROAD final ASIC QoR prediction.

Main ideas:
- Replace benchmark one-hot memorization with semantic benchmark descriptors.
- Add physically meaningful ratio features.
- Predict normalized/residual targets:
    area_ratio  = final_area / Bambu area_est
    delay_ratio = final_delay / OpenROAD input clock constraint
    power_per_area = final_power / Bambu area_est
- Reconstruct final predictions and evaluate in original units.
- Use leave-one-benchmark-out / GroupKFold validation.

Input:
- data/processed/openroad_modeling_dataset.csv

Output:
- reports/modeling/openroad_eda_aware_results.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUTPUT_DIR = Path("reports/modeling")
OUTPUT_CSV = OUTPUT_DIR / "openroad_eda_aware_results.csv"

RANDOM_STATE = 42

BENCHMARK_PREFIX = "benchmark_"

ORIGINAL_TARGETS = {
    "area_ratio": {
        "original_target": "openroad_synth_chip_area",
        "denominator": "area_est",
        "description": "final_area / area_est",
    },
    "delay_ratio": {
        "original_target": "openroad_critical_path_delay",
        "denominator": "openroad_clock_constraint",
        "description": "critical_path_delay / openroad_clock_constraint",
    },
    "power_per_area": {
        "original_target": "openroad_total_power",
        "denominator": "area_est",
        "description": "total_power / area_est",
    },
}

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


SEMANTIC_BENCHMARK_FEATURES = {
    # Values are intentionally coarse descriptors, not derived from OpenROAD.
    # scale: 0 low, 1 medium, 2 high where applicable
    "2mm": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "3mm": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "atax": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "bicg": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "floyd-warshall": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 0,
        "sem_regular_access": 0,
        "sem_matrix_kernel": 1,
    },
    "gemm": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "gemver": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "gesummv": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "jacobi-1d": {
        "sem_compute_intensity": 0,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 1,
        "sem_has_reduction": 0,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 0,
    },
    "jacobi-2d": {
        "sem_compute_intensity": 0,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 1,
        "sem_has_reduction": 0,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 0,
    },
    "mvt": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "nussinov": {
        "sem_compute_intensity": 1,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 0,
        "sem_has_reduction": 0,
        "sem_regular_access": 0,
        "sem_matrix_kernel": 1,
    },
    "seidel-2d": {
        "sem_compute_intensity": 0,
        "sem_memory_intensity": 2,
        "sem_is_stencil": 1,
        "sem_has_reduction": 0,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 0,
    },
    "symm": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "syr2k": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "syrk": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
    "trmm": {
        "sem_compute_intensity": 2,
        "sem_memory_intensity": 1,
        "sem_is_stencil": 0,
        "sem_has_reduction": 1,
        "sem_regular_access": 1,
        "sem_matrix_kernel": 1,
    },
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


def add_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    groups = infer_benchmark_group(df)

    semantic_cols = [
        "sem_compute_intensity",
        "sem_memory_intensity",
        "sem_is_stencil",
        "sem_has_reduction",
        "sem_regular_access",
        "sem_matrix_kernel",
    ]

    for col in semantic_cols:
        df[col] = groups.map(
            lambda b: SEMANTIC_BENCHMARK_FEATURES.get(b, {}).get(col, 0)
        )

    return df


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has(cols):
        return all(c in df.columns for c in cols)

    if has(["num_arithmetic_ops", "num_array_accesses"]):
        df["ratio_compute_to_memory"] = safe_divide(df["num_arithmetic_ops"], df["num_array_accesses"])

    if has(["num_mul_ops", "num_arithmetic_ops"]):
        df["ratio_mul_density"] = safe_divide(df["num_mul_ops"], df["num_arithmetic_ops"])

    if has(["num_if_statements", "source_lines"]):
        df["ratio_control_density"] = safe_divide(df["num_if_statements"], df["source_lines"])

    if has(["flipflops", "yosys_total_cells"]):
        df["ratio_ff_per_cell"] = safe_divide(df["flipflops"], df["yosys_total_cells"])

    if has(["registers", "yosys_total_cells"]):
        df["ratio_registers_per_cell"] = safe_divide(df["registers"], df["yosys_total_cells"])

    if has(["yosys_total_wire_bits", "yosys_total_cells"]):
        df["ratio_wire_bits_per_cell"] = safe_divide(df["yosys_total_wire_bits"], df["yosys_total_cells"])

    if has(["yosys_total_wires", "yosys_total_cells"]):
        df["ratio_wires_per_cell"] = safe_divide(df["yosys_total_wires"], df["yosys_total_cells"])

    if has(["mux_area", "total_area"]):
        df["ratio_mux_area_share"] = safe_divide(df["mux_area"], df["total_area"])

    if has(["area_est", "control_steps"]):
        df["ratio_area_per_control_step"] = safe_divide(df["area_est"], df["control_steps"])

    return df


def add_normalized_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for normalized_target, cfg in ORIGINAL_TARGETS.items():
        original = cfg["original_target"]
        denom = cfg["denominator"]

        if original in df.columns and denom in df.columns:
            df[normalized_target] = safe_divide(df[original], df[denom])

    return df


def get_feature_columns(df: pd.DataFrame, normalized_target: str, include_benchmark_onehot: bool):
    drop_cols = set(OPENROAD_LEAKAGE_COLUMNS)
    drop_cols.update(ORIGINAL_TARGETS.keys())
    drop_cols.add(normalized_target)

    feature_cols = []

    for col in df.columns:
        if col in drop_cols:
            continue

        if not include_benchmark_onehot and col.startswith(BENCHMARK_PREFIX):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def make_kernel(name: str):
    if name == "matern":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(length_scale=1.0, nu=1.5, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    if name == "rbf":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=1.0, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    if name == "rq":
        return (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")
        )

    raise ValueError(name)


def get_model_specs():
    specs = [
        ("ridge_a0.1", Ridge(alpha=0.1)),
        ("ridge_a1", Ridge(alpha=1.0)),
        ("ridge_a10", Ridge(alpha=10.0)),
    ]

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            specs.append((
                f"elasticnet_a{alpha}_l1{l1_ratio}",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=RANDOM_STATE,
                    max_iter=50000,
                ),
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
            ),
        ))

    return specs


def make_pipeline(model, scaler_name: str, reduction: str, n_train_rows: int, n_features: int):
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ]

    if reduction.startswith("select_k_"):
        k = int(reduction.replace("select_k_", ""))
        k = max(1, min(k, n_features, n_train_rows - 1))
        steps.append(("select", SelectKBest(score_func=f_regression, k=k)))

    elif reduction == "none":
        pass

    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    steps.append(("model", model))
    return Pipeline(steps)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def compute_original_metrics(y_true_original, y_pred_original):
    mae = mean_absolute_error(y_true_original, y_pred_original)
    mean_abs = float(np.mean(np.abs(y_true_original)))

    return {
        "mae_original": mae,
        "rmse_original": rmse(y_true_original, y_pred_original),
        "r2_original": safe_r2(y_true_original, y_pred_original),
        "relative_mae_original": mae / mean_abs if mean_abs != 0 else np.nan,
    }


def reconstruct_original_prediction(pred_normalized, denominator):
    return np.asarray(pred_normalized).reshape(-1) * np.asarray(denominator).reshape(-1)


def evaluate_random_split(df, normalized_target, feature_cols, model_name, model, scaler_name, reduction):
    cfg = ORIGINAL_TARGETS[normalized_target]
    original_target = cfg["original_target"]
    denominator = cfg["denominator"]

    target_df = df.dropna(subset=[normalized_target, original_target, denominator]).copy()

    if len(target_df) < 6:
        return None

    X = target_df[feature_cols]
    y_norm = target_df[normalized_target]
    y_orig = target_df[original_target]
    denom = target_df[denominator]

    X_train, X_test, y_train, y_test, y_orig_train, y_orig_test, denom_train, denom_test = train_test_split(
        X,
        y_norm,
        y_orig,
        denom,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    pipe = make_pipeline(
        model=model,
        scaler_name=scaler_name,
        reduction=reduction,
        n_train_rows=len(X_train),
        n_features=len(feature_cols),
    )

    pipe.fit(X_train, y_train)
    pred_norm = pipe.predict(X_test)
    pred_orig = reconstruct_original_prediction(pred_norm, denom_test)

    return {
        "evaluation": "random_split",
        "normalized_target": normalized_target,
        "original_target": original_target,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
        "rows": len(target_df),
        "features": len(feature_cols),
        "test_rows": len(y_test),
        **compute_original_metrics(y_orig_test, pred_orig),
    }


def evaluate_kfold(df, normalized_target, feature_cols, model_name, model, scaler_name, reduction):
    cfg = ORIGINAL_TARGETS[normalized_target]
    original_target = cfg["original_target"]
    denominator = cfg["denominator"]

    target_df = df.dropna(subset=[normalized_target, original_target, denominator]).copy()

    if len(target_df) < 6:
        return None

    X = target_df[feature_cols]
    y_norm = target_df[normalized_target]
    y_orig = target_df[original_target]
    denom = target_df[denominator]

    n_splits = min(5, len(target_df))
    if len(target_df) < 10:
        n_splits = 3

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_norm.iloc[train_idx]
        y_orig_test = y_orig.iloc[test_idx]
        denom_test = denom.iloc[test_idx]

        pipe = make_pipeline(
            model=model,
            scaler_name=scaler_name,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
        )

        pipe.fit(X_train, y_train)
        pred_norm = pipe.predict(X_test)
        pred_orig = reconstruct_original_prediction(pred_norm, denom_test)

        fold_rows.append(compute_original_metrics(y_orig_test, pred_orig))

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "kfold",
        "normalized_target": normalized_target,
        "original_target": original_target,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
        "rows": len(target_df),
        "features": len(feature_cols),
        "folds": n_splits,
        "mae_original": fold_df["mae_original"].mean(),
        "mae_original_std": fold_df["mae_original"].std(),
        "rmse_original": fold_df["rmse_original"].mean(),
        "rmse_original_std": fold_df["rmse_original"].std(),
        "r2_original": fold_df["r2_original"].mean(),
        "r2_original_std": fold_df["r2_original"].std(),
        "relative_mae_original": fold_df["relative_mae_original"].mean(),
        "relative_mae_original_std": fold_df["relative_mae_original"].std(),
    }


def evaluate_groupkfold(df, normalized_target, feature_cols, model_name, model, scaler_name, reduction):
    cfg = ORIGINAL_TARGETS[normalized_target]
    original_target = cfg["original_target"]
    denominator = cfg["denominator"]

    target_df = df.dropna(subset=[normalized_target, original_target, denominator]).copy()
    target_df["_benchmark_group"] = infer_benchmark_group(target_df)

    unique_groups = target_df["_benchmark_group"].nunique()

    if unique_groups < 2:
        return None

    X = target_df[feature_cols]
    y_norm = target_df[normalized_target]
    y_orig = target_df[original_target]
    denom = target_df[denominator]
    groups = target_df["_benchmark_group"]

    splitter = GroupKFold(n_splits=unique_groups)

    fold_rows = []

    for train_idx, test_idx in splitter.split(X, y_norm, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_norm.iloc[train_idx]
        y_orig_test = y_orig.iloc[test_idx]
        denom_test = denom.iloc[test_idx]

        pipe = make_pipeline(
            model=model,
            scaler_name=scaler_name,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
        )

        pipe.fit(X_train, y_train)
        pred_norm = pipe.predict(X_test)
        pred_orig = reconstruct_original_prediction(pred_norm, denom_test)

        row = compute_original_metrics(y_orig_test, pred_orig)
        row["heldout_group"] = groups.iloc[test_idx].iloc[0]
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "groupkfold_by_benchmark",
        "normalized_target": normalized_target,
        "original_target": original_target,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
        "rows": len(target_df),
        "features": len(feature_cols),
        "groups": unique_groups,
        "mae_original": fold_df["mae_original"].mean(),
        "mae_original_std": fold_df["mae_original"].std(),
        "rmse_original": fold_df["rmse_original"].mean(),
        "rmse_original_std": fold_df["rmse_original"].std(),
        "r2_original": fold_df["r2_original"].mean(),
        "r2_original_std": fold_df["r2_original"].std(),
        "relative_mae_original": fold_df["relative_mae_original"].mean(),
        "relative_mae_original_std": fold_df["relative_mae_original"].std(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    df = add_semantic_features(df)
    df = add_ratio_features(df)
    df = add_normalized_targets(df)

    print("\n=== EDA-aware OpenROAD modeling ===")
    print(f"Dataset: {DATASET}")
    print(f"Shape after feature engineering: {df.shape}")

    reductions = [
        "none",
        "select_k_5",
        "select_k_10",
    ]

    scalers = [
        "standard",
        "robust",
    ]

    benchmark_onehot_modes = [
        False,
        True,
    ]

    model_specs = get_model_specs()

    rows = []

    for normalized_target in ORIGINAL_TARGETS:
        print("\n" + "=" * 80)
        print(f"Normalized target: {normalized_target} ({ORIGINAL_TARGETS[normalized_target]['description']})")
        print("=" * 80)

        for include_benchmark_onehot in benchmark_onehot_modes:
            feature_cols = get_feature_columns(
                df=df,
                normalized_target=normalized_target,
                include_benchmark_onehot=include_benchmark_onehot,
            )

            print(f"benchmark_onehot={include_benchmark_onehot}, features={len(feature_cols)}")

            for scaler_name in scalers:
                for reduction in reductions:
                    for model_name, model in model_specs:
                        if model_name.startswith("pls_") and reduction != "none":
                            continue

                        print(
                            f"{normalized_target} | onehot={include_benchmark_onehot} | "
                            f"scaler={scaler_name} | reduction={reduction} | model={model_name}"
                        )

                        common = {
                            "include_benchmark_onehot": include_benchmark_onehot,
                        }

                        for eval_func in [evaluate_random_split, evaluate_kfold, evaluate_groupkfold]:
                            try:
                                result = eval_func(
                                    df=df,
                                    normalized_target=normalized_target,
                                    feature_cols=feature_cols,
                                    model_name=model_name,
                                    model=model,
                                    scaler_name=scaler_name,
                                    reduction=reduction,
                                )

                                if result is not None:
                                    result.update(common)
                                    rows.append(result)

                            except Exception as exc:
                                rows.append({
                                    "evaluation": eval_func.__name__,
                                    "normalized_target": normalized_target,
                                    "original_target": ORIGINAL_TARGETS[normalized_target]["original_target"],
                                    "include_benchmark_onehot": include_benchmark_onehot,
                                    "scaler": scaler_name,
                                    "reduction": reduction,
                                    "model": model_name,
                                    "error": str(exc),
                                })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    print("\n=== EDA-aware modeling complete ===")
    print(f"Output: {OUTPUT_CSV}")

    for evaluation in ["random_split", "kfold", "groupkfold_by_benchmark"]:
        eval_df = results[results["evaluation"] == evaluation].copy()
        eval_df = eval_df.dropna(subset=["relative_mae_original"])

        if eval_df.empty:
            continue

        print("\n" + "=" * 80)
        print(f"Best EDA-aware results for {evaluation}")
        print("=" * 80)

        for normalized_target, cfg in ORIGINAL_TARGETS.items():
            original_target = cfg["original_target"]
            tdf = eval_df[eval_df["normalized_target"] == normalized_target].copy()

            if tdf.empty:
                continue

            best = tdf.sort_values("relative_mae_original", ascending=True).iloc[0]

            print(
                f"{original_target} via {normalized_target}: "
                f"rel_MAE={best['relative_mae_original']:.4f}, "
                f"MAE={best['mae_original']:.4f}, "
                f"RMSE={best['rmse_original']:.4f}, "
                f"R2={best['r2_original']:.4f}, "
                f"onehot={best['include_benchmark_onehot']}, "
                f"scaler={best['scaler']}, "
                f"reduction={best['reduction']}, "
                f"model={best['model']}"
            )


if __name__ == "__main__":
    main()