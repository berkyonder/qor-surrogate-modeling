#!/usr/bin/env python3
"""
train_ablation_models.py

Direct-target ablation study for OpenROAD final ASIC QoR prediction.

Purpose:
- Keep targets as direct final OpenROAD outputs.
- Compare which feature groups are useful:
    code_only
    hls_only
    yosys_only
    hls_plus_yosys
    ratios_only
    all_direct_features
- Evaluate with random split, k-fold, and GroupKFold by benchmark.

Input:
- data/processed/openroad_modeling_dataset.csv

Output:
- reports/modeling/openroad_ablation_results.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUTPUT_DIR = Path("reports/modeling")
OUTPUT_CSV = OUTPUT_DIR / "openroad_ablation_results.csv"

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

CODE_FEATURES = [
    "source_lines",
    "num_for_loops",
    "num_while_loops",
    "num_total_loops",
    "max_loop_depth",
    "num_if_statements",
    "num_array_accesses",
    "num_add_ops",
    "num_sub_ops",
    "num_mul_ops",
    "num_div_ops",
    "num_arithmetic_ops",
    "num_assignments",
    "num_function_calls",
    "num_var_decls",
]

HLS_FEATURES = [
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
]

YOSYS_FEATURES = [
    "yosys_num_wires",
    "yosys_num_wire_bits",
    "yosys_num_pub_wires",
    "yosys_num_pub_wire_bits",
    "yosys_num_memories",
    "yosys_num_memory_bits",
    "yosys_num_processes",
    "yosys_num_cells",
    "yosys_total_modules",
    "yosys_total_wires",
    "yosys_total_wire_bits",
    "yosys_total_pub_wires",
    "yosys_total_pub_wire_bits",
    "yosys_total_memories",
    "yosys_total_memory_bits",
    "yosys_total_processes",
    "yosys_total_cells",
]

CONFIG_FEATURE_PREFIXES = [
    "dataset_size_",
    "mem_policy_",
    "opt_level_",
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
    mean_abs = float(np.mean(np.abs(y_true)))
    return {
        "mae": mae,
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "relative_mae": mae / mean_abs if mean_abs != 0 else np.nan,
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


def existing(cols, df):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def config_features(df):
    cols = []
    for c in df.columns:
        if any(c.startswith(prefix) for prefix in CONFIG_FEATURE_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def benchmark_features(df):
    return [
        c for c in df.columns
        if c.startswith(BENCHMARK_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]


def ratio_features(df):
    return [
        c for c in df.columns
        if c.startswith("ratio_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def all_allowed_numeric_features(df, target, include_benchmark_onehot):
    drop = set(OPENROAD_LEAKAGE_COLUMNS)
    drop.add(target)

    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if not include_benchmark_onehot and c.startswith(BENCHMARK_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def get_feature_set(df, feature_set_name, target, include_benchmark_onehot):
    cfg = config_features(df)
    bench = benchmark_features(df) if include_benchmark_onehot else []

    code = existing(CODE_FEATURES, df)
    hls = existing(HLS_FEATURES, df)
    yosys = existing(YOSYS_FEATURES, df)
    ratios = ratio_features(df)

    if feature_set_name == "code_only":
        cols = code + cfg + bench

    elif feature_set_name == "hls_only":
        cols = hls + cfg + bench

    elif feature_set_name == "yosys_only":
        cols = yosys + cfg + bench

    elif feature_set_name == "hls_plus_yosys":
        cols = hls + yosys + cfg + bench

    elif feature_set_name == "ratios_only":
        cols = ratios + cfg + bench

    elif feature_set_name == "all_direct_features":
        cols = all_allowed_numeric_features(df, target, include_benchmark_onehot)

    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    # Remove target/leakage just in case.
    cols = [
        c for c in cols
        if c not in OPENROAD_LEAKAGE_COLUMNS and c != target
    ]

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for c in cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped


def make_kernel(name):
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

    raise ValueError(name)


def get_model_specs():
    specs = []

    specs.extend([
        ("ridge_a0.1", Ridge(alpha=0.1)),
        ("ridge_a1", Ridge(alpha=1.0)),
        ("ridge_a10", Ridge(alpha=10.0)),
    ])

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

    specs.append((
        "gpr_matern",
        GaussianProcessRegressor(
            kernel=make_kernel("matern"),
            normalize_y=True,
            alpha=1e-6,
            random_state=RANDOM_STATE,
        )
    ))

    specs.append((
        "gpr_rbf",
        GaussianProcessRegressor(
            kernel=make_kernel("rbf"),
            normalize_y=True,
            alpha=1e-6,
            random_state=RANDOM_STATE,
        )
    ))

    return specs


def make_pipeline(model, scaler_name, reduction, n_train_rows, n_features):
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


def evaluate_random_split(df, target, feature_cols, feature_set_name, include_benchmark_onehot, model_name, model, scaler_name, reduction):
    target_df = df.dropna(subset=[target]).copy()

    if len(target_df) < 6 or len(feature_cols) == 0:
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
        scaler_name=scaler_name,
        reduction=reduction,
        n_train_rows=len(X_train),
        n_features=len(feature_cols),
    )

    pipe.fit(X_train, y_train)
    pred = np.asarray(pipe.predict(X_test)).reshape(-1)

    return {
        "evaluation": "random_split",
        "target": target,
        "feature_set": feature_set_name,
        "include_benchmark_onehot": include_benchmark_onehot,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
        "rows": len(target_df),
        "features": len(feature_cols),
        "test_rows": len(y_test),
        **compute_metrics(y_test, pred),
    }


def evaluate_kfold(df, target, feature_cols, feature_set_name, include_benchmark_onehot, model_name, model, scaler_name, reduction):
    target_df = df.dropna(subset=[target]).copy()

    if len(target_df) < 6 or len(feature_cols) == 0:
        return None

    X = target_df[feature_cols]
    y = target_df[target]

    n_splits = min(5, len(target_df))
    if len(target_df) < 10:
        n_splits = 3

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipeline(
            model=model,
            scaler_name=scaler_name,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
        )

        pipe.fit(X_train, y_train)
        pred = np.asarray(pipe.predict(X_test)).reshape(-1)
        fold_rows.append(compute_metrics(y_test, pred))

    fold_df = pd.DataFrame(fold_rows)

    return {
        "evaluation": "kfold",
        "target": target,
        "feature_set": feature_set_name,
        "include_benchmark_onehot": include_benchmark_onehot,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
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


def evaluate_groupkfold(df, target, feature_cols, feature_set_name, include_benchmark_onehot, model_name, model, scaler_name, reduction):
    target_df = df.dropna(subset=[target]).copy()
    target_df["_benchmark_group"] = infer_benchmark_group(target_df)

    unique_groups = target_df["_benchmark_group"].nunique()

    if unique_groups < 2 or len(feature_cols) == 0:
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
            scaler_name=scaler_name,
            reduction=reduction,
            n_train_rows=len(X_train),
            n_features=len(feature_cols),
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
        "feature_set": feature_set_name,
        "include_benchmark_onehot": include_benchmark_onehot,
        "model": model_name,
        "scaler": scaler_name,
        "reduction": reduction,
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

    df = pd.read_csv(DATASET)
    df = add_ratio_features(df)

    print("\n=== OpenROAD ablation study ===")
    print(f"Dataset: {DATASET}")
    print(f"Shape after ratio features: {df.shape}")

    feature_sets = [
        "code_only",
        "hls_only",
        "yosys_only",
        "hls_plus_yosys",
        "ratios_only",
        "all_direct_features",
    ]

    benchmark_modes = [
        False,
        True,
    ]

    scalers = [
        "standard",
        "robust",
    ]

    reductions = [
        "none",
        "select_k_5",
        "select_k_10",
    ]

    model_specs = get_model_specs()

    rows = []

    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"Target: {target}")
        print("=" * 80)

        for feature_set_name in feature_sets:
            for include_benchmark_onehot in benchmark_modes:
                feature_cols = get_feature_set(
                    df=df,
                    feature_set_name=feature_set_name,
                    target=target,
                    include_benchmark_onehot=include_benchmark_onehot,
                )

                print(
                    f"Feature set: {feature_set_name}, "
                    f"benchmark_onehot={include_benchmark_onehot}, "
                    f"features={len(feature_cols)}"
                )

                for scaler_name in scalers:
                    for reduction in reductions:
                        for model_name, model in model_specs:
                            if model_name.startswith("pls_") and reduction != "none":
                                continue

                            print(
                                f"{target} | {feature_set_name} | "
                                f"onehot={include_benchmark_onehot} | "
                                f"scaler={scaler_name} | reduction={reduction} | model={model_name}"
                            )

                            for eval_func in [
                                evaluate_random_split,
                                evaluate_kfold,
                                evaluate_groupkfold,
                            ]:
                                try:
                                    result = eval_func(
                                        df=df,
                                        target=target,
                                        feature_cols=feature_cols,
                                        feature_set_name=feature_set_name,
                                        include_benchmark_onehot=include_benchmark_onehot,
                                        model_name=model_name,
                                        model=model,
                                        scaler_name=scaler_name,
                                        reduction=reduction,
                                    )

                                    if result is not None:
                                        rows.append(result)

                                except Exception as exc:
                                    rows.append({
                                        "evaluation": eval_func.__name__,
                                        "target": target,
                                        "feature_set": feature_set_name,
                                        "include_benchmark_onehot": include_benchmark_onehot,
                                        "model": model_name,
                                        "scaler": scaler_name,
                                        "reduction": reduction,
                                        "error": str(exc),
                                    })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Ablation study complete ===")
    print(f"Output: {OUTPUT_CSV}")

    for evaluation in ["random_split", "kfold", "groupkfold_by_benchmark"]:
        eval_df = results[results["evaluation"] == evaluation].copy()
        eval_df = eval_df.dropna(subset=["relative_mae"])

        if eval_df.empty:
            continue

        print("\n" + "=" * 80)
        print(f"Best ablation results for {evaluation}")
        print("=" * 80)

        for target in TARGETS:
            tdf = eval_df[eval_df["target"] == target].copy()

            if tdf.empty:
                continue

            best = tdf.sort_values("relative_mae", ascending=True).iloc[0]

            print(
                f"{target}: "
                f"rel_MAE={best['relative_mae']:.4f}, "
                f"MAE={best['mae']:.4f}, "
                f"RMSE={best['rmse']:.4f}, "
                f"R2={best['r2']:.4f}, "
                f"feature_set={best['feature_set']}, "
                f"onehot={best['include_benchmark_onehot']}, "
                f"scaler={best['scaler']}, "
                f"reduction={best['reduction']}, "
                f"model={best['model']}"
            )


if __name__ == "__main__":
    main()