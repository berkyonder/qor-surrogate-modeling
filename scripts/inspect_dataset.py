#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

DATASET = Path(__file__).resolve().parent.parent / "data" / "extracted_metrics" / "early_hls_metrics_raw.csv"

df = pd.read_csv(DATASET)

print("\n=== Dataset shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Missing values ===")
print(df.isna().sum())

print("\n=== Dataset sizes ===")
print(df["dataset_size"].value_counts())

print("\n=== Benchmarks ===")
print(df["benchmark"].value_counts().sort_index())

numeric_cols = df.select_dtypes(include="number").columns

print("\n=== Identifier vs Metric Columns ===")
text_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Text/ID columns: {text_cols}")
print(f"Metric columns: {list(numeric_cols)}")

print("\n=== Duplicates in metric columns only ===")
# Check for duplicate metric values (excluding text identifiers)
metric_dups = df[numeric_cols].duplicated().sum()
print(f"Rows with duplicate metric values: {metric_dups}")
if metric_dups > 0:
    dup_rows = df[df[numeric_cols].duplicated(keep=False)]
    print("\nRows with duplicate metrics:")
    print(dup_rows[["benchmark", "dataset_size"] + list(numeric_cols)])

print("\n=== Constant numeric columns ===")
constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
print(constant_cols)

print("\n=== Numeric summary ===")
print(df[numeric_cols].describe().T)

print("\n=== Correlation with total_area ===")
print(df[numeric_cols].corr()["total_area"].sort_values(ascending=False))

print("\n=== Correlation with control_steps ===")
print(df[numeric_cols].corr()["control_steps"].sort_values(ascending=False))

print("\n=== Mean total_area by opt_level ===")
print(df.groupby("opt_level")["total_area"].mean())

print("\n=== Mean control_steps by opt_level ===")
print(df.groupby("opt_level")["control_steps"].mean())

print("\n=== Mean total_area by mem_policy ===")
print(df.groupby("mem_policy")["total_area"].mean())

print("\n=== Mean control_steps by mem_policy ===")
print(df.groupby("mem_policy")["control_steps"].mean())

