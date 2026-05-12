#!/usr/bin/env python3
"""
extract_static_features.py

Extract lightweight static code features from PolyBench C benchmark source files.

The extracted features describe the source code structure before HLS:
- source lines
- loops
- conditionals
- array accesses
- arithmetic operators
- approximate function calls
- memory-like references

Output:
    data/extracted_metrics/static_code_features.csv
"""

from pathlib import Path
import argparse
import re

import pandas as pd


DEFAULT_POLYBENCH_ROOT = Path(
    "D:/qor_project/workspace/PandA-bambu/examples/PolyBench/PolyBenchC"
)

DEFAULT_OUTPUT = Path("data/extracted_metrics/static_code_features.csv")


BENCHMARKS = {
    # Linear algebra / BLAS
    "gemm": ("linear-algebra/blas/gemm", "gemm.c"),
    "gemver": ("linear-algebra/blas/gemver", "gemver.c"),
    "gesummv": ("linear-algebra/blas/gesummv", "gesummv.c"),
    "symm": ("linear-algebra/blas/symm", "symm.c"),
    "syr2k": ("linear-algebra/blas/syr2k", "syr2k.c"),
    "syrk": ("linear-algebra/blas/syrk", "syrk.c"),
    "trmm": ("linear-algebra/blas/trmm", "trmm.c"),

    # Linear algebra / kernels
    "atax": ("linear-algebra/kernels/atax", "atax.c"),
    "bicg": ("linear-algebra/kernels/bicg", "bicg.c"),
    "2mm": ("linear-algebra/kernels/2mm", "2mm.c"),
    "3mm": ("linear-algebra/kernels/3mm", "3mm.c"),
    "mvt": ("linear-algebra/kernels/mvt", "mvt.c"),

    # Stencils
    "jacobi-1d": ("stencils/jacobi-1d", "jacobi-1d.c"),
    "jacobi-2d": ("stencils/jacobi-2d", "jacobi-2d.c"),
    "seidel-2d": ("stencils/seidel-2d", "seidel-2d.c"),

    # Medley
    "floyd-warshall": ("medley/floyd-warshall", "floyd-warshall.c"),
    "deriche": ("medley/deriche", "deriche.c"),
    "nussinov": ("medley/nussinov", "nussinov.c"),
}


def remove_comments(code: str) -> str:
    """Remove C/C++ style comments."""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def count_max_loop_depth(code: str) -> int:
    """
    Approximate maximum loop nesting depth.

    This is a heuristic based on braces after for/while loops.
    It is not a full C parser, but works reasonably for PolyBench-style code.
    """
    tokens = re.findall(r"\bfor\s*\(|\bwhile\s*\(|\{|\}", code)

    depth = 0
    max_depth = 0
    pending_loop = False

    for token in tokens:
        if token.startswith("for") or token.startswith("while"):
            pending_loop = True
        elif token == "{":
            if pending_loop:
                depth += 1
                max_depth = max(max_depth, depth)
                pending_loop = False
        elif token == "}":
            if depth > 0:
                depth -= 1

    return max_depth


def extract_features_from_file(benchmark: str, source_path: Path) -> dict:
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found for {benchmark}: {source_path}")

    raw_code = source_path.read_text(encoding="utf-8", errors="ignore")
    code = remove_comments(raw_code)

    nonempty_lines = [
        line for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    num_for_loops = len(re.findall(r"\bfor\s*\(", code))
    num_while_loops = len(re.findall(r"\bwhile\s*\(", code))
    num_if_statements = len(re.findall(r"\bif\s*\(", code))

    # Array access approximation: counts occurrences of [ ... ].
    num_array_accesses = len(re.findall(r"\[[^\]]+\]", code))

    # Arithmetic operator approximation.
    # This is intentionally simple and heuristic.
    num_add_ops = len(re.findall(r"(?<!\+)\+(?!\+|=)", code))
    num_sub_ops = len(re.findall(r"(?<!-)-(?!-|=|>)", code))
    num_mul_ops = len(re.findall(r"(?<!/)\*(?!=|/)", code))
    num_div_ops = len(re.findall(r"(?<!/)/(?!/|\*)", code))

    num_arithmetic_ops = num_add_ops + num_sub_ops + num_mul_ops + num_div_ops

    # Approximate assignment operations.
    num_assignments = len(re.findall(r"(?<![=!<>])=(?!=)", code))

    # Approximate function calls:
    # identifier(...) but exclude control keywords.
    call_candidates = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
    excluded = {"if", "for", "while", "switch", "return", "sizeof"}
    num_function_calls = sum(1 for name in call_candidates if name not in excluded)

    # Variable declarations: rough approximation.
    num_int_decls = len(re.findall(r"\bint\s+[A-Za-z_]", code))
    num_float_decls = len(re.findall(r"\bfloat\s+[A-Za-z_]", code))
    num_double_decls = len(re.findall(r"\bdouble\s+[A-Za-z_]", code))
    num_var_decls = num_int_decls + num_float_decls + num_double_decls

    return {
        "benchmark": benchmark,
        "source_file": str(source_path),
        "source_lines": len(nonempty_lines),
        "num_for_loops": num_for_loops,
        "num_while_loops": num_while_loops,
        "num_total_loops": num_for_loops + num_while_loops,
        "max_loop_depth": count_max_loop_depth(code),
        "num_if_statements": num_if_statements,
        "num_array_accesses": num_array_accesses,
        "num_add_ops": num_add_ops,
        "num_sub_ops": num_sub_ops,
        "num_mul_ops": num_mul_ops,
        "num_div_ops": num_div_ops,
        "num_arithmetic_ops": num_arithmetic_ops,
        "num_assignments": num_assignments,
        "num_function_calls": num_function_calls,
        "num_var_decls": num_var_decls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static code features from PolyBench C files.")
    parser.add_argument(
        "--polybench-root",
        type=Path,
        default=DEFAULT_POLYBENCH_ROOT,
        help="Path to PolyBenchC root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    rows = []

    for benchmark, (relative_dir, filename) in BENCHMARKS.items():
        source_path = args.polybench_root / relative_dir / filename
        features = extract_features_from_file(benchmark, source_path)
        rows.append(features)

    df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n=== Static feature extraction complete ===")
    print(f"PolyBench root: {args.polybench_root}")
    print(f"Output: {args.output}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(df.head())


if __name__ == "__main__":
    main()
