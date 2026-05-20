#!/usr/bin/env python3
"""
parse_yosys_reports.py

Parse Yosys JSON/stat reports and extract ASIC synthesis QoR metrics.

Input:
    data/yosys_reports/*_yosys.json

Output:
    data/extracted_metrics/yosys_qor_targets.csv

The output is merged later with Bambu early-HLS metrics using:
benchmark, dataset_size, clock_period, mem_policy, dsp_coeff, opt_level

Important:
    Bambu-generated Verilog is hierarchical. The top wrapper module may contain
    only one child instance, so this parser extracts both:
      1. selected top/implementation module metrics
      2. aggregate metrics summed across all modules

The aggregate metrics are often more useful as Yosys-level synthesis QoR.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/yosys_reports")
DEFAULT_OUTPUT = Path("data/extracted_metrics/yosys_qor_targets.csv")


TOP_MODULES = {
    "2mm": "kernel_2mm",
    "3mm": "kernel_3mm",
    "atax": "kernel_atax",
    "bicg": "kernel_bicg",
    "deriche": "kernel_deriche",
    "floyd-warshall": "kernel_floyd_warshall",
    "gemm": "kernel_gemm",
    "gemver": "kernel_gemver",
    "gesummv": "kernel_gesummv",
    "jacobi-1d": "kernel_jacobi_1d",
    "jacobi-2d": "kernel_jacobi_2d",
    "mvt": "kernel_mvt",
    "nussinov": "kernel_nussinov",
    "seidel-2d": "kernel_seidel_2d",
    "symm": "kernel_symm",
    "syr2k": "kernel_syr2k",
    "syrk": "kernel_syrk",
    "trmm": "kernel_trmm",
}


def sanitize_cell_type(cell_type: str) -> str:
    """Make a Yosys cell type safe for use as a CSV column name."""
    return (
        str(cell_type)
        .replace("$", "cell_")
        .replace("\\", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def parse_config_from_filename(path: Path) -> dict:
    """
    Parse filenames like:
    gemm_SMALL_clk10_memALL_BRAM_dsp1.0_optO2_yosys.json
    """
    stem = path.stem.replace("_yosys", "")

    pattern = (
        r"(?P<benchmark>.+)_"
        r"(?P<dataset_size>SMALL|MEDIUM|MINI|LARGE)_"
        r"clk(?P<clock_period>\d+)_"
        r"mem(?P<mem_policy>[A-Z_]+)_"
        r"dsp(?P<dsp_coeff>[\d.]+)_"
        r"opt(?P<opt_level>O\d+)"
    )

    match = re.match(pattern, stem)

    if not match:
        raise ValueError(f"Could not parse configuration from filename: {path.name}")

    config = match.groupdict()
    config["clock_period"] = int(config["clock_period"])
    config["dsp_coeff"] = float(config["dsp_coeff"])

    return config


def find_selected_module(data: dict, benchmark: str) -> tuple[str, dict]:
    """
    Select a useful module for reporting.

    Bambu often creates:
      \\kernel_gemm       -> thin wrapper
      \\_kernel_gemm      -> implementation wrapper
      \\controller_*      -> control FSM
      \\datapath_*        -> datapath

    Prefer \\_kernel_* when present because \\kernel_* can be only a wrapper.
    Fall back to largest module if expected names are not found.
    """
    modules = data.get("modules", {})

    if not modules:
        return "UNKNOWN", {}

    expected_top = TOP_MODULES.get(benchmark)

    if expected_top:
        candidates = [
            "\\_" + expected_top,
            "_" + expected_top,
            "\\" + expected_top,
            expected_top,
            "\\datapath_" + expected_top.replace("kernel_", ""),
            "datapath_" + expected_top.replace("kernel_", ""),
            "\\controller_" + expected_top.replace("kernel_", ""),
            "controller_" + expected_top.replace("kernel_", ""),
        ]

        for candidate in candidates:
            if candidate in modules:
                return candidate, modules[candidate]

    # Fallback: choose module with largest number of cells.
    best_name = None
    best_module = None
    best_cells = -1

    for name, module in modules.items():
        num_cells = module.get("num_cells", 0) or 0
        if num_cells > best_cells:
            best_cells = num_cells
            best_name = name
            best_module = module

    return best_name or "UNKNOWN", best_module or {}


def aggregate_design_metrics(data: dict) -> dict:
    """
    Aggregate metrics across all modules.

    This is useful because Bambu Verilog is hierarchical and a single top
    wrapper may not expose the full implementation complexity.
    """
    modules = data.get("modules", {})

    total_wires = 0
    total_wire_bits = 0
    total_pub_wires = 0
    total_pub_wire_bits = 0
    total_memories = 0
    total_memory_bits = 0
    total_processes = 0
    total_cells = 0
    total_cell_types = {}

    for module in modules.values():
        total_wires += module.get("num_wires", 0) or 0
        total_wire_bits += module.get("num_wire_bits", 0) or 0
        total_pub_wires += module.get("num_pub_wires", 0) or 0
        total_pub_wire_bits += module.get("num_pub_wire_bits", 0) or 0
        total_memories += module.get("num_memories", 0) or 0
        total_memory_bits += module.get("num_memory_bits", 0) or 0
        total_processes += module.get("num_processes", 0) or 0
        total_cells += module.get("num_cells", 0) or 0

        for cell_type, count in module.get("num_cells_by_type", {}).items():
            total_cell_types[cell_type] = total_cell_types.get(cell_type, 0) + count

    metrics = {
        "yosys_total_modules": len(modules),
        "yosys_total_wires": total_wires,
        "yosys_total_wire_bits": total_wire_bits,
        "yosys_total_pub_wires": total_pub_wires,
        "yosys_total_pub_wire_bits": total_pub_wire_bits,
        "yosys_total_memories": total_memories,
        "yosys_total_memory_bits": total_memory_bits,
        "yosys_total_processes": total_processes,
        "yosys_total_cells": total_cells,
    }
    """

    for cell_type, count in total_cell_types.items():
        safe_name = sanitize_cell_type(cell_type)
        metrics[f"yosys_total_celltype_{safe_name}"] = count
    """
    return metrics


def extract_yosys_metrics(json_path: Path) -> dict:
    config = parse_config_from_filename(json_path)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {
            **config,
            "yosys_parse_status": "failed",
            "yosys_error": str(exc),
            "yosys_report_file": json_path.name,
        }

    selected_module_name, selected_module = find_selected_module(data, config["benchmark"])
    aggregate_metrics = aggregate_design_metrics(data)

    metrics = {
        **config,
        "yosys_parse_status": "ok",
        "yosys_selected_module": selected_module_name,
        "yosys_num_wires": selected_module.get("num_wires"),
        "yosys_num_wire_bits": selected_module.get("num_wire_bits"),
        "yosys_num_pub_wires": selected_module.get("num_pub_wires"),
        "yosys_num_pub_wire_bits": selected_module.get("num_pub_wire_bits"),
        "yosys_num_memories": selected_module.get("num_memories"),
        "yosys_num_memory_bits": selected_module.get("num_memory_bits"),
        "yosys_num_processes": selected_module.get("num_processes"),
        "yosys_num_cells": selected_module.get("num_cells"),
        **aggregate_metrics,
        "yosys_report_file": json_path.name,
    }
    """
    # Selected-module cell type counts.
    for cell_type, count in selected_module.get("num_cells_by_type", {}).items():
        safe_name = sanitize_cell_type(cell_type)
        metrics[f"yosys_celltype_{safe_name}"] = count
    """
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Yosys JSON reports into QoR target CSV.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Yosys report directory not found: {args.input_dir}")

    json_files = sorted(args.input_dir.glob("*_yosys.json"))

    if not json_files:
        raise FileNotFoundError(f"No Yosys JSON reports found in: {args.input_dir}")

    rows = []

    for path in json_files:
        try:
            rows.append(extract_yosys_metrics(path))
        except Exception as exc:
            print(f"WARNING: failed to parse {path.name}: {exc}")

    df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n=== Yosys parsing complete ===")
    print(f"Input reports: {args.input_dir}")
    print(f"Reports found: {len(json_files)}")
    print(f"Rows written: {df.shape[0]}")
    print(f"Columns written: {df.shape[1]}")
    print(f"Output: {args.output}")

    if "yosys_parse_status" in df.columns:
        print("\nParse status:")
        print(df["yosys_parse_status"].value_counts())

    useful_cols = [
        "benchmark",
        "dataset_size",
        "clock_period",
        "mem_policy",
        "dsp_coeff",
        "opt_level",
        "yosys_selected_module",
        "yosys_num_cells",
        "yosys_total_cells",
        "yosys_total_wire_bits",
        "yosys_total_modules",
    ]
    existing = [c for c in useful_cols if c in df.columns]

    print("\nPreview:")
    print(df[existing].head())


if __name__ == "__main__":
    main()
