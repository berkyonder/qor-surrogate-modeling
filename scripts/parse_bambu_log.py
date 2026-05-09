#!/usr/bin/env python3
"""
parse_bambu_log.py

This script parses Bambu HLS tool output logs to extract QoR (Quality of Results) metrics
such as area, latency, and other synthesis parameters for surrogate modeling.

Usage:
    python parse_bambu_log.py <log_file_path> [--output <output_file>]

Arguments:
    log_file_path: Path to the Bambu log file to parse
    --output: Optional output file path (default: print to stdout)

Output:
    JSON format with extracted metrics
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

CSV_FIELDS = [
    "benchmark", "design_name", "control_steps", "min_slack", "frequency_mhz",
    "states", "modules_instantiated", "performance_conflicts", "flipflops",
    "area_est", "mux_area", "total_area", "registers", "dsps", "log_file"
]


def parse_bambu_log(log_file_path):
    """
    Parse a Bambu log file and extract key metrics.

    Args:
        log_file_path (str): Path to the log file

    Returns:
        dict: Dictionary containing extracted metrics
    """
    metrics = {}

    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.", file=sys.stderr)
        return metrics
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return metrics

    # Extract metrics for the top-level function (main kernel function)
    # Find the top function by looking for the last function analysis section
    # This avoids including helper functions like floating-point add/multiply

    # Control steps (renamed from latency_cycles) - take the last occurrence
    control_steps_matches = re.findall(r'Number of control steps:\s*(\d+)', content)
    if control_steps_matches:
        metrics['control_steps'] = int(control_steps_matches[-1])

    # Minimum slack - take the last occurrence
    slack_matches = re.findall(r'Minimum slack:\s*([\d.]+)', content)
    if slack_matches:
        metrics['min_slack'] = float(slack_matches[-1])

    # Frequency - take the last occurrence
    freq_matches = re.findall(r'Estimated max frequency \(MHz\):\s*([\d.]+)', content)
    if freq_matches:
        metrics['frequency_mhz'] = float(freq_matches[-1])

    # Number of states - take the last occurrence
    states_matches = re.findall(r'Number of states:\s*(\d+)', content)
    if states_matches:
        metrics['states'] = int(states_matches[-1])

    # Module instances - take the last occurrence
    modules_matches = re.findall(r'Number of modules instantiated:\s*(\d+)', content)
    if modules_matches:
        metrics['modules_instantiated'] = int(modules_matches[-1])

    # Performance conflicts - take the last occurrence
    conflict_matches = re.findall(r'Number of performance conflicts:\s*(\d+)', content)
    if conflict_matches:
        metrics['performance_conflicts'] = int(conflict_matches[-1])

    # Flipflops - take the last occurrence
    ff_matches = re.findall(r'Total number of flip-flops in function [^:]+:\s*(\d+)', content)
    if ff_matches:
        metrics['flipflops'] = int(ff_matches[-1])

    # Resource area without muxes (area_est) - take the last occurrence
    area_est_matches = re.findall(r'Estimated resources area.*:\s*(\d+)', content)
    if area_est_matches:
        metrics['area_est'] = int(area_est_matches[-1])

    # MUX area - take the last occurrence
    mux_matches = re.findall(r'Estimated area of MUX21:\s*(\d+)', content)
    if mux_matches:
        metrics['mux_area'] = int(mux_matches[-1])

    # Total area - take the last occurrence
    total_area_matches = re.findall(r'Total estimated area:\s*(\d+)', content)
    if total_area_matches:
        metrics['total_area'] = int(total_area_matches[-1])

    # Registers (from Register allocation algorithm) - take the last occurrence
    reg_matches = re.findall(r'Register allocation algorithm obtains a sub-optimal result:\s*(\d+)\s*registers', content)
    if reg_matches:
        metrics['registers'] = int(reg_matches[-1])

    # DSPs - take the last occurrence
    dsp_matches = re.findall(r'Estimated number of DSPs:\s*(\d+)', content)
    if dsp_matches:
        metrics['dsps'] = int(dsp_matches[-1])

    # Extract design name from the command line
    design_match = re.search(r'--top-fname=(\w+)', content)
    if design_match:
        metrics['design_name'] = design_match.group(1)

    # Add benchmark identifier derived from the log filename
    benchmark_name = Path(log_file_path).stem.replace('_log', '')
    metrics['benchmark'] = benchmark_name

    # Add metadata
    metrics['log_file'] = str(Path(log_file_path).name)

    return metrics


def append_to_csv(metrics, output_csv):
    """
    Append metrics to a CSV file.

    Args:
        metrics (dict): Dictionary containing extracted metrics
        output_csv (str): Path to the CSV file
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def main():
    parser = argparse.ArgumentParser(description='Parse Bambu HLS log files for QoR metrics')
    parser.add_argument('log_file', help='Path to the Bambu log file')
    parser.add_argument('--output', '-o', help='Output file path (default: stdout)')

    args = parser.parse_args()

    metrics = parse_bambu_log(args.log_file)

    if not metrics:
        print("No metrics extracted from the log file.", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(metrics, indent=2)

    if args.output:
        # If output is .csv → append to CSV
        if args.output.endswith('.csv'):
            append_to_csv(metrics, args.output)
            print(f"Metrics appended to {args.output}")
        else:
            # Otherwise write JSON
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()