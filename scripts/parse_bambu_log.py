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
import json
import re
import sys
from pathlib import Path


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

    # Extract metrics for the main function (typically the last/top-level function)
    # Based on Bambu output format, we look for the final results by finding all matches and taking the last one

    # Area metrics - take the last occurrence
    area_matches = re.findall(r'Total estimated area:\s*(\d+)', content)
    if area_matches:
        metrics['area_total'] = int(area_matches[-1])

    # Frequency - take the last occurrence
    freq_matches = re.findall(r'Estimated max frequency \(MHz\):\s*([\d.]+)', content)
    if freq_matches:
        metrics['frequency_mhz'] = float(freq_matches[-1])

    # Latency (number of control steps) - take the last occurrence
    latency_matches = re.findall(r'Number of control steps:\s*(\d+)', content)
    if latency_matches:
        metrics['latency_cycles'] = int(latency_matches[-1])

    # Flip-flops (registers) - take the last occurrence
    ff_matches = re.findall(r'Total number of flip-flops in function [^:]+:\s*(\d+)', content)
    if ff_matches:
        metrics['flip_flops'] = int(ff_matches[-1])

    # DSPs - take the last occurrence
    dsp_matches = re.findall(r'Estimated number of DSPs:\s*(\d+)', content)
    if dsp_matches:
        metrics['dsps'] = int(dsp_matches[-1])

    # Number of states - take the last occurrence
    states_matches = re.findall(r'Number of states:\s*(\d+)', content)
    if states_matches:
        metrics['states'] = int(states_matches[-1])

    # Module instances - take the last occurrence
    modules_matches = re.findall(r'Number of modules instantiated:\s*(\d+)', content)
    if modules_matches:
        metrics['modules_instantiated'] = int(modules_matches[-1])

    # Minimum slack - take the last occurrence
    slack_matches = re.findall(r'Minimum slack:\s*([\d.]+)', content)
    if slack_matches:
        metrics['min_slack'] = float(slack_matches[-1])

    # Extract design name from the command line
    design_match = re.search(r'--top-fname=(\w+)', content)
    if design_match:
        metrics['design_name'] = design_match.group(1)

    # Add metadata
    metrics['log_file'] = str(Path(log_file_path).name)

    return metrics


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
        try:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Metrics saved to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output)


if __name__ == '__main__':
    main()