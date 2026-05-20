from pathlib import Path
import re
import pandas as pd


OPENROAD_DIR = Path(r"D:\qor_project\openroad_reports_to_copy")
OUTPUT_CSV = Path("data/extracted_metrics/openroad_qor_targets.csv")


def extract_first_float(pattern: str, text: str):
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1))


def extract_first_int(pattern: str, text: str):
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def parse_config_name(config_name: str) -> dict:
    """
    Parse names like:
    gemm_SMALL_clk10_memALL_BRAM_dsp1.0_optO0_orclk2
    """
    pattern = (
        r"(?P<benchmark>.+)_"
        r"(?P<dataset_size>SMALL|MEDIUM|MINI|LARGE)_"
        r"clk(?P<clock_period>\d+)_"
        r"mem(?P<mem_policy>[A-Z_]+)_"
        r"dsp(?P<dsp_coeff>[\d.]+)_"
        r"opt(?P<opt_level>O\d+)_"
        r"orclk(?P<openroad_clock_constraint>[\d.]+)"
    )

    m = re.match(pattern, config_name)

    if not m:
        return {
            "benchmark": None,
            "dataset_size": None,
            "clock_period": None,
            "mem_policy": None,
            "dsp_coeff": None,
            "opt_level": None,
            "openroad_clock_constraint": None,
            "config_parse_status": "failed",
        }

    data = m.groupdict()
    data["clock_period"] = int(data["clock_period"])
    data["dsp_coeff"] = float(data["dsp_coeff"])
    data["openroad_clock_constraint"] = float(data["openroad_clock_constraint"])
    data["config_parse_status"] = "ok"

    return data


def parse_global_route_report(report_path: Path) -> dict:
    text = report_path.read_text(errors="ignore")

    data = {}

    data["openroad_tns"] = extract_first_float(
        r"tns max ([\-0-9\.]+)", text
    )

    data["openroad_wns"] = extract_first_float(
        r"wns max ([\-0-9\.]+)", text
    )

    data["openroad_worst_slack"] = extract_first_float(
        r"worst slack max ([\-0-9\.]+)", text
    )

    data["openroad_clock_period_min"] = extract_first_float(
        r"clock period_min = ([0-9\.]+)", text
    )

    data["openroad_fmax"] = extract_first_float(
        r"fmax = ([0-9\.]+)", text
    )

    data["openroad_critical_path_delay"] = extract_first_float(
        r"critical path delay\s*\n[-]+\s*\n([0-9\.]+)",
        text,
    )

    data["openroad_critical_path_slack"] = extract_first_float(
        r"critical path slack\s*\n[-]+\s*\n([\-0-9\.]+)",
        text,
    )

    data["openroad_setup_violation_count"] = extract_first_int(
        r"setup violation count ([0-9]+)",
        text,
    )

    data["openroad_hold_violation_count"] = extract_first_int(
        r"hold violation count ([0-9]+)",
        text,
    )

    data["openroad_max_slew_violation_count"] = extract_first_int(
        r"max slew violation count ([0-9]+)",
        text,
    )

    data["openroad_max_cap_violation_count"] = extract_first_int(
        r"max cap violation count ([0-9]+)",
        text,
    )

    data["openroad_total_power"] = extract_first_float(
        r"Total\s+[0-9eE\-\+\.]+\s+[0-9eE\-\+\.]+\s+[0-9eE\-\+\.]+\s+([0-9eE\-\+\.]+)",
        text,
    )

    return data


def parse_synth_stat(report_path: Path) -> dict:
    text = report_path.read_text(errors="ignore")

    return {
        "openroad_synth_chip_area": extract_first_float(
            r"Chip area for module .*?:\s*([0-9\.]+)",
            text,
        )
    }


def find_report(run_dir: Path, filename: str):
    candidates = [
        run_dir / "reports" / filename,
        run_dir / "reports" / "base" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def parse_run_folder(run_dir: Path) -> dict:
    result = {
        "config_name": run_dir.name,
        **parse_config_name(run_dir.name),
    }

    status_file = run_dir / "status.txt"
    if status_file.exists():
        result["openroad_run_status"] = status_file.read_text(errors="ignore").strip()
    else:
        result["openroad_run_status"] = "unknown"

    route_report = find_report(run_dir, "5_global_route.rpt")

    if route_report is None:
        result["openroad_parse_status"] = "missing_global_route_report"
        return result

    try:
        result.update(parse_global_route_report(route_report))

        synth_stat = find_report(run_dir, "synth_stat.txt")
        if synth_stat is not None:
            result.update(parse_synth_stat(synth_stat))
        else:
            result["openroad_synth_chip_area"] = None

        result["openroad_parse_status"] = "ok"

    except Exception as exc:
        result["openroad_parse_status"] = f"parse_error: {exc}"

    return result


def main() -> None:
    if not OPENROAD_DIR.exists():
        raise FileNotFoundError(f"OpenROAD report directory not found: {OPENROAD_DIR}")

    rows = []

    for run_dir in sorted(OPENROAD_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        rows.append(parse_run_folder(run_dir))

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== OpenROAD parsing complete ===")
    print(f"Input directory: {OPENROAD_DIR}")
    print(f"Rows written: {len(df)}")
    print(f"Output: {OUTPUT_CSV}")

    if not df.empty:
        print("\nParse status:")
        print(df["openroad_parse_status"].value_counts(dropna=False))

        preview_cols = [
            "config_name",
            "benchmark",
            "dataset_size",
            "clock_period",
            "mem_policy",
            "dsp_coeff",
            "opt_level",
            "openroad_clock_constraint",
            "openroad_synth_chip_area",
            "openroad_critical_path_delay",
            "openroad_total_power",
            "openroad_parse_status",
        ]

        existing = [c for c in preview_cols if c in df.columns]

        print("\nPreview:")
        print(df[existing].head(20))


if __name__ == "__main__":
    main()
