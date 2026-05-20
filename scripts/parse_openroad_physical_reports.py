#!/usr/bin/env python3
"""
parse_openroad_physical_reports.py

Extracts additional physical-design effects from copied OpenROAD reports:
- Route DRC report indicators from 5_route_drc.rpt
- Approximate routed wirelength from 6_final.def
- Route guide line/segment counts from route.guide

Input:
- D:/qor_project/openroad_reports_to_copy/<config_name>/

Output:
- data/extracted_metrics/openroad_physical_metrics.csv
- reports/modeling/openroad_wirelength_drc_summary.csv
- reports/modeling/openroad_wirelength_drc_summary.md
"""

from pathlib import Path
import re
import pandas as pd


OPENROAD_DIR = Path(r"D:\qor_project\openroad_reports_to_copy")
OUTPUT_CSV = Path("data/extracted_metrics/openroad_physical_metrics.csv")
SUMMARY_CSV = Path("reports/modeling/openroad_wirelength_drc_summary.csv")
SUMMARY_MD = Path("reports/modeling/openroad_wirelength_drc_summary.md")


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path and path.exists() else ""


def find_file(run_dir: Path, filename: str):
    candidates = [
        run_dir / filename,
        run_dir / "reports" / filename,
        run_dir / "results" / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(run_dir.rglob(filename))
    return matches[0] if matches else None


def find_first_matching(run_dir: Path, pattern: str):
    matches = list(run_dir.rglob(pattern))
    return matches[0] if matches else None


def parse_drc_report(run_dir: Path) -> dict:
    """
    Parses route DRC reports.

    Since OpenROAD report formatting may vary by command/version,
    this parser records conservative indicators:
    - whether a DRC report exists
    - non-empty line count
    - keyword-based violation count
    - whether the report appears clean
    """
    drc_files = sorted(run_dir.rglob("5_route_drc.rpt*"))

    result = {
        "openroad_route_drc_report_count": len(drc_files),
        "openroad_route_drc_nonempty_lines": 0,
        "openroad_route_drc_violation_keyword_count": 0,
        "openroad_route_drc_has_report": int(len(drc_files) > 0),
        "openroad_route_drc_status": "missing",
    }

    if not drc_files:
        return result

    total_nonempty = 0
    total_keyword_hits = 0
    all_text = []

    for path in drc_files:
        text = read_text(path)
        all_text.append(text)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        total_nonempty += len(lines)

        # Count likely violation records conservatively.
        # This intentionally avoids counting the filename itself.
        for ln in lines:
            lower = ln.lower()
            if (
                "violation" in lower
                or "drc" in lower and ("error" in lower or "viol" in lower)
                or re.search(r"\bmet[0-9]+\b", lower)
            ):
                total_keyword_hits += 1

    joined = "\n".join(all_text).lower()

    result["openroad_route_drc_nonempty_lines"] = total_nonempty
    result["openroad_route_drc_violation_keyword_count"] = total_keyword_hits

    if total_nonempty == 0:
        result["openroad_route_drc_status"] = "empty_or_clean"
    elif "no drc violations" in joined or "number of violations: 0" in joined:
        result["openroad_route_drc_status"] = "clean"
    elif total_keyword_hits == 0:
        result["openroad_route_drc_status"] = "no_violation_keywords"
    else:
        result["openroad_route_drc_status"] = "possible_violations"

    return result


def parse_def_units(text: str):
    m = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", text)
    if not m:
        return None
    return int(m.group(1))


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def parse_def_wirelength(def_path: Path) -> dict:
    """
    Approximate routed wirelength from DEF.

    This parser scans coordinate pairs in NETS/SPECIALNETS routing statements
    and sums Manhattan distance between consecutive coordinate points.
    It is an approximation, but useful as a physical-design effect indicator.
    """
    result = {
        "openroad_def_exists": 0,
        "openroad_def_units_per_micron": None,
        "openroad_def_routed_wirelength_dbu": None,
        "openroad_def_routed_wirelength_microns": None,
        "openroad_def_coordinate_pair_count": 0,
        "openroad_def_routed_net_count": 0,
    }

    if def_path is None or not def_path.exists():
        return result

    text = read_text(def_path)
    result["openroad_def_exists"] = 1

    units = parse_def_units(text)
    result["openroad_def_units_per_micron"] = units

    # Extract NETS and SPECIALNETS blocks if present.
    blocks = []
    for section in ["SPECIALNETS", "NETS"]:
        m = re.search(rf"{section}\s+\d+\s*;(.*?END\s+{section})", text, flags=re.S)
        if m:
            blocks.append(m.group(1))

    if not blocks:
        blocks = [text]

    total_wl = 0
    total_pairs = 0
    routed_net_count = 0

    coord_re = re.compile(r"\(\s*([\-0-9*]+)\s+([\-0-9*]+)\s*\)")

    for block in blocks:
        # Split roughly by DEF net records.
        records = re.split(r"\n\s*-\s+", block)

        for rec in records:
            if "ROUTED" not in rec and "FIXED" not in rec and "COVER" not in rec:
                continue

            routed_net_count += 1

            prev = None
            last_x = None
            last_y = None

            for m in coord_re.finditer(rec):
                x_raw, y_raw = m.group(1), m.group(2)

                if x_raw == "*":
                    x = last_x
                else:
                    x = int(x_raw)

                if y_raw == "*":
                    y = last_y
                else:
                    y = int(y_raw)

                if x is None or y is None:
                    continue

                point = (x, y)
                total_pairs += 1

                if prev is not None:
                    total_wl += manhattan_distance(prev, point)

                prev = point
                last_x, last_y = x, y

    result["openroad_def_routed_wirelength_dbu"] = total_wl
    result["openroad_def_coordinate_pair_count"] = total_pairs
    result["openroad_def_routed_net_count"] = routed_net_count

    if units and units != 0:
        result["openroad_def_routed_wirelength_microns"] = total_wl / units

    return result


def parse_route_guide(route_guide_path: Path) -> dict:
    result = {
        "openroad_route_guide_exists": 0,
        "openroad_route_guide_nonempty_lines": None,
        "openroad_route_guide_segment_like_lines": None,
    }

    if route_guide_path is None or not route_guide_path.exists():
        return result

    text = read_text(route_guide_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    segment_like = 0
    for ln in lines:
        # Route guide lines often contain layer + coordinate boxes.
        if re.search(r"\bmet[0-9]+\b", ln.lower()) or len(re.findall(r"\d+", ln)) >= 4:
            segment_like += 1

    result["openroad_route_guide_exists"] = 1
    result["openroad_route_guide_nonempty_lines"] = len(lines)
    result["openroad_route_guide_segment_like_lines"] = segment_like

    return result


def parse_run(run_dir: Path) -> dict:
    row = {
        "config_name": run_dir.name,
    }

    row.update(parse_drc_report(run_dir))

    def_path = find_first_matching(run_dir, "6_final.def")
    row.update(parse_def_wirelength(def_path))

    guide_path = find_first_matching(run_dir, "route.guide")
    row.update(parse_route_guide(guide_path))

    return row


def write_summary(df: pd.DataFrame):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        c for c in df.columns
        if c != "config_name" and pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = df[numeric_cols].describe().T
    summary.to_csv(SUMMARY_CSV)

    lines = []
    lines.append("# OpenROAD Wirelength and DRC Physical Effects Summary\n")
    lines.append(f"- Parsed OpenROAD runs: {len(df)}")
    lines.append(f"- Runs with DEF: {int(df['openroad_def_exists'].fillna(0).sum()) if 'openroad_def_exists' in df else 0}")
    lines.append(f"- Runs with route DRC report: {int(df['openroad_route_drc_has_report'].fillna(0).sum()) if 'openroad_route_drc_has_report' in df else 0}")
    lines.append(f"- Runs with route guide: {int(df['openroad_route_guide_exists'].fillna(0).sum()) if 'openroad_route_guide_exists' in df else 0}")
    lines.append("")

    if "openroad_def_routed_wirelength_microns" in df.columns:
        y = df["openroad_def_routed_wirelength_microns"].dropna()
        if not y.empty:
            lines.append("## Approximate routed wirelength from final DEF")
            lines.append(f"- Mean: {y.mean():.4f} microns")
            lines.append(f"- Median: {y.median():.4f} microns")
            lines.append(f"- Min: {y.min():.4f} microns")
            lines.append(f"- Max: {y.max():.4f} microns")
            lines.append("")

    if "openroad_route_drc_status" in df.columns:
        lines.append("## Route DRC report status")
        counts = df["openroad_route_drc_status"].value_counts(dropna=False)
        for status, count in counts.items():
            lines.append(f"- {status}: {count}")

        lines.append("")

    if "openroad_route_drc_violation_keyword_count" in df.columns:
        y = df["openroad_route_drc_violation_keyword_count"].dropna()
        lines.append("## DRC keyword indicators")
        lines.append(f"- Total keyword hits: {int(y.sum())}")
        lines.append(f"- Max keyword hits per design: {int(y.max()) if not y.empty else 0}")
        lines.append("")

    lines.append("## Note")
    lines.append(
        "Routed wirelength is approximated from coordinate sequences in the final DEF. "
        "The DRC report parser records conservative report indicators because exact "
        "OpenROAD DRC report formatting can vary. These metrics are intended as "
        "physical-effect descriptors rather than primary ML targets."
    )

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not OPENROAD_DIR.exists():
        raise FileNotFoundError(f"OpenROAD directory not found: {OPENROAD_DIR}")

    rows = []
    for run_dir in sorted(OPENROAD_DIR.iterdir()):
        if run_dir.is_dir():
            rows.append(parse_run(run_dir))

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    write_summary(df)

    print("\n=== OpenROAD physical report parsing complete ===")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary MD: {SUMMARY_MD}")

    preview_cols = [
        "config_name",
        "openroad_def_routed_wirelength_microns",
        "openroad_route_drc_status",
        "openroad_route_drc_violation_keyword_count",
        "openroad_route_guide_segment_like_lines",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("\nPreview:")
    print(df[preview_cols].head(20))


if __name__ == "__main__":
    main()