#!/usr/bin/env python3
"""
summarize_openroad_physical_effects.py

Summarizes physical implementation effects parsed from OpenROAD reports:
- WNS/TNS/worst slack
- setup/hold violation counts
- max slew/capacitance violation counts
- clock/fmax-related metrics

This directly supports the advisor request to inspect physical effects.
"""

from pathlib import Path
import pandas as pd

INPUT = Path("data/extracted_metrics/openroad_qor_targets.csv")
OUT_CSV = Path("reports/modeling/openroad_physical_effects_summary.csv")
OUT_MD = Path("reports/modeling/openroad_physical_effects_summary.md")

PHYSICAL_COLUMNS = [
    "openroad_tns",
    "openroad_wns",
    "openroad_worst_slack",
    "openroad_clock_period_min",
    "openroad_fmax",
    "openroad_critical_path_slack",
    "openroad_setup_violation_count",
    "openroad_hold_violation_count",
    "openroad_max_slew_violation_count",
    "openroad_max_cap_violation_count",
]

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    existing = [c for c in PHYSICAL_COLUMNS if c in df.columns]
    summary = df[existing].describe().T
    summary.to_csv(OUT_CSV)

    lines = ["# OpenROAD Physical Effects Summary\n"]
    lines.append(f"- Parsed OpenROAD runs: {len(df)}")
    lines.append(f"- Successfully parsed runs: {(df['openroad_parse_status'] == 'ok').sum() if 'openroad_parse_status' in df.columns else 'N/A'}")
    lines.append("")

    lines.append("## Physical/timing closure metrics inspected")
    for c in existing:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Violation counts")
    for c in [
        "openroad_setup_violation_count",
        "openroad_hold_violation_count",
        "openroad_max_slew_violation_count",
        "openroad_max_cap_violation_count",
    ]:
        if c in df.columns:
            lines.append(f"- `{c}` total: {df[c].fillna(0).sum():.0f}, max per design: {df[c].fillna(0).max():.0f}")

    lines.append("")
    lines.append("## Note")
    lines.append(
        "The current parser captures OpenROAD timing/electrical closure indicators "
        "including WNS, TNS, worst slack, setup/hold violations, and slew/capacitance violations. "
        "Explicit routed wirelength/congestion extraction can be added as future work from DEF/routing reports."
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:")
    print(OUT_CSV)
    print(OUT_MD)

if __name__ == "__main__":
    main()