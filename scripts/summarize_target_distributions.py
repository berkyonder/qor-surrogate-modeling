#!/usr/bin/env python3
"""
summarize_target_distributions.py

Summarizes OpenROAD target distributions so MAE and relative MAE
can be interpreted clearly.
"""

from pathlib import Path
import pandas as pd

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUT_CSV = Path("reports/modeling/openroad_target_distribution_summary.csv")
OUT_MD = Path("reports/modeling/openroad_target_distribution_summary.md")

TARGETS = {
    "openroad_synth_chip_area": "Final ASIC area",
    "openroad_critical_path_delay": "Final critical path delay",
    "openroad_total_power": "Final total power",
}

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)

    rows = []
    lines = ["# OpenROAD Target Distribution Summary\n"]

    for col, label in TARGETS.items():
        y = df[col].dropna()

        row = {
            "target": col,
            "label": label,
            "count": len(y),
            "mean": y.mean(),
            "median": y.median(),
            "std": y.std(),
            "min": y.min(),
            "q25": y.quantile(0.25),
            "q75": y.quantile(0.75),
            "max": y.max(),
        }
        rows.append(row)

        lines.append(f"## {label}")
        lines.append(f"- Count: {row['count']}")
        lines.append(f"- Mean: {row['mean']:.6g}")
        lines.append(f"- Median: {row['median']:.6g}")
        lines.append(f"- Min: {row['min']:.6g}")
        lines.append(f"- Max: {row['max']:.6g}")
        lines.append("")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:")
    print(OUT_CSV)
    print(OUT_MD)

if __name__ == "__main__":
    main()