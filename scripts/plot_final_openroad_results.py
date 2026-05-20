#!/usr/bin/env python3
"""
plot_final_openroad_results.py

Creates final target distribution plots for the OpenROAD QoR dataset.
These plots help interpret MAE and relative MAE.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATASET = Path("data/processed/openroad_modeling_dataset.csv")
OUT_DIR = Path("reports/modeling/final_plots")

TARGETS = {
    "openroad_synth_chip_area": "Final ASIC area",
    "openroad_critical_path_delay": "Final critical path delay (ns)",
    "openroad_total_power": "Final total power (W)",
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)

    for col, label in TARGETS.items():
        y = df[col].dropna()

        plt.figure()
        plt.hist(y, bins=min(10, max(3, len(y) // 2)))
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.title(f"Distribution of {label}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{col}_histogram.png", dpi=200)
        plt.close()

        plt.figure()
        plt.boxplot(y, vert=True)
        plt.ylabel(label)
        plt.title(f"Boxplot of {label}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{col}_boxplot.png", dpi=200)
        plt.close()

    print(f"Wrote plots to: {OUT_DIR}")

if __name__ == "__main__":
    main()