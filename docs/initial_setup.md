# Initial Setup and Toolchain Description

This document describes the initial setup required to run the QoR surrogate modeling project and the toolchain components involved.

## Prerequisites

- Python 3.8 or higher
- Git
- Bambu HLS tool (for generating synthesis logs)
- Machine learning libraries (scikit-learn, pandas, numpy, etc.)

## Toolchain Overview

### 1. Bambu HLS
Bambu is an open-source High-Level Synthesis (HLS) tool that converts C/C++ code into RTL (Verilog/VHDL) for FPGA/ASIC implementation. In this project, Bambu is used to:

- Synthesize high-level C code into hardware
- Generate detailed synthesis reports with QoR metrics
- Provide early-stage area and latency estimates

**Installation:**
```bash
# Clone the PandA-Bambu repository
git clone https://github.com/ferrandi/PandA-bambu.git
cd PandA-bambu
# Follow the build instructions in the README
```

### 2. Python Environment
The project uses Python for data processing, machine learning model training, and analysis.

**Setup:**
```bash
# Create virtual environment
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

### 3. Data Pipeline
- **Raw Data Collection**: Bambu logs are collected from various HLS runs
- **Feature Extraction**: `scripts/parse_bambu_log.py` extracts QoR metrics
- **Model Training**: Machine learning models predict final QoR from early metrics

## Project Structure

```
qor-surrogate-modeling/
├── scripts/
│   └── parse_bambu_log.py    # Log parsing script
├── data/
│   ├── raw_reports/          # Bambu log files
│   └── extracted_metrics/    # Processed metrics
├── models/                   # Trained ML models
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## Quick Start

1. Set up the Python environment
2. Run Bambu on your C code to generate logs
3. Use `parse_bambu_log.py` to extract metrics
4. Train/predict with the ML pipeline

For detailed usage instructions, see the main README.md.