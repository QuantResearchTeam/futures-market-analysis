## Quick Start Guide

### 1. Clone Repository

```bash
git clone https://github.com/QuantResearchTeam/futures-market-analysis.git
cd futures-market-analysis
```

### 2. Set Up Data (50GB)

This project uses a large dataset not tracked in Git. The data is organized by market:

- Download the required market data files from our shared team storage and unzip.
- Place files in the appropriate `/data` subdirectories maintaining this structure:
- you shoul dend up with following structure of data directory
```
data/
├── FTSE_2024_data_parquet/    # FTSE market data
├── NASDAQ_2024_data_parquet/  # NASDAQ market data
├── SANDP_2024_data_parquet/   # S&P market data
└── futures_data_local/        # Additional futures data
```


### 3. Set Up Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify Setup

```bash
# Test your environment
python -c "import pandas as pd; import numpy as np; print('Setup successful!')"

# Ensure data directories exist with the correct structure
ls -la data/
```

## Project Structure

```
futures-market-analysis/
├── README.md           # This file
├── data/               # Market data files (50GB, not in Git)
│   ├── FTSE_2024_data_parquet/
│   ├── NASDAQ_2024_data_parquet/
│   ├── SANDP_2024_data_parquet/
│   └── futures_data_local/
├── notebooks/          # Jupyter notebooks for analysis
│   └── Hedge_matching.ipynb
├── requirements.txt    # Python dependencies
├── scripts/            # Utility scripts
├── src/                # Source code
└── tests/              # Unit tests
```

## Running Analysis

Start Jupyter to explore the notebooks:

```bash
jupyter lab
# or
jupyter notebook
```

## Troubleshooting

- **Environment issues**: Make sure you've activated the virtual environment
- **Missing data**: Verify data files are in the correct location
- **Package errors**: Try `pip install -r requirements.txt --upgrade`
