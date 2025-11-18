# Equity Data Exporter

A lightweight CLI tool that uses Yahoo Finance (via `yfinance`) to collect fundamentals for tickers and export them to Excel.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > If you are behind a proxy or in a restricted network, make sure `pip` can
   > reach PyPI; otherwise dependency installation will fail.

2. Create a text file containing one ticker per line, e.g.:
   ```text
   AAPL
   MSFT
   AMZN
   ```

## Usage

Run the tool by passing the input file and optional output path:

```bash
python main.py tickers.txt --output output.xlsx
```

Key details:
- Pulls financial statement data from Yahoo Finance where possible and tries common label variants.
- Fields that are not available via Yahoo Finance are marked as `"N/A (requires external source)"`.
- Logging level can be adjusted with `--log-level` (e.g., `DEBUG`).

The CLI first attempts to use `Ticker.get_info()` (recommended by `yfinance`) and
only falls back to the legacy `Ticker.info` property if necessary, which helps
keep the exporter compatible across `yfinance` releases.

The resulting Excel file contains one row per ticker with the requested metrics.
