# Equity Data Exporter

A lightweight CLI tool that uses Yahoo Finance (via `yfinance`) to collect fundamentals for tickers and export them to Excel, with an optional Alpha Vantage backup when Yahoo data is missing.

## Setup

1. Install dependencies (latest `yfinance` is required for `get_info` support):
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

Run the tool by passing the input file and optional output path. You can also
provide an Alpha Vantage API key (via `--alphavantage-api-key` or the
`ALPHAVANTAGE_API_KEY` environment variable) to backfill values when Yahoo
Finance does not provide them:

```bash
python main.py tickers.txt --output output.xlsx --alphavantage-api-key <YOUR_KEY>
```

Key details:
- Pulls financial statement data from Yahoo Finance where possible and tries common label variants.
- When Yahoo Finance data is unavailable, the tool will attempt to backfill key
  metrics (price, shares outstanding, cash, debt, income statement, and cash
  flow figures) from Alpha Vantage if an API key is supplied. Fields that are
  still unavailable remain blank or `"N/A (requires external source)"`.
- Logging level can be adjusted with `--log-level` (e.g., `DEBUG`).

The CLI first attempts to use `Ticker.get_info()` (recommended by `yfinance`) and
only falls back to the legacy `Ticker.info` property if necessary, which helps
keep the exporter compatible across `yfinance` releases.

## Testing and verification

The repository includes lightweight unit tests that mock both Yahoo Finance and
Alpha Vantage so you can verify the end-to-end flow (input tickers → data
collection → Excel output) without real network calls:

```bash
python -m unittest tests/test_pipeline.py
```

> These tests rely on the Python dependencies declared in `requirements.txt`.

**Limitations and reliability:** The exporter depends on upstream data providers.
If Yahoo Finance or Alpha Vantage change their APIs/fields or are unavailable,
the tool may return partial data. Some metrics (e.g., WACC, DCF per share,
insider activity, analyst ratings beyond Yahoo, and Altman Z) are not supplied
by either source and will remain `N/A`.

The resulting Excel file contains one row per ticker with the requested metrics.
