# Equity Data Exporter

A lightweight CLI tool that uses Yahoo Finance (via `yfinance`) to collect fundamentals for tickers and export them to Excel, with optional Alpha Vantage and FinancialModelingPrep backups when Yahoo data is missing.

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
provide Alpha Vantage and/or FinancialModelingPrep API keys (via
`--alphavantage-api-key` / `ALPHAVANTAGE_API_KEY` and `--fmp-api-key` /
`FMP_API_KEY`) to backfill values when Yahoo Finance does not provide them:

```bash
python main.py tickers.txt --output output.xlsx --alphavantage-api-key <YOUR_KEY> --fmp-api-key <YOUR_FMP_KEY>
```

Key details:
- Pulls financial statement data from Yahoo Finance where possible and tries common label variants.
- When Yahoo Finance data is unavailable, the tool will attempt to backfill key
  metrics (price, shares outstanding, cash, debt, income statement, and cash
  flow figures) from Alpha Vantage if an API key is supplied. A tertiary
  fallback to FinancialModelingPrep (FMP) fills cash flow, current asset/
  liability, debt, beta, and WACC metrics when an `FMP_API_KEY` is provided.
  Fields that are still unavailable remain blank.
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
If Yahoo Finance, Alpha Vantage, or FMP change their APIs/fields or are
unavailable, the tool may return partial data. Some metrics (e.g., DCF per
share, insider activity, analyst ratings beyond Yahoo, and Altman Z) are not
supplied by any of the sources and will remain blank. WACC is filled from FMP
when possible or estimated via CAPM using Beta, capital structure, and
interest expense when statement data is available.

The resulting Excel file contains one row per ticker with the requested metrics.
