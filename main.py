import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf


# Configure pandas display options to avoid SettingWithCopy warnings in logs
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Lightweight Alpha Vantage helper for backup data retrieval.

    The client only issues requests when an API key is available and will
    gracefully return ``None`` when requests fail or the service responds
    with an error.
    """

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, function: str, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        try:
            resp = self.session.get(
                "https://www.alphavantage.co/query",
                params={"function": function, "symbol": symbol, "apikey": self.api_key},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            # Alpha Vantage signals throttling with a specific note key
            if "Note" in payload or "Error Message" in payload:
                logger.debug("Alpha Vantage error for %s (%s): %s", symbol, function, payload)
                return None
            return payload
        except Exception as exc:  # network errors, JSON errors, etc.
            logger.debug("Alpha Vantage request failed for %s (%s): %s", symbol, function, exc)
            return None

    def get_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get("OVERVIEW", symbol)

    def get_income_statement(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        payload = self._get("INCOME_STATEMENT", symbol)
        if payload is None:
            return None
        return payload.get("annualReports")

    def get_balance_sheet(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        payload = self._get("BALANCE_SHEET", symbol)
        if payload is None:
            return None
        return payload.get("annualReports")

    def get_cash_flow(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        payload = self._get("CASH_FLOW", symbol)
        if payload is None:
            return None
        return payload.get("annualReports")

    def get_price(self, symbol: str) -> Optional[float]:
        payload = self._get("TIME_SERIES_DAILY_ADJUSTED", symbol)
        if payload is None:
            return None
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            return None
        latest_key = sorted(series.keys())[-1]
        close = series.get(latest_key, {}).get("4. close")
        try:
            return float(close)
        except (TypeError, ValueError):
            return None


FIELDS = [
    "Price",
    "Shares_Out (M)",
    "Total_Debt",
    "Cash",
    "EBITDA",
    "EBIT",
    "Net_Income",
    "Revenue",
    "COGS",
    "Total_Equity",
    "Total_Assets",
    "OCF",
    "FCF",
    "Dividends_Paid",
    "Interest_Expense",
    "Current_Assets",
    "Current_Liabilities",
    "Receivables",
    "Inventory",
    "WACC",
    "Tax_Rate",
    "DCF_Value_per_Share",
    "Insider_Net_Buys",
    "Institutional_%",
    "Fund_Flows_3M (M)",
    "Short_Interest_%",
    "Analyst_Rating(1-5)",
    "Beta",
    "EPS_ttm",
    "EPS_CAGR_3Y",
    "Revenue_CAGR_3Y",
    "FCF_CAGR_3Y",
    "Dividend_CAGR_3Y",
    "Altman_Z",
]


def _read_tickers(path: Path) -> List[str]:
    tickers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not tickers:
        raise ValueError("Input file must contain at least one ticker")
    return tickers


def _latest_column(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    ordered = df.sort_index(axis=1, ascending=False)
    return ordered.iloc[:, 0]


def _extract_field(df: pd.DataFrame, names: Iterable[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    series = _latest_column(df)
    lowered_index = {str(idx).lower(): idx for idx in series.index}
    for name in names:
        key = name.lower()
        if key in lowered_index:
            raw_value = series[lowered_index[key]]
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None
    return None


def _safe_info_lookup(info: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in info and info[key] is not None:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                return None
    return None


def _calculate_cagr(values: List[float], years_between: int) -> Optional[float]:
    usable = [v for v in values if v is not None]
    if len(usable) < 2 or years_between <= 0:
        return None
    start, end = usable[-1], usable[0]
    if start <= 0 or end <= 0:
        return None
    try:
        periods = years_between
        return (end / start) ** (1 / periods) - 1
    except ZeroDivisionError:
        return None


def _annual_series(df: pd.DataFrame, name_candidates: Iterable[str]) -> List[float]:
    if df is None or df.empty:
        return []
    ordered = df.sort_index(axis=1, ascending=False)
    result: List[float] = []
    lowered_index = {str(idx).lower(): idx for idx in ordered.index}
    match_idx = None
    for cand in name_candidates:
        if cand.lower() in lowered_index:
            match_idx = lowered_index[cand.lower()]
            break
    if match_idx is None:
        return result
    series = ordered.loc[match_idx]
    for value in series:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(None)
    return result


def _av_number(raw: Any) -> Optional[float]:
    try:
        if raw in (None, "", "None"):
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _av_latest(reports: Optional[List[Dict[str, Any]]], field: str) -> Optional[float]:
    if not reports:
        return None
    sorted_reports = sorted(
        reports,
        key=lambda r: r.get("fiscalDateEnding", ""),
        reverse=True,
    )
    for report in sorted_reports:
        value = _av_number(report.get(field))
        if value is not None:
            return value
    return None


def _free_cash_flow(cashflow_df: pd.DataFrame) -> Optional[float]:
    operating = _extract_field(
        cashflow_df,
        [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        ],
    )
    capex = _extract_field(cashflow_df, ["Capital Expenditures"])
    if operating is None or capex is None:
        return None
    return operating + capex  # CapEx is usually negative in statements


def _tax_rate(income_df: pd.DataFrame) -> Optional[float]:
    income_tax = _extract_field(income_df, ["Income Tax Expense", "Provision For Income Taxes"])
    pretax_income = _extract_field(income_df, ["Income Before Tax", "Pretax Income"])
    if income_tax is None or pretax_income is None or pretax_income == 0:
        return None
    return income_tax / pretax_income


def _revenue_cagr(income_df: pd.DataFrame) -> Optional[float]:
    revenues = _annual_series(income_df, ["Total Revenue", "Revenue"])
    if len(revenues) < 4:
        return None
    return _calculate_cagr(revenues[:4], 3)


def _fcf_cagr(cashflow_df: pd.DataFrame) -> Optional[float]:
    fcf_series = []
    if cashflow_df is None or cashflow_df.empty:
        return None
    ordered = cashflow_df.sort_index(axis=1, ascending=False)
    for _, column in enumerate(ordered.columns):
        operating = _extract_field(ordered[[column]], ["Total Cash From Operating Activities", "Operating Cash Flow"])
        capex = _extract_field(ordered[[column]], ["Capital Expenditures"])
        if operating is None or capex is None:
            fcf_series.append(None)
        else:
            fcf_series.append(operating + capex)
    if len(fcf_series) < 4:
        return None
    return _calculate_cagr(fcf_series[:4], 3)


def _dividend_cagr(ticker: yf.Ticker) -> Optional[float]:
    try:
        dividends = ticker.dividends
    except Exception as exc:  # network errors etc.
        logger.debug("Unable to load dividends: %s", exc)
        return None
    if dividends is None or dividends.empty:
        return None
    annual = dividends.groupby(dividends.index.year).sum().sort_index(ascending=False)
    if len(annual) < 4:
        return None
    recent = annual.iloc[0]
    oldest = annual.iloc[3]
    if oldest <= 0 or recent <= 0:
        return None
    return (recent / oldest) ** (1 / 3) - 1


def _augment_with_alpha_vantage(
    av_client: AlphaVantageClient, ticker_symbol: str, data: Dict[str, Any]
) -> None:
    """Fill missing values using Alpha Vantage as a backup source.

    Only fields absent from Yahoo Finance are overwritten, preserving Yahoo
    values when available.
    """

    overview = av_client.get_overview(ticker_symbol)
    income_reports = av_client.get_income_statement(ticker_symbol)
    balance_reports = av_client.get_balance_sheet(ticker_symbol)
    cashflow_reports = av_client.get_cash_flow(ticker_symbol)

    if data.get("Price") is None:
        data["Price"] = av_client.get_price(ticker_symbol)

    if data.get("Shares_Out (M)") is None:
        shares = _av_number(overview.get("SharesOutstanding")) if overview else None
        data["Shares_Out (M)"] = shares / 1_000_000 if shares else None

    if data.get("Beta") is None and overview:
        data["Beta"] = _av_number(overview.get("Beta"))

    if data.get("EPS_ttm") is None and overview:
        data["EPS_ttm"] = _av_number(overview.get("EPS"))

    if data.get("EBITDA") is None:
        ebitda = _av_number(overview.get("EBITDA")) if overview else None
        if ebitda is None:
            ebitda = _av_latest(income_reports, "ebitda")
        data["EBITDA"] = ebitda

    if data.get("Net_Income") is None:
        data["Net_Income"] = _av_latest(income_reports, "netIncome")

    if data.get("EBIT") is None:
        data["EBIT"] = _av_latest(income_reports, "ebit")

    if data.get("Revenue") is None:
        data["Revenue"] = _av_latest(income_reports, "totalRevenue")

    if data.get("COGS") is None:
        data["COGS"] = _av_latest(income_reports, "costOfRevenue")

    if data.get("Total_Assets") is None:
        data["Total_Assets"] = _av_latest(balance_reports, "totalAssets")

    if data.get("Total_Equity") is None:
        data["Total_Equity"] = _av_latest(
            balance_reports, "totalShareholderEquity"
        )

    if data.get("Cash") is None:
        cash = _av_latest(balance_reports, "cashAndCashEquivalentsAtCarryingValue")
        if cash is None:
            cash = _av_latest(balance_reports, "cashAndShortTermInvestments")
        data["Cash"] = cash

    if data.get("Total_Debt") is None:
        total_debt = _av_latest(balance_reports, "shortLongTermDebtTotal")
        if total_debt is None:
            total_debt = _av_latest(balance_reports, "longTermDebt")
        data["Total_Debt"] = total_debt

    if data.get("Current_Assets") is None:
        data["Current_Assets"] = _av_latest(balance_reports, "totalCurrentAssets")

    if data.get("Current_Liabilities") is None:
        data["Current_Liabilities"] = _av_latest(
            balance_reports, "totalCurrentLiabilities"
        )

    if data.get("Receivables") is None:
        data["Receivables"] = _av_latest(balance_reports, "currentNetReceivables")

    if data.get("Inventory") is None:
        data["Inventory"] = _av_latest(balance_reports, "inventory")

    if data.get("OCF") is None:
        data["OCF"] = _av_latest(cashflow_reports, "operatingCashflow")

    if data.get("FCF") is None:
        op_cf = _av_latest(cashflow_reports, "operatingCashflow")
        capex = _av_latest(cashflow_reports, "capitalExpenditures")
        if op_cf is not None and capex is not None:
            data["FCF"] = op_cf + capex

    if data.get("Dividends_Paid") is None:
        data["Dividends_Paid"] = _av_latest(cashflow_reports, "dividendPayout")

    if data.get("Interest_Expense") is None:
        data["Interest_Expense"] = _av_latest(income_reports, "interestExpense")


def _shares_outstanding(info: Dict[str, Any], ticker: yf.Ticker) -> Optional[float]:
    shares = _safe_info_lookup(info, ["sharesOutstanding", "floatShares", "shares_outstanding"])
    if shares:
        return shares / 1_000_000
    try:
        full = ticker.get_shares_full()
        if full is not None and not full.empty:
            latest = full.sort_index().iloc[-1]
            return float(latest) / 1_000_000
    except Exception as exc:
        logger.debug("Unable to fetch full share history: %s", exc)
    return None


def _price(info: Dict[str, Any], ticker: yf.Ticker) -> Optional[float]:
    price = _safe_info_lookup(info, ["currentPrice", "regularMarketPrice", "lastPrice"])
    if price is not None:
        return price

    try:
        fast = getattr(ticker, "fast_info", None)
        if fast is not None:
            if isinstance(fast, dict) and "last_price" in fast:
                return float(fast["last_price"])
            for key in ("last_price", "lastPrice"):
                value = getattr(fast, key, None)
                if value is not None:
                    return float(value)
    except Exception as exc:
        logger.debug("Unable to load fast price: %s", exc)

    try:
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("Unable to load price history: %s", exc)
    return None


def _get_info_dict(ticker: yf.Ticker) -> Dict[str, Any]:
    """Obtain the info dictionary using the most reliable available endpoint.

    Newer versions of ``yfinance`` recommend :py:meth:`Ticker.get_info` instead of
    the ``info`` property. This helper tries ``get_info`` first and falls back to
    ``info`` when needed while swallowing network-related exceptions.
    """

    for attr in ("get_info", "info"):
        try:
            source = getattr(ticker, attr, None)
            if callable(source):
                result = source()
            else:
                result = source
            if isinstance(result, dict):
                return result
        except Exception as exc:
            logger.debug("Unable to load %s for %s: %s", attr, ticker.ticker, exc)
            continue
    return {}


def _prepare_row(ticker_symbol: str, av_client: Optional[AlphaVantageClient] = None) -> Dict[str, Any]:
    ticker = yf.Ticker(ticker_symbol)
    info = _get_info_dict(ticker)

    income_df = getattr(ticker, "income_stmt", None)
    balance_df = getattr(ticker, "balance_sheet", None)
    cashflow_df = getattr(ticker, "cashflow", None)

    for attr in ("get_income_stmt", "get_balance_sheet", "get_cashflow"):
        getter = getattr(ticker, attr, None)
        if callable(getter):
            try:
                value = getter()
                if attr == "get_income_stmt" and (income_df is None or income_df.empty):
                    income_df = value
                if attr == "get_balance_sheet" and (balance_df is None or balance_df.empty):
                    balance_df = value
                if attr == "get_cashflow" and (cashflow_df is None or cashflow_df.empty):
                    cashflow_df = value
            except Exception as exc:
                logger.debug("Unable to call %s for %s: %s", attr, ticker_symbol, exc)
                continue

    data: Dict[str, Any] = {field: None for field in FIELDS}
    data["Price"] = _price(info, ticker)
    data["Shares_Out (M)"] = _shares_outstanding(info, ticker)
    data["Total_Debt"] = _extract_field(balance_df, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])
    data["Cash"] = _extract_field(balance_df, ["Cash And Cash Equivalents", "Cash"])
    data["EBITDA"] = _extract_field(income_df, ["Ebitda"])
    data["EBIT"] = _extract_field(income_df, ["Ebit"])
    data["Net_Income"] = _extract_field(income_df, ["Net Income"])
    data["Revenue"] = _extract_field(income_df, ["Total Revenue", "Revenue"])
    data["COGS"] = _extract_field(income_df, ["Cost Of Revenue", "Cost Of Goods Sold"])
    data["Total_Equity"] = _extract_field(
        balance_df, ["Total Stockholder Equity", "Total Equity Gross Minority Interest"]
    )
    data["Total_Assets"] = _extract_field(balance_df, ["Total Assets"])
    data["OCF"] = _extract_field(
        cashflow_df,
        ["Total Cash From Operating Activities", "Operating Cash Flow", "Cash Provided By Operating Activities"],
    )
    data["FCF"] = _free_cash_flow(cashflow_df)
    data["Dividends_Paid"] = _extract_field(cashflow_df, ["Cash Dividends Paid", "Dividends Paid"])
    data["Interest_Expense"] = _extract_field(income_df, ["Interest Expense"])
    data["Current_Assets"] = _extract_field(balance_df, ["Total Current Assets"])
    data["Current_Liabilities"] = _extract_field(balance_df, ["Total Current Liabilities"])
    data["Receivables"] = _extract_field(balance_df, ["Net Receivables", "Accounts Receivable"])
    data["Inventory"] = _extract_field(balance_df, ["Inventory"])
    data["WACC"] = "N/A (requires external source)"
    data["Tax_Rate"] = _tax_rate(income_df)
    data["DCF_Value_per_Share"] = "N/A (requires external source)"
    data["Insider_Net_Buys"] = "N/A (requires external source)"
    inst_pct = _safe_info_lookup(info, ["heldPercentInstitutions"])
    data["Institutional_%"] = inst_pct * 100 if inst_pct is not None else None
    data["Fund_Flows_3M (M)"] = "N/A (requires external source)"
    short_pct = _safe_info_lookup(info, ["shortPercentOfFloat", "shortRatio"])
    data["Short_Interest_%"] = short_pct * 100 if short_pct is not None else None
    data["Analyst_Rating(1-5)"] = _safe_info_lookup(info, ["recommendationMean"])
    data["Beta"] = _safe_info_lookup(info, ["beta", "beta3Year"])
    data["EPS_ttm"] = _safe_info_lookup(info, ["trailingEps"])
    data["EPS_CAGR_3Y"] = "N/A (requires external source)"
    data["Revenue_CAGR_3Y"] = _revenue_cagr(income_df)
    data["FCF_CAGR_3Y"] = _fcf_cagr(cashflow_df)
    data["Dividend_CAGR_3Y"] = _dividend_cagr(ticker)
    data["Altman_Z"] = "N/A (requires external source)"

    if av_client is not None:
        _augment_with_alpha_vantage(av_client, ticker_symbol, data)

    return data


def build_dataset(tickers: List[str], av_client: Optional[AlphaVantageClient]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for symbol in tickers:
        logger.info("Processing %s", symbol)
        try:
            row = _prepare_row(symbol, av_client)
            row["Ticker"] = symbol
            rows.append(row)
        except Exception as exc:
            logger.exception("Failed to process %s", symbol)
            fallback = {field: None for field in FIELDS}
            fallback["Ticker"] = symbol
            fallback["Error"] = str(exc)
            rows.append(fallback)
    df = pd.DataFrame(rows)
    column_order = ["Ticker"] + FIELDS
    for col in df.columns:
        if col not in column_order:
            column_order.append(col)
    return df[column_order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch fundamental metrics from Yahoo Finance for a list of tickers and export the data to an Excel file."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a text file containing one ticker per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.xlsx"),
        help="Where to write the resulting Excel file.",
    )
    parser.add_argument(
        "--alphavantage-api-key",
        default=os.getenv("ALPHAVANTAGE_API_KEY"),
        help=(
            "Alpha Vantage API key to use as a backup data source when Yahoo Finance data is missing or unavailable."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    av_client = AlphaVantageClient(args.alphavantage_api_key)
    tickers = _read_tickers(args.input_file)
    dataset = build_dataset(tickers, av_client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_excel(args.output, index=False)
    logger.info("Saved %d rows to %s", len(dataset), args.output)


if __name__ == "__main__":
    main()
