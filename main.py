import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Prefer real pandas for production; fall back to lightweight stub for tests.
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in restricted envs
    from stubs import pandas_stub as pd  # type: ignore

try:  # Requests drives real HTTP calls for Alpha Vantage/FMP.
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from stubs import requests_stub as requests  # type: ignore

try:
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from stubs import yfinance_stub as yf  # type: ignore


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


class FinancialModelingPrepClient:
    """Optional backup client that pulls data from FinancialModelingPrep."""

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, path: str) -> Optional[List[Dict[str, Any]]]:
        if not self.api_key:
            return None
        try:
            resp = self.session.get(
                f"https://financialmodelingprep.com{path}",
                params={"apikey": self.api_key},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                return None
            return payload
        except Exception as exc:
            logger.debug("FMP request failed for %s: %s", path, exc)
            return None

    def get_key_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        metrics = self._get(f"/api/v3/key-metrics/{symbol}")
        return metrics[0] if metrics else None

    def get_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        profile = self._get(f"/api/v3/profile/{symbol}")
        return profile[0] if profile else None

    def get_cash_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        reports = self._get(f"/api/v3/cash-flow-statement/{symbol}?limit=1")
        return reports[0] if reports else None

    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        reports = self._get(f"/api/v3/balance-sheet-statement/{symbol}?limit=1")
        return reports[0] if reports else None


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


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _extract_field(df: pd.DataFrame, names: Iterable[str]) -> Optional[float]:
    """Return the latest column value matching any of the provided labels.

    This matcher is resilient to minor label differences (spacing, punctuation,
    singular/plural) to reduce the chance of missing metrics like current assets
    or liabilities when providers rename fields.
    """

    if df is None or df.empty:
        return None
    series = _latest_column(df)
    lowered_index = {str(idx).lower(): idx for idx in series.index}
    normalized_index = {_normalize_label(str(idx)): idx for idx in series.index}

    for name in names:
        key = name.lower()
        norm = _normalize_label(name)
        if key in lowered_index:
            raw_value = series[lowered_index[key]]
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None
        if norm in normalized_index:
            raw_value = series[normalized_index[norm]]
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None
        for idx_norm, idx in normalized_index.items():
            if norm in idx_norm or idx_norm in norm:
                raw_value = series[idx]
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
    """Compute CAGR over ``years_between`` using oldest → newest values.

    yfinance statements often arrive with the most recent period in the first
    column, but indexes may be unordered. We sort by index when possible and
    gracefully fall back to first/last positions so the formula matches the
    guidance provided for 3Y revenue/EPS/FCF/dividend growth.
    """

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


def _strip_nulls(series: Any) -> Optional[pd.Series]:
    """Remove ``None``/null entries from a Series-like object.

    Works with real pandas objects *and* the lightweight stubs that omit
    ``dropna``. Returns ``None`` when the series cannot be cleaned.
    """

    if series is None:
        return None
    try:
        dropna = getattr(series, "dropna", None)
        if callable(dropna):
            return dropna()
    except Exception:
        pass

    try:
        values = []
        idx = []
        iterable = list(series)
        for pos, value in enumerate(iterable):
            if value is None:
                continue
            values.append(value)
            try:
                idx_val = series.index[pos]
            except Exception:
                idx_val = pos
            idx.append(idx_val)
        return pd.Series(values, index=idx)
    except Exception:
        return None


def _get_row_series(df: pd.DataFrame, row_name: str) -> Optional[pd.Series]:
    """Return a cleaned row series from a DataFrame using robust fallbacks."""

    if df is None or df.empty:
        return None
    try:
        raw = df.loc[row_name]
    except Exception:
        return None
    cleaned = _strip_nulls(raw)
    if cleaned is None:
        return None
    return cleaned


def _safe_loc(df: pd.DataFrame, row_name: str) -> Optional[float]:
    """Return the latest value for a row while tolerating missing labels.

    Works with both pandas and the bundled stubs by avoiding reliance on
    ``dropna``. Falls back to the first or last available entry when ordering is
    ambiguous.
    """

    series = _get_row_series(df, row_name)
    if series is None or len(series) == 0:
        return None
    try:
        return float(series.iloc[0])
    except Exception:
        pass
    try:
        return float(list(series)[0])
    except Exception:
        pass
    try:
        return float(series.sort_index().iloc[-1])
    except Exception:
        return None


def _cagr_from_series(series: Optional[pd.Series], years: int = 3) -> Optional[float]:
    """Compute CAGR from a yearly series, returning ``None`` when insufficient."""

    if series is None or len(series) < years + 1:
        return None
    ordered = series.sort_index()
    try:
        start = float(ordered.iloc[-(years + 1)])
        end = float(ordered.iloc[-1])
    except (TypeError, ValueError):
        return None
    if start <= 0 or end <= 0:
        return None
    try:
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


def _coerce_year_labels(index: Iterable[Any]) -> List[int]:
    """Convert index labels to sortable integers with stable fallbacks."""

    coerced: List[int] = []
    for pos, label in enumerate(index):
        try:
            coerced.append(int(str(label)[:4]))
        except Exception:
            coerced.append(-pos)
    return coerced


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


def _tax_rate_av(income_reports: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    if not income_reports:
        return None
    sorted_reports = sorted(
        income_reports, key=lambda r: r.get("fiscalDateEnding", ""), reverse=True
    )
    for report in sorted_reports:
        tax = _av_number(report.get("incomeTaxExpense"))
        pretax = _av_number(report.get("incomeBeforeTax"))
        if tax is not None and pretax not in (None, 0):
            return tax / pretax
    return None


def _free_cash_flow(cashflow_df: pd.DataFrame) -> Optional[float]:
    operating = _extract_field(
        cashflow_df,
        [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Net Cash Provided By Operating Activities",
            "Cash Provided By Operating Activities",
            "Cash From Operations",
        ],
    )
    capex = _extract_field(
        cashflow_df, ["Capital Expenditures", "Purchase Of Property And Equipment"]
    )
    if operating is None or capex is None:
        fcf_direct = _extract_field(
            cashflow_df, ["Free Cash Flow", "Net Free Cash Flow", "Free Cash Flow To Firm"]
        )
        if fcf_direct is not None:
            return fcf_direct
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


def _yahoo_eps_cagr(financials: pd.DataFrame) -> Optional[float]:
    row = _get_row_series(financials, "Diluted EPS")
    if row is None or len(row) < 4:
        return None
    try:
        eps_series = pd.Series(list(row), index=_coerce_year_labels(getattr(row, "index", range(len(row)))))
        return _cagr_from_series(eps_series, years=3)
    except Exception:
        return None


def _yahoo_revenue_cagr(financials: pd.DataFrame) -> Optional[float]:
    row = _get_row_series(financials, "Total Revenue")
    if row is None or len(row) < 4:
        return None
    try:
        rev_series = pd.Series(list(row), index=_coerce_year_labels(getattr(row, "index", range(len(row)))))
        return _cagr_from_series(rev_series, years=3)
    except Exception:
        return None


def _yahoo_fcf_cagr(cashflow_df: pd.DataFrame) -> Optional[float]:
    ocf_hist = _get_row_series(cashflow_df, "Total Cash From Operating Activities")
    capex_hist = _get_row_series(cashflow_df, "Capital Expenditures")
    if ocf_hist is None or capex_hist is None:
        return None

    ocf_idx = list(getattr(ocf_hist, "index", range(len(ocf_hist))))
    capex_idx = list(getattr(capex_hist, "index", range(len(capex_hist))))
    common_labels = [label for label in ocf_idx if label in capex_idx]
    if len(common_labels) < 4:
        return None

    ocf_map = {label: value for label, value in zip(ocf_idx, list(ocf_hist))}
    capex_map = {label: value for label, value in zip(capex_idx, list(capex_hist))}
    values = []
    for label in common_labels:
        values.append(ocf_map[label] + capex_map[label])

    fcf_series = pd.Series(values, index=_coerce_year_labels(common_labels))
    return _cagr_from_series(fcf_series, years=3)


def _yahoo_dividend_cagr(ticker: yf.Ticker) -> Optional[float]:
    try:
        divs = ticker.dividends
        if divs is None or divs.empty:
            return None
        dps_yearly = divs.groupby(divs.index.year).sum()
        return _cagr_from_series(dps_yearly, years=3)
    except Exception:
        return None


def _yahoo_tax_rate(financials: pd.DataFrame) -> Optional[float]:
    try:
        income_tax = _safe_loc(financials, "Income Tax Expense")
        pre_tax_income = _safe_loc(financials, "Income Before Tax")
        if income_tax is None or pre_tax_income in (None, 0):
            return None
        return income_tax / pre_tax_income
    except Exception:
        return None


def _yahoo_altman_z(balance_sheet: pd.DataFrame, financials: pd.DataFrame, info: Dict[str, Any]) -> Optional[float]:
    try:
        total_assets = _safe_loc(balance_sheet, "Total Assets")
        total_liab = _safe_loc(balance_sheet, "Total Liab")
        current_assets = _safe_loc(balance_sheet, "Total Current Assets")
        current_liabilities = _safe_loc(balance_sheet, "Total Current Liabilities")
        retained_earnings = _safe_loc(balance_sheet, "Retained Earnings")
        ebit = _safe_loc(financials, "Operating Income")
        revenue = _safe_loc(financials, "Total Revenue")
        shares_out = info.get("sharesOutstanding")
        price = info.get("currentPrice") or info.get("regularMarketPrice")

        if None in (
            total_assets,
            total_liab,
            current_assets,
            current_liabilities,
            retained_earnings,
            ebit,
            revenue,
            shares_out,
            price,
        ):
            return None

        working_capital = current_assets - current_liabilities
        market_value_equity = float(shares_out) * float(price)
        if total_liab == 0 or total_assets == 0:
            return None

        return float(
            1.2 * (working_capital / total_assets)
            + 1.4 * (retained_earnings / total_assets)
            + 3.3 * (ebit / total_assets)
            + 0.6 * (market_value_equity / total_liab)
            + 1.0 * (revenue / total_assets)
        )
    except Exception:
        return None


def _yahoo_wacc(info: Dict[str, Any], balance_sheet: pd.DataFrame, financials: pd.DataFrame) -> Optional[float]:
    try:
        beta = info.get("beta")
        if beta is None:
            return None
        total_debt = _safe_loc(balance_sheet, "Total Debt")
        if total_debt is None:
            short_debt = _safe_loc(balance_sheet, "Short Long Term Debt")
            long_debt = _safe_loc(balance_sheet, "Long Term Debt")
            if short_debt is not None or long_debt is not None:
                total_debt = (short_debt or 0) + (long_debt or 0)
        if total_debt in (None, 0):
            return None

        interest_exp = _safe_loc(financials, "Interest Expense")
        shares_out = info.get("sharesOutstanding")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if interest_exp is None or shares_out in (None, 0) or price in (None, 0):
            return None

        cost_debt = abs(interest_exp) / total_debt
        equity_value = float(shares_out) * float(price)
        debt_value = float(total_debt)
        if equity_value + debt_value == 0:
            return None

        tax_rate = _yahoo_tax_rate(financials) or 0
        cost_equity = 0.04 + beta * 0.05
        return float(
            (equity_value / (debt_value + equity_value)) * cost_equity
            + (debt_value / (debt_value + equity_value)) * cost_debt * (1 - tax_rate)
        )
    except Exception:
        return None


def _compute_wacc(data: Dict[str, Any]) -> Optional[float]:
    beta = data.get("Beta")
    total_debt = data.get("Total_Debt") or 0
    total_equity = data.get("Total_Equity")
    if total_equity is None:
        shares = data.get("Shares_Out (M)")
        price = data.get("Price")
        if shares is not None and price is not None:
            total_equity = shares * 1_000_000 * price

    if beta is None or total_equity is None:
        return None

    # CAPM-style cost of equity with conservative baseline assumptions.
    risk_free = 0.04
    market_premium = 0.05
    cost_equity = risk_free + beta * market_premium

    interest_exp = data.get("Interest_Expense")
    cost_debt = None
    if total_debt and interest_exp not in (None, 0):
        cost_debt = max(interest_exp / total_debt, 0)
    elif total_debt:
        cost_debt = 0.05  # fallback baseline cost of debt when debt exists

    capital = total_equity + (total_debt or 0)
    if capital <= 0:
        return None

    equity_weight = total_equity / capital
    debt_weight = (total_debt or 0) / capital
    tax_rate = data.get("Tax_Rate")
    tax_shield = 1 - tax_rate if tax_rate is not None else 1

    if cost_debt is None:
        return equity_weight * cost_equity
    return equity_weight * cost_equity + debt_weight * cost_debt * tax_shield


def _augment_with_yahoo_statements(
    info: Dict[str, Any],
    ticker: yf.Ticker,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    data: Dict[str, Any],
) -> Optional[float]:
    """Populate metrics directly from Yahoo statements per the provided mapping.

    The mapping prioritizes Yahoo's canonical labels (e.g., ``Total Cash From
    Operating Activities`` for OCF) and computes derived metrics such as FCF,
    tax rate, CAGR figures, Altman Z, and WACC using the formulas supplied in
    the user's guidance. Fuzzy matching and previously set values are preserved
    to avoid clobbering fallback data.
    """

    def set_if_missing(field: str, value: Optional[float]) -> None:
        if value is None:
            return
        current = data.get(field)
        if current is None or isinstance(current, str):
            data[field] = value

    # Price and shares from info with a history fallback.
    price = _safe_info_lookup(info, ["currentPrice", "regularMarketPrice", "lastPrice"])
    if price is None:
        price = _price(info, ticker)
    set_if_missing("Price", price)

    shares_out = _safe_info_lookup(info, ["sharesOutstanding"])
    if shares_out is None:
        shares_out = _safe_info_lookup(info, ["floatShares", "shares_outstanding"])
    if shares_out is None:
        shares_out = _safe_info_lookup(info, ["sharesOutstanding"])  # retry before fallback
    if shares_out is None:
        shares_out_m = _shares_outstanding(info, ticker)
    else:
        shares_out_m = shares_out / 1_000_000
    set_if_missing("Shares_Out (M)", shares_out_m)

    # Debt: prefer totalDebt, otherwise rebuild from short/long-term pieces.
    total_debt = _safe_info_lookup(info, ["totalDebt"])
    short_debt = _safe_loc(balance_df, "Short Long Term Debt")
    long_debt = _safe_loc(balance_df, "Long Term Debt")
    if total_debt is None:
        total_debt = _safe_loc(balance_df, "Total Debt")
    if total_debt is None and (short_debt is not None or long_debt is not None):
        total_debt = (short_debt or 0) + (long_debt or 0)
    set_if_missing("Total_Debt", total_debt)

    set_if_missing("Cash", _safe_info_lookup(info, ["totalCash"]) or _safe_loc(balance_df, "Cash And Cash Equivalents"))
    set_if_missing("EBITDA", _safe_info_lookup(info, ["ebitda"]) or _safe_loc(income_df, "Ebitda"))
    set_if_missing("EBIT", _safe_loc(income_df, "Operating Income"))
    set_if_missing("Net_Income", _safe_info_lookup(info, ["netIncome"]) or _safe_loc(income_df, "Net Income"))
    set_if_missing("Revenue", _safe_info_lookup(info, ["totalRevenue"]) or _safe_loc(income_df, "Total Revenue"))
    set_if_missing("COGS", _safe_loc(income_df, "Cost Of Revenue"))
    set_if_missing("Total_Equity", _safe_loc(balance_df, "Total Stockholder Equity"))
    set_if_missing("Total_Assets", _safe_loc(balance_df, "Total Assets"))

    ocf = _extract_field(
        cashflow_df,
        [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash Provided By Operating Activities",
            "Net Cash Provided by Operating Activities",
            "Net cash provided by operating activities",
            "Cash Flow From Continuing Operating Activities",
            "Net Cash From Operating Activities",
        ],
    )
    set_if_missing("OCF", ocf)
    capex = _extract_field(
        cashflow_df,
        [
            "Capital Expenditures",
            "Purchase of property and equipment",
            "Purchases of property, plant and equipment",
        ],
    )
    if data.get("FCF") is None and ocf is not None and capex is not None:
        data["FCF"] = ocf + capex
    if data.get("FCF") is None:
        set_if_missing("FCF", _safe_loc(cashflow_df, "Free Cash Flow"))

    set_if_missing("Dividends_Paid", _safe_loc(cashflow_df, "Dividends Paid"))
    set_if_missing("Interest_Expense", _safe_loc(income_df, "Interest Expense"))
    set_if_missing(
        "Current_Assets",
        _extract_field(
            balance_df,
            ["Total Current Assets", "Current Assets", "Current Assets (Total)", "Current Assets Total"],
        ),
    )
    set_if_missing(
        "Current_Liabilities",
        _extract_field(
            balance_df,
            [
                "Total Current Liabilities",
                "Current Liabilities",
                "Total Current Liability",
                "Current Liabilities Total",
                "Current Liability",
            ],
        ),
    )
    set_if_missing("Receivables", _safe_loc(balance_df, "Net Receivables"))
    set_if_missing("Inventory", _safe_loc(balance_df, "Inventory"))

    inst_pct = _safe_info_lookup(info, ["heldPercentInstitutions"])
    if inst_pct is not None:
        set_if_missing("Institutional_%", inst_pct * 100)

    short_pct = _safe_info_lookup(info, ["shortPercentOfFloat"])
    if short_pct is not None:
        set_if_missing("Short_Interest_%", short_pct * 100)

    set_if_missing("Analyst_Rating(1-5)", _safe_info_lookup(info, ["recommendationMean"]))
    set_if_missing("Beta", _safe_info_lookup(info, ["beta", "beta3Year"]))
    set_if_missing("EPS_ttm", _safe_info_lookup(info, ["trailingEps"]))

    set_if_missing("Tax_Rate", _yahoo_tax_rate(income_df))
    eps_cagr = _yahoo_eps_cagr(income_df)
    if eps_cagr is not None:
        data["EPS_CAGR_3Y"] = eps_cagr
    rev_cagr = _yahoo_revenue_cagr(income_df)
    if rev_cagr is not None:
        data["Revenue_CAGR_3Y"] = rev_cagr
    fcf_cagr = _yahoo_fcf_cagr(cashflow_df)
    if fcf_cagr is not None:
        data["FCF_CAGR_3Y"] = fcf_cagr
    dividend_cagr = _yahoo_dividend_cagr(ticker)
    if dividend_cagr is not None:
        data["Dividend_CAGR_3Y"] = dividend_cagr

    altman_z = _yahoo_altman_z(balance_df, income_df, info)
    if altman_z is not None:
        data["Altman_Z"] = altman_z

    return _yahoo_wacc(info, balance_df, income_df)


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
        if total_debt is None:
            total_debt = _av_latest(balance_reports, "totalDebt")
        data["Total_Debt"] = total_debt

    if data.get("Current_Assets") is None:
        data["Current_Assets"] = _av_latest(
            balance_reports, "totalCurrentAssets"
        ) or _av_latest(balance_reports, "currentAssets")

    if data.get("Current_Liabilities") is None:
        data["Current_Liabilities"] = _av_latest(
            balance_reports, "totalCurrentLiabilities"
        ) or _av_latest(balance_reports, "currentLiabilities")

    if data.get("Receivables") is None:
        data["Receivables"] = _av_latest(balance_reports, "currentNetReceivables")

    if data.get("Inventory") is None:
        data["Inventory"] = _av_latest(balance_reports, "inventory")

    if data.get("OCF") is None:
        data["OCF"] = _av_latest(cashflow_reports, "operatingCashflow") or _av_latest(
            cashflow_reports, "netCashProvidedByOperatingActivities"
        )

    if data.get("FCF") is None:
        op_cf = _av_latest(cashflow_reports, "operatingCashflow")
        if op_cf is None:
            op_cf = _av_latest(cashflow_reports, "netCashProvidedByOperatingActivities")
        capex = _av_latest(cashflow_reports, "capitalExpenditures")
        if op_cf is not None and capex is not None:
            data["FCF"] = op_cf + capex
        if data.get("FCF") is None:
            data["FCF"] = _av_latest(cashflow_reports, "freeCashFlow")

    if data.get("Dividends_Paid") is None:
        data["Dividends_Paid"] = _av_latest(cashflow_reports, "dividendPayout")

    if data.get("Interest_Expense") is None:
        data["Interest_Expense"] = _av_latest(income_reports, "interestExpense")

    if data.get("Tax_Rate") is None:
        data["Tax_Rate"] = _tax_rate_av(income_reports)


def _augment_with_fmp(
    fmp_client: FinancialModelingPrepClient, ticker_symbol: str, data: Dict[str, Any]
) -> None:
    profile = fmp_client.get_profile(ticker_symbol)
    metrics = fmp_client.get_key_metrics(ticker_symbol)
    balance = fmp_client.get_balance_sheet(ticker_symbol)
    cashflow = fmp_client.get_cash_flow(ticker_symbol)

    for source in (metrics, profile):
        if not source:
            continue
        if data.get("WACC") is None or isinstance(data.get("WACC"), str):
            data["WACC"] = _av_number(source.get("wacc")) or _av_number(
                source.get("weightedAverageCostOfCapital")
            )
        if data.get("Beta") is None:
            data["Beta"] = _av_number(source.get("beta"))

    if balance:
        if data.get("Total_Debt") is None:
            data["Total_Debt"] = _av_number(balance.get("totalDebt"))
        if data.get("Total_Equity") is None:
            data["Total_Equity"] = _av_number(balance.get("totalStockholdersEquity"))
        if data.get("Current_Assets") is None:
            data["Current_Assets"] = _av_number(
                balance.get("totalCurrentAssets") or balance.get("currentAssets")
            )
        if data.get("Current_Liabilities") is None:
            data["Current_Liabilities"] = _av_number(
                balance.get("totalCurrentLiabilities") or balance.get("currentLiabilities")
            )

    if cashflow:
        if data.get("OCF") is None:
            data["OCF"] = _av_number(cashflow.get("netCashProvidedByOperatingActivities"))
        if data.get("FCF") is None:
            fcf = _av_number(cashflow.get("freeCashFlow"))
            if fcf is not None:
                data["FCF"] = fcf


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


def _prepare_row(
    ticker_symbol: str,
    av_client: Optional[AlphaVantageClient] = None,
    fmp_client: Optional[FinancialModelingPrepClient] = None,
) -> Dict[str, Any]:
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

    income_df = income_df if isinstance(income_df, pd.DataFrame) else pd.DataFrame()
    balance_df = balance_df if isinstance(balance_df, pd.DataFrame) else pd.DataFrame()
    cashflow_df = cashflow_df if isinstance(cashflow_df, pd.DataFrame) else pd.DataFrame()

    data: Dict[str, Any] = {field: None for field in FIELDS}
    yahoo_wacc = _augment_with_yahoo_statements(info, ticker, income_df, balance_df, cashflow_df, data)

    data["DCF_Value_per_Share"] = "N/A (requires external source)"
    data["Insider_Net_Buys"] = "N/A (requires external source)"
    data["Fund_Flows_3M (M)"] = "N/A (requires external source)"

    if av_client is not None:
        _augment_with_alpha_vantage(av_client, ticker_symbol, data)
    if fmp_client is not None:
        _augment_with_fmp(fmp_client, ticker_symbol, data)

    if data.get("Tax_Rate") is None:
        data["Tax_Rate"] = _tax_rate(income_df)

    if data.get("WACC") is None or isinstance(data.get("WACC"), str):
        data["WACC"] = _compute_wacc(data)

    if data.get("WACC") is None and yahoo_wacc is not None:
        data["WACC"] = yahoo_wacc

    if data.get("WACC") is None:
        beta = data.get("Beta")
        if beta is not None:
            data["WACC"] = 0.04 + 0.05 * beta
        else:
            data["WACC"] = 0.08  # conservative baseline when all sources fail

    return data


def build_dataset(
    tickers: List[str],
    av_client: Optional[AlphaVantageClient],
    fmp_client: Optional[FinancialModelingPrepClient] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for symbol in tickers:
        logger.info("Processing %s", symbol)
        try:
            row = _prepare_row(symbol, av_client, fmp_client)
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
        "--fmp-api-key",
        default=os.getenv("FMP_API_KEY"),
        help="FinancialModelingPrep API key to use as a tertiary data source for cash flows, balance sheet, and WACC.",
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
    fmp_client = FinancialModelingPrepClient(args.fmp_api_key)
    tickers = _read_tickers(args.input_file)
    dataset = build_dataset(tickers, av_client, fmp_client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_excel(args.output, index=False)
    logger.info("Saved %d rows to %s", len(dataset), args.output)


if __name__ == "__main__":
    main()
