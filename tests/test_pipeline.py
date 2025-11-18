import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import main


class StubTicker:
    def __init__(self, symbol: str, info, income_df, balance_df, cashflow_df, dividends=None):
        self.ticker = symbol
        self.info = info
        self.income_stmt = income_df
        self.balance_sheet = balance_df
        self.cashflow = cashflow_df
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)

    def get_info(self):
        return self.info

    def get_income_stmt(self):
        return self.income_stmt

    def get_balance_sheet(self):
        return self.balance_sheet

    def get_cashflow(self):
        return self.cashflow


def _stub_yahoo_frames():
    income_df = pd.DataFrame(
        {
            "2023": {
                "Total Revenue": 1000,
                "Ebitda": 400,
                "Ebit": 350,
                "Net Income": 250,
                "Cost Of Revenue": 600,
                "Income Tax Expense": 50,
                "Income Before Tax": 300,
                "Interest Expense": 20,
            },
            "2022": {
                "Total Revenue": 900,
                "Ebitda": 380,
                "Ebit": 320,
                "Net Income": 200,
                "Cost Of Revenue": 540,
                "Income Tax Expense": 40,
                "Income Before Tax": 260,
                "Interest Expense": 18,
            },
            "2021": {
                "Total Revenue": 800,
                "Ebitda": 360,
                "Ebit": 300,
                "Net Income": 180,
                "Cost Of Revenue": 480,
                "Income Tax Expense": 36,
                "Income Before Tax": 216,
                "Interest Expense": 16,
            },
            "2020": {
                "Total Revenue": 700,
                "Ebitda": 340,
                "Ebit": 280,
                "Net Income": 150,
                "Cost Of Revenue": 420,
                "Income Tax Expense": 30,
                "Income Before Tax": 180,
                "Interest Expense": 14,
            },
        }
    )

    balance_df = pd.DataFrame(
        {
            "2023": {
                "Total Debt": 500,
                "Long Term Debt": 480,
                "Short Long Term Debt": 20,
                "Cash And Cash Equivalents": 200,
                "Total Stockholder Equity": 600,
                "Total Assets": 1200,
                "Total Current Assets": 300,
                "Total Current Liabilities": 150,
                "Net Receivables": 110,
                "Inventory": 75,
            }
        }
    )

    cashflow_df = pd.DataFrame(
        {
            "2023": {
                "Total Cash From Operating Activities": 320,
                "Capital Expenditures": -80,
                "Cash Dividends Paid": -20,
            },
            "2022": {
                "Total Cash From Operating Activities": 300,
                "Capital Expenditures": -70,
                "Cash Dividends Paid": -18,
            },
            "2021": {
                "Total Cash From Operating Activities": 280,
                "Capital Expenditures": -60,
                "Cash Dividends Paid": -16,
            },
            "2020": {
                "Total Cash From Operating Activities": 260,
                "Capital Expenditures": -50,
                "Cash Dividends Paid": -14,
            },
        }
    )

    dividends = pd.Series(
        data=[0.5, 0.4, 0.3, 0.2],
        index=pd.to_datetime(["2023-03-01", "2022-03-01", "2021-03-01", "2020-03-01"]),
    )
    return income_df, balance_df, cashflow_df, dividends


class PipelineTests(unittest.TestCase):
    def test_build_dataset_populates_from_yahoo(self):
        income_df, balance_df, cashflow_df, dividends = _stub_yahoo_frames()
        stub_info = {
            "currentPrice": 123.45,
            "sharesOutstanding": 1_000_000,
            "heldPercentInstitutions": 0.42,
            "shortPercentOfFloat": 0.05,
            "recommendationMean": 1.8,
            "beta": 1.2,
            "trailingEps": 5.5,
        }

        def fake_ticker(symbol):
            return StubTicker(symbol, stub_info, income_df, balance_df, cashflow_df, dividends=dividends)

        with patch("main.yf.Ticker", side_effect=fake_ticker):
            df = main.build_dataset(["TEST"], av_client=None)

        row = df.iloc[0]
        self.assertEqual(row["Price"], 123.45)
        self.assertEqual(row["Shares_Out (M)"], 1.0)
        self.assertEqual(row["Revenue"], 1000)
        self.assertEqual(row["Total_Debt"], 500)
        self.assertEqual(row["FCF"], 240)  # 320 + (-80)
        self.assertEqual(row["Tax_Rate"], 50 / 300)
        self.assertEqual(row["Institutional_%"], 42.0)
        self.assertEqual(row["Short_Interest_%"], 5.0)
        self.assertIsNotNone(row["Dividend_CAGR_3Y"])

    def test_alpha_vantage_backfills_missing_fields(self):
        # Empty Yahoo structures force the Alpha Vantage fallback path.
        def fake_ticker(symbol):
            return StubTicker(symbol, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        class StubAV(main.AlphaVantageClient):
            def __init__(self):
                pass

            def get_overview(self, symbol):
                return {
                    "SharesOutstanding": "2000000",
                    "Beta": "1.1",
                    "EPS": "3.3",
                }

            def get_income_statement(self, symbol):
                return [
                    {"fiscalDateEnding": "2023-12-31", "totalRevenue": "1500", "ebit": "400", "netIncome": "250"}
                ]

            def get_balance_sheet(self, symbol):
                return [
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "totalAssets": "2000",
                        "totalShareholderEquity": "900",
                        "cashAndCashEquivalentsAtCarryingValue": "300",
                        "shortLongTermDebtTotal": "250",
                        "totalCurrentAssets": "600",
                        "totalCurrentLiabilities": "350",
                        "currentNetReceivables": "200",
                        "inventory": "150",
                    }
                ]

            def get_cash_flow(self, symbol):
                return [
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "operatingCashflow": "500",
                        "capitalExpenditures": "-120",
                        "dividendPayout": "-30",
                    }
                ]

            def get_price(self, symbol):
                return 222.2

        with patch("main.yf.Ticker", side_effect=fake_ticker):
            df = main.build_dataset(["AVBK"], av_client=StubAV())

        row = df.iloc[0]
        self.assertEqual(row["Price"], 222.2)
        self.assertEqual(row["Shares_Out (M)"], 2.0)
        self.assertEqual(row["Revenue"], 1500.0)
        self.assertEqual(row["Total_Assets"], 2000.0)
        self.assertEqual(row["FCF"], 380.0)
        self.assertEqual(row["Dividends_Paid"], -30.0)

    def test_main_writes_output_file(self):
        income_df, balance_df, cashflow_df, dividends = _stub_yahoo_frames()

        def fake_ticker(symbol):
            return StubTicker(
                symbol, {"currentPrice": 10, "sharesOutstanding": 1_000_000}, income_df, balance_df, cashflow_df, dividends
            )

        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tickers.txt"
            input_path.write_text("TEST\n")
            output_path = Path(tmpdir) / "out.xlsx"

            with patch("main.yf.Ticker", side_effect=fake_ticker):
                with patch.object(sys, "argv", ["main.py", str(input_path), "--output", str(output_path)]):
                    main.main()

            self.assertTrue(output_path.exists())
            result_df = pd.read_excel(output_path)
            self.assertEqual(list(result_df["Ticker"]), ["TEST"])
            self.assertEqual(float(result_df.iloc[0]["Price"]), 10.0)


if __name__ == "__main__":
    unittest.main()
