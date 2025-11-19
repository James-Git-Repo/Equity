try:  # pragma: no cover - shim path
    from pandas import DataFrame  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from . import pandas_stub as _pd  # type: ignore

    DataFrame = _pd.DataFrame


class Ticker:
    def __init__(self, symbol):
        self.ticker = symbol

    # Placeholder attributes for compatibility; real behavior is patched in tests.
    info = {}
    income_stmt = None
    balance_sheet = None
    cashflow = None

    def history(self, period="1d"):
        return DataFrame()

    def get_shares_full(self):
        return DataFrame()
