class Ticker:
    def __init__(self, symbol):
        self.ticker = symbol

    # Placeholder attributes for compatibility; real behavior is patched in tests.
    info = {}
    income_stmt = None
    balance_sheet = None
    cashflow = None

    def history(self, period="1d"):
        from pandas import DataFrame
        return DataFrame()

    def get_shares_full(self):
        from pandas import DataFrame
        return DataFrame()
