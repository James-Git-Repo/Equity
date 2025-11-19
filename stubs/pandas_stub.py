import csv
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union


class _Options:
    def __init__(self):
        class _Mode:
            def __init__(self):
                self.chained_assignment = None

        self.mode = _Mode()


options = _Options()


def to_datetime(values: Iterable[Any]):
    result = []
    for v in values:
        if isinstance(v, datetime):
            result.append(v)
        else:
            try:
                result.append(datetime.fromisoformat(str(v)))
            except Exception:
                result.append(v)
    return result


class Index:
    def __init__(self, values: List[Any]):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    @property
    def year(self):
        years = []
        for v in self._values:
            year_val = getattr(v, "year", None)
            years.append(year_val if year_val is not None else v)
        return years

    def __repr__(self):
        return f"Index({self._values})"


class Series:
    def __init__(self, data: Optional[List[Any]] = None, index: Optional[List[Any]] = None, dtype=None):
        self.data = list(data) if data is not None else []
        self.index = Index(list(index) if index is not None else list(range(len(self.data))))

    @property
    def empty(self) -> bool:
        return len(self.data) == 0

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        try:
            pos = list(self.index).index(idx)
            return self.data[pos]
        except ValueError:
            raise

    def groupby(self, keys):
        key_list = list(keys)
        groups: Dict[Any, List[Any]] = {}
        for key, value in zip(key_list, self.data):
            groups.setdefault(key, []).append(value)
        return GroupBy(groups)

    def sum(self):
        s = 0
        for v in self.data:
            s += v
        return s

    def sort_index(self, ascending=True):
        paired = list(zip(self.index, self.data))
        paired.sort(key=lambda x: x[0], reverse=not ascending)
        new_index = [p[0] for p in paired]
        new_data = [p[1] for p in paired]
        return Series(new_data, index=new_index)

    @property
    def iloc(self):
        return _SeriesILoc(self)

    def __repr__(self):
        return f"Series({self.data})"


class GroupBy:
    def __init__(self, groups: Dict[Any, List[Any]]):
        self.groups = groups

    def sum(self):
        return Series([sum(vals) for _, vals in sorted(self.groups.items())], index=[k for k, _ in sorted(self.groups.items())])


class _SeriesILoc:
    def __init__(self, series: Series):
        self.series = series

    def __getitem__(self, idx):
        return self.series.data[idx]


class _ILoc:
    def __init__(self, df: "DataFrame"):
        self.df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_sel, col_sel = key
            if isinstance(col_sel, slice):
                cols = self.df.columns[col_sel]
            elif isinstance(col_sel, list):
                cols = [self.df.columns[i] for i in col_sel]
            else:
                cols = [self.df.columns[col_sel]]
            if isinstance(row_sel, slice):
                row_indices = range(len(self.df.index))[row_sel]
            else:
                row_indices = range(len(self.df.index))
            data = []
            for ri in row_indices:
                row = {}
                for c in cols:
                    row[c] = self.df._get_value_by_pos(ri, c)
                data.append(row)
            if len(cols) == 1 and not isinstance(col_sel, slice):
                return Series([row[cols[0]] for row in data], index=[self.df.index[i] for i in row_indices])
            return DataFrame(data)
        else:
            # row selection only
            return self.df._get_row_by_pos(key)


class _Loc:
    def __init__(self, df: "DataFrame"):
        self.df = df

    def __getitem__(self, key):
        return self.df._get_row_by_label(key)


class DataFrame:
    def __init__(self, data: Optional[Union[Dict[str, Dict[Any, Any]], List[Dict[str, Any]]]] = None):
        self._data: Dict[Any, Dict[str, Any]] = {}
        self.columns: List[str] = []
        if data is None:
            return
        if isinstance(data, list):
            # list of row dicts
            for idx, row in enumerate(data):
                self._data[idx] = dict(row)
                for col in row.keys():
                    if col not in self.columns:
                        self.columns.append(col)
        elif isinstance(data, dict):
            # column-oriented
            self.columns = list(data.keys())
            primary_keys: List[Any] = []
            first_values = next(iter(data.values())) if data else {}
            if isinstance(first_values, dict):
                primary_keys = list(first_values.keys())
            all_index = list(primary_keys)
            for col, col_values in data.items():
                if isinstance(col_values, dict):
                    for key in col_values.keys():
                        if key not in all_index:
                            all_index.append(key)
            for idx in all_index:
                self._data[idx] = {}
                for col, col_values in data.items():
                    self._data[idx][col] = col_values.get(idx)
        self.index = Index(list(self._data.keys()))

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    @property
    def iloc(self):
        return _ILoc(self)

    @property
    def loc(self):
        return _Loc(self)

    def sort_index(self, axis=0, ascending=True):
        if axis == 0:
            sorted_idx = sorted(self._data.keys(), reverse=not ascending)
            new_data = [self._get_row_by_label(idx) for idx in sorted_idx]
            return DataFrame(new_data)
        elif axis == 1:
            sorted_cols = sorted(self.columns, reverse=not ascending)
            df = DataFrame()
            df.columns = sorted_cols
            df._data = {}
            for idx in self.index:
                df._data[idx] = {col: self._data.get(idx, {}).get(col) for col in sorted_cols}
            df.index = self.index
            return df
        else:
            return self

    def __getitem__(self, key):
        if isinstance(key, list):
            new_data = []
            for idx in self.index:
                row = {col: self._data.get(idx, {}).get(col) for col in key}
                new_data.append(row)
            df = DataFrame(new_data)
            df.columns = key
            df.index = list(range(len(new_data)))
            return df
        else:
            return Series([self._data[idx].get(key) for idx in self.index], index=self.index)

    def _get_row_by_pos(self, pos: int):
        idx = list(self.index)[pos]
        return self._get_row_by_label(idx)

    def _get_value_by_pos(self, pos: int, column: str):
        idx = list(self.index)[pos]
        return self._data.get(idx, {}).get(column)

    def _get_row_by_label(self, label: Any):
        row = self._data.get(label, {})
        values = [row.get(col) for col in self.columns]
        series = Series(values, index=self.columns)
        return series

    def to_excel(self, path, index=False):
        # Write simple CSV-compatible file
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for idx in self.index:
                row = self._data[idx]
                writer.writerow([row.get(col, "") for col in self.columns])

    def __getitem_row__(self, idx):
        return self._get_row_by_pos(idx)

    def __repr__(self):
        return f"DataFrame(columns={self.columns}, rows={len(self.index)})"


def read_excel(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return DataFrame()
    headers = rows[0]
    data_rows = [dict(zip(headers, row)) for row in rows[1:]]
    return DataFrame(data_rows)


__all__ = ["DataFrame", "Series", "to_datetime", "read_excel"]
