import json
from urllib import request, parse


class Response:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data

    def json(self):
        try:
            return json.loads(self._data)
        except Exception:
            return {}

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")


class Session:
    def get(self, url, params=None, timeout=10):
        if params:
            url = f"{url}?{parse.urlencode(params)}"
        try:
            with request.urlopen(url, timeout=timeout) as resp:
                data = resp.read().decode()
                return Response(resp.getcode(), data)
        except Exception as exc:
            raise exc


__all__ = ["Session"]
