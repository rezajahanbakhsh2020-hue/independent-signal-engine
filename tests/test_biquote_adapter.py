"""BiQuote adapter tests; no live network calls."""
from __future__ import annotations

import json
import urllib.error

import pandas as pd
import pytest

from signal_engine.data.biquote_source import (
    BiQuoteUnavailableError,
    fetch_range,
)
from signal_engine.data.validate import DataValidationError


def _response(payload):
    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode()
    return Resp()


def test_fetch_xauusd_filters_open_bar_and_records_tick_volume(monkeypatch):
    payload = {
        "symbol": "XAUUSD",
        "interval": "1h",
        "bars": [
            {
                "openTime": "2024-01-01T02:00:00Z",
                "open": 2020, "high": 2022, "low": 2019, "close": 2021,
                "volume": 0, "tickVolume": 20, "isOpen": True,
            },
            {
                "openTime": "2024-01-01T01:00:00Z",
                "open": 2010, "high": 2012, "low": 2009, "close": 2011,
                "volume": 0, "tickVolume": 17, "isOpen": False,
            },
            {
                "openTime": "2024-01-01T00:00:00Z",
                "open": 2000, "high": 2002, "low": 1999, "close": 2001,
                "volume": 0, "tickVolume": 12, "isOpen": False,
            },
        ],
    }
    monkeypatch.setattr(
        "signal_engine.data.biquote_source.urllib.request.urlopen",
        lambda req, timeout=None: _response(payload),
    )
    df = fetch_range(
        "XAUUSD", "1H",
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T03:00:00Z"),
    )
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df.index[-1] == pd.Timestamp("2024-01-01T01:00:00Z")
    assert df["volume"].tolist() == [12.0, 17.0]
    assert df.attrs["source"] == "public:biquote-ohlc"
    assert df.attrs["volume_semantics"].startswith("tickVolume")


def test_biquote_does_not_map_btcusd_to_btcusdt():
    with pytest.raises(DataValidationError, match="XAUUSD only"):
        fetch_range(
            "BTCUSDT", "1H",
            pd.Timestamp("2024-01-01T00:00:00Z"),
            pd.Timestamp("2024-01-02T00:00:00Z"),
        )


def test_biquote_network_failure_fails_closed(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr("signal_engine.data.biquote_source.urllib.request.urlopen", fail)
    with pytest.raises(BiQuoteUnavailableError, match="offline"):
        fetch_range(
            "XAUUSD", "1H",
            pd.Timestamp("2024-01-01T00:00:00Z"),
            pd.Timestamp("2024-01-02T00:00:00Z"),
        )
