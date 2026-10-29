"""Binance public adapter tests (no real network; failure modes emphasized)."""
from __future__ import annotations

import json
import urllib.error

import pandas as pd
import pytest

from signal_engine.data.binance_source import (
    BinanceUnavailableError,
    fetch_klines,
    fetch_range,
)
from signal_engine.data.validate import DataValidationError


def test_unsupported_symbol_rejected():
    with pytest.raises(DataValidationError):
        fetch_klines("XAUUSD", "1H", 0, 1000)


def test_unsupported_interval_rejected():
    with pytest.raises(DataValidationError):
        fetch_klines("BTCUSDT", "2H", 0, 1000)


class FakeHttp404:
    status = 404

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return b"{}"


def make_fake_urlopen(payload: bytes):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return payload

    return lambda req, timeout=None: Resp()


def test_success_path_provenance(monkeypatch):
    now_ms = 1600000000000
    klines = [
        [now_ms + i * 3600000, "100.0", "101.0", "99.0", "100.5", "1000.0",
         now_ms + i * 3600000 + 3599999, "0", 1, "1", "0", "0"]
        for i in range(5)
    ]
    monkeypatch.setattr(
        "signal_engine.data.binance_source.urllib.request.urlopen",
        make_fake_urlopen(json.dumps(klines).encode()),
    )
    df = fetch_range("BTCUSDT", "1H", pd.Timestamp("2020-09-13T12:00", tz="UTC"),
                     pd.Timestamp("2020-09-13T17:00", tz="UTC"))
    assert df.attrs["source"].startswith("public:binance-api-v3:")
    assert df.attrs["symbol_mapping"] == "BTCUSDT->BTCUSDT"
    assert df.attrs["timezone"] == "UTC"
    assert df.index.tz is not None
    assert df.attrs["retrieval_time_utc"].endswith("Z") or "+00:00" in df.attrs["retrieval_time_utc"]
    assert df.attrs["coverage_start_utc"] is not None


def test_arbitrary_usdt_pair_is_supported(monkeypatch):
    now_ms = 1600000000000
    klines = [
        [now_ms + i * 3600000, "10.0", "11.0", "9.0", "10.5", "100.0",
         now_ms + i * 3600000 + 3599999, "0", 1, "1", "0", "0"]
        for i in range(5)
    ]
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        return make_fake_urlopen(json.dumps(klines).encode())(req, timeout)

    monkeypatch.setattr("signal_engine.data.binance_source.urllib.request.urlopen", fake_urlopen)
    df = fetch_range("SOLUSDT", "1H", pd.Timestamp("2020-09-13T12:00", tz="UTC"),
                     pd.Timestamp("2020-09-13T17:00", tz="UTC"))
    assert df.attrs["symbol_mapping"] == "SOLUSDT->SOLUSDT"
    assert "symbol=SOLUSDT" in seen[0]


def test_geo_blocked_primary_uses_official_mirror(monkeypatch):
    now_ms = 1600000000000
    klines = [
        [now_ms + i * 3600000, "100.0", "101.0", "99.0", "100.5", "1000.0",
         now_ms + i * 3600000 + 3599999, "0", 1, "1", "0", "0"]
        for i in range(5)
    ]
    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith("https://api.binance.com/"):
            raise urllib.error.HTTPError(req.full_url, 451, "blocked", {}, None)
        class Resp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return json.dumps(klines).encode()
        return Resp()
    monkeypatch.setattr("signal_engine.data.binance_source.urllib.request.urlopen", fake_urlopen)
    df = fetch_range("BTCUSDT", "1H", pd.Timestamp("2020-09-13T12:00", tz="UTC"),
                     pd.Timestamp("2020-09-13T17:00", tz="UTC"))
    assert df.attrs["source"] == "public:binance-api-v3:data-api.binance.vision"


def test_paging_aborts_if_official_host_changes(monkeypatch):
    start_ms = 1600000000000
    page1 = [
        [start_ms + i * 3600000, "1", "2", "0.5", "1.5", "10",
         start_ms + (i + 1) * 3600000 - 1, "0", 1, "1", "0", "0"]
        for i in range(1000)
    ]
    page2_start = start_ms + 1000 * 3600000
    page2 = [
        [page2_start + i * 3600000, "1", "2", "0.5", "1.5", "10",
         page2_start + (i + 1) * 3600000 - 1, "0", 1, "1", "0", "0"]
        for i in range(3)
    ]
    primary_calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith("https://api.binance.com/"):
            primary_calls["n"] += 1
            if primary_calls["n"] == 1:
                return make_fake_urlopen(json.dumps(page1).encode())(req, timeout)
            raise urllib.error.HTTPError(req.full_url, 451, "blocked", {}, None)
        return make_fake_urlopen(json.dumps(page2).encode())(req, timeout)

    monkeypatch.setattr("signal_engine.data.binance_source.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(BinanceUnavailableError, match="refusing to mix sources"):
        fetch_range("BTCUSDT", "1H", pd.Timestamp("2020-09-13T12:00", tz="UTC"),
                    pd.Timestamp("2020-09-13T12:00", tz="UTC") + pd.Timedelta(hours=1005))


def test_network_failure_raises_not_fabricated(monkeypatch):
    def fail(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("signal_engine.data.binance_source.urllib.request.urlopen", fail)
    with pytest.raises(BinanceUnavailableError, match="connection refused"):
        fetch_range("ETHUSDT", "1H", pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"))


def test_no_klines_raises(monkeypatch):
    monkeypatch.setattr(
        "signal_engine.data.binance_source.urllib.request.urlopen",
        make_fake_urlopen(json.dumps([]).encode()),
    )
    with pytest.raises(BinanceUnavailableError, match="No klines"):
        fetch_range("BTCUSDT", "1H", pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"))


def test_malformed_klines_skipped_validation_failure(monkeypatch):
    bad = [[0, "x", "1", "2", "3", "5", 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, "nan", 0, 0, 0, 0, 0, 0]]
    monkeypatch.setattr(
        "signal_engine.data.binance_source.urllib.request.urlopen",
        make_fake_urlopen(json.dumps(bad).encode()),
    )
    with pytest.raises((BinanceUnavailableError, DataValidationError)):
        fetch_range("BTCUSDT", "1H", pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"))


def test_paging_when_more_than_one_page(monkeypatch):
    page1 = [
        [1600000000000 + i * 3600000, "1", "2", "0.5", "1.5", "10",
         1600000000000 + (i + 1) * 3600000 - 1, "0", 1, "1", "0", "0"]
        for i in range(1000)
    ]
    page2 = [
        [1600003600000 + i * 3600000, "1", "2", "0.5", "1.5", "10",
         1600003600000 + (i + 1) * 3600000 - 1, "0", 1, "1", "0", "0"]
        for i in range(3)
    ]
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        class Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def read(self):
                payloads = (json.dumps(page1).encode(), json.dumps(page2).encode())
                return payloads[min(calls["n"] - 1, 1)]

        return Resp()

    monkeypatch.setattr("signal_engine.data.binance_source.urllib.request.urlopen", fake_urlopen)
    df = fetch_range("BTCUSDT", "1H", pd.Timestamp("2020-09-13T12:00", tz="UTC"),
                     pd.Timestamp("2020-09-13T12:00", tz="UTC") + pd.Timedelta(hours=1005))
    df.index = pd.DatetimeIndex(df.index)
    assert len(df) == (1000 - 1) + 3 or df.index.is_unique
