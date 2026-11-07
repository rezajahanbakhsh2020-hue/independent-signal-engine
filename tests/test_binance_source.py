from __future__ import annotations

import pandas as pd
import pytest

from signal_engine.data import binance_source
from signal_engine.data.binance_source import BinanceUnavailableError


def _kline(open_ms, *, close="100", open_price="100"):
    return [
        open_ms, open_price, "101", "99", close, "10",
        open_ms + 59_999, "1000", 1, "5", "500", "0",
    ]


def test_binance_range_refuses_silent_partial_result_at_request_cap(monkeypatch):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    start_ms = int(start.timestamp() * 1000)
    chunk = [_kline(start_ms + i * 60_000) for i in range(1000)]

    def fake_fetch(symbol, interval, page_start, page_end, timeout_seconds):
        return chunk, "api.binance.com"

    monkeypatch.setattr(binance_source, "_fetch_klines_with_host", fake_fetch)
    with pytest.raises(BinanceUnavailableError, match="incomplete range"):
        binance_source.fetch_range(
            "BTCUSDT", "1m", start, start + pd.Timedelta(minutes=2000),
            max_requests=1,
        )


def test_binance_reports_malformed_and_non_positive_rows(monkeypatch):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    start_ms = int(start.timestamp() * 1000)
    chunk = [
        _kline(start_ms),
        _kline(start_ms + 60_000, close="not-a-number"),
        _kline(start_ms + 120_000),
        _kline(start_ms + 180_000, open_price="0"),
    ]

    def fake_fetch(symbol, interval, page_start, page_end, timeout_seconds):
        return chunk, "api.binance.com"

    monkeypatch.setattr(binance_source, "_fetch_klines_with_host", fake_fetch)
    result = binance_source.fetch_range(
        "BTCUSDT", "1m", start, start + pd.Timedelta(minutes=4),
    )
    assert result.attrs["provider"] == "Binance"
    assert result.attrs["malformed_klines_skipped"] == 1
    assert result.attrs["validation_report"]["rejection_reasons"]["non_positive_ohlc_price"] == 1
    assert len(result) == 2
