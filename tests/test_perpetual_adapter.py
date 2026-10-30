from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from signal_engine.data import perpetual_source as source


@pytest.mark.parametrize(
    ("provider", "expected_contract"),
    [
        ("Bitget", "BTCUSDT"),
        ("KuCoin", "XBTUSDTM"),
        ("HTX", "BTC-USDT"),
        ("MEXC", "BTC_USDT"),
    ],
)
def test_provider_contract_symbol_mapping_is_explicit(provider, expected_contract):
    assert source._contract_symbol(provider, "BTCUSDT") == expected_contract


@pytest.mark.parametrize("provider", source.PROVIDERS)
def test_each_provider_uses_explicit_timestamp_unit_and_returns_closed_utc_candle(
    monkeypatch, provider
):
    start = pd.Timestamp("2025-02-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=2)
    seconds = int(start.timestamp())
    millis = seconds * 1000
    calls = []

    def fake_json(url, timeout):
        calls.append((url, timeout))
        if provider == "Bitget":
            return {"code": "00000", "data": [[str(millis), "100", "102", "99", "101", "12", "1200"], [str(millis + 60000), "101", "103", "100", "102", "13", "1300"]]}
        if provider == "KuCoin":
            return {"code": "200000", "data": [[millis, "100", "102", "99", "101", "12", "1200"], [millis + 60000, "101", "103", "100", "102", "13", "1300"]]}
        if provider == "HTX":
            return {"status": "ok", "data": [{
                "id": seconds, "open": "100", "high": "102", "low": "99",
                "close": "101", "vol": "12",
            }, {
                "id": seconds + 60, "open": "101", "high": "103", "low": "100",
                "close": "102", "vol": "13",
            }]}
        return {"success": True, "data": {
            "time": [seconds, seconds + 60], "open": ["100", "101"], "high": ["102", "103"],
            "low": ["99", "100"], "close": ["101", "102"], "vol": ["12", "13"],
        }}

    monkeypatch.setattr(source, "_json", fake_json)
    frame = source.fetch_range(provider, "BTCUSDT", "1m", start, end)

    assert len(calls) == 1
    assert len(frame) == 2
    assert frame.index[0] == start
    assert str(frame.index.tz) == "UTC"
    assert frame.iloc[0]["close"] == 101.0
    assert frame.attrs["market_type"] == "PERPETUAL"
    assert frame.attrs["provider"] == provider
    assert frame.attrs["closed_candles_only"] is True
    assert frame.attrs["timestamp_unit"] == (
        "milliseconds" if provider in {"Bitget", "KuCoin"} else "seconds"
    )
    query = parse_qs(urlparse(calls[0][0]).query)
    assert "spot" not in calls[0][0].lower()
    if provider == "Bitget":
        assert query["productType"] == ["USDT-FUTURES"]
        assert "/api/v2/mix/market/history-candles" in calls[0][0]
    elif provider == "KuCoin":
        assert "api-futures.kucoin.com" in calls[0][0]
    elif provider == "HTX":
        assert "linear-swap-ex/market/history/kline" in calls[0][0]
    else:
        assert "/api/v1/contract/kline/BTC_USDT" in calls[0][0]


def test_perpetual_provider_is_required_and_no_provider_fallback(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid provider must fail before network access")

    monkeypatch.setattr(source, "_json", unexpected)
    with pytest.raises(ValueError, match="provider must be one of"):
        source.fetch_range("binance", "BTCUSDT", "1m",
                           pd.Timestamp("2025-01-01", tz="UTC"),
                           pd.Timestamp("2025-01-01T00:01:00Z"))


def test_all_seven_timeframes_have_independent_provider_mappings():
    assert tuple(source.TIMEFRAME_SECONDS) == ("1m", "5m", "15m", "30m", "1H", "4H", "1D")
    for provider_map in (
        source.BITGET_INTERVAL, source.KUCOIN_GRANULARITY,
        source.HTX_PERIOD, source.MEXC_INTERVAL,
    ):
        assert tuple(provider_map) == tuple(source.TIMEFRAME_SECONDS)
        assert len(provider_map) == 7


def test_perpetual_page_is_separate_and_identifies_selected_exchange():
    from signal_engine.perpetual_dashboard import _page

    page = _page()
    for provider in source.PROVIDERS:
        assert provider in page
    for timeframe in source.TIMEFRAME_SECONDS:
        assert timeframe in page
    assert "PERPETUAL" in page
    assert "Mark Price" in page
    assert "Funding" in page
    assert '<select id="symbol" name="symbol"' in page
    assert "/api/perpetual/symbols?provider=" in page


def test_existing_dashboard_links_to_separate_perpetual_route():
    from signal_engine.dashboard import _page

    existing = _page()
    assert 'href="/perpetual"' in existing
    assert "ورود به صفحه مستقل Perpetual" in existing


@pytest.mark.parametrize(
    ("provider", "payload", "expected"),
    [
        ("Bitget", {"code": "00000", "data": [
            {"symbol": "BTCUSDT", "quoteCoin": "USDT", "symbolType": "perpetual", "symbolStatus": "normal"},
            {"symbol": "ETHUSDT", "quoteCoin": "USDT", "symbolType": "perpetual", "symbolStatus": "normal"},
            {"symbol": "SOLUSDT", "quoteCoin": "USDT", "symbolType": "delivery", "symbolStatus": "normal"},
            {"symbol": "XRPUSDT", "quoteCoin": "USDT", "symbolType": "perpetual", "symbolStatus": "maintain"},
        ]}, ["BTCUSDT", "ETHUSDT"]),
        ("KuCoin", {"code": "200000", "data": [
            {"baseCurrency": "XBT", "quoteCurrency": "USDT", "settleCurrency": "USDT", "status": "Open"},
            {"baseCurrency": "ETH", "quoteCurrency": "USDT", "settleCurrency": "USDT", "status": "Open"},
            {"baseCurrency": "SOL", "quoteCurrency": "USDT", "settleCurrency": "USDT", "status": "Done"},
        ]}, ["BTCUSDT", "ETHUSDT"]),
        ("HTX", {"status": "ok", "data": [
            {"contract_code": "BTC-USDT", "contract_status": 1},
            {"contract_code": "ETH-USDT", "contract_status": 1},
            {"contract_code": "SOL-USDT", "contract_status": 0},
            {"contract_code": "XRP-USD", "contract_status": 1},
        ]}, ["BTCUSDT", "ETHUSDT"]),
        ("MEXC", {"success": True, "data": [
            {"symbol": "BTC_USDT", "quoteCoin": "USDT", "settleCoin": "USDT"},
            {"symbol": "ETH_USDT", "quoteCoin": "USDT", "settleCoin": "USDT"},
            {"symbol": "SOL_USDC", "quoteCoin": "USDC", "settleCoin": "USDC"},
        ]}, ["BTCUSDT", "ETHUSDT"]),
    ],
)
def test_live_symbol_discovery_is_provider_specific_and_usdt_perpetual_only(
    monkeypatch, provider, payload, expected
):
    calls = []
    monkeypatch.setattr(source, "_json", lambda url, timeout: calls.append(url) or payload)
    assert source.list_symbols(provider) == expected
    assert len(calls) == 1
    assert "spot" not in calls[0].lower()


def test_symbol_discovery_rejects_unknown_provider_without_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unknown provider must fail before network access")

    monkeypatch.setattr(source, "_json", unexpected)
    with pytest.raises(ValueError, match="provider must be one of"):
        source.list_symbols("Binance")
