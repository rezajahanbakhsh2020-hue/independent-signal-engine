"""Fast, fail-closed public crypto OHLCV provider rotation.

Each provider must independently deliver and validate the complete requested
range. Candles from different exchanges are never mixed in one DataFrame.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from signal_engine.constants import validate_crypto_symbol, validate_timeframe
from signal_engine.data.validate import DataValidationError, validate_ohlcv

# Short per-request timeout keeps an unavailable provider from stalling rotation.
DEFAULT_TIMEOUT_SECONDS = 2.5
MAX_REQUESTS = 250
LIMIT = 1000
DURATIONS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1H": 3600, "4H": 14400, "1D": 86400,
}
BYBIT_INTERVALS = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1H": "60", "4H": "240", "1D": "D",
}
GATE_INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "4H": "4h", "1D": "1d",
}
KUCOIN_INTERVALS = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "1hour", "4H": "4hour", "1D": "1day",
}


class CryptoProvidersUnavailableError(RuntimeError):
    """Raised when no independent public crypto source can provide valid data."""


def _utc(value: pd.Timestamp, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise DataValidationError(f"{name} must be a valid timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _get_json(url: str, timeout: float) -> object:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "signal-engine/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _validated_frame(rows: list[dict], symbol: str, timeframe: str, source: str) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("provider returned no candles in requested range")
    raw = pd.DataFrame(rows)
    frame, report = validate_ohlcv(raw, strict_gap_policy=True)
    frame.attrs.update({
        "source": source,
        "provider": source.split(":")[1],
        "symbol_mapping": f"{symbol}->{symbol}",
        "timeframe": timeframe,
        "retrieval_time_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_start_utc": frame.index[0].isoformat(),
        "coverage_end_utc": frame.index[-1].isoformat(),
        "timezone": "UTC",
        "validation_report": report.as_dict(),
        "provider_failover": True,
    })
    return frame


def _bybit(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
           timeout: float) -> pd.DataFrame:
    interval = BYBIT_INTERVALS[timeframe]
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    cursor_end = end_ms - 1
    rows: list[dict] = []
    for _ in range(MAX_REQUESTS):
        query = urllib.parse.urlencode({
            "category": "spot", "symbol": symbol, "interval": interval,
            "start": start_ms, "end": cursor_end, "limit": LIMIT,
        })
        payload = _get_json("https://api.bybit.com/v5/market/kline?" + query, timeout)
        if not isinstance(payload, dict) or payload.get("retCode") != 0:
            raise RuntimeError(f"unexpected Bybit response: {str(payload)[:240]}")
        result = payload.get("result") or {}
        batch = result.get("list") or []
        if not batch:
            break
        timestamps = []
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed Bybit kline")
            stamp_ms = int(item[0])
            timestamps.append(stamp_ms)
            if start_ms <= stamp_ms < end_ms:
                rows.append({
                    "timestamp": pd.Timestamp(stamp_ms, unit="ms", tz="UTC"),
                    "open": float(item[1]), "high": float(item[2]),
                    "low": float(item[3]), "close": float(item[4]),
                    "volume": float(item[5]),
                })
        oldest = min(timestamps)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
    else:
        raise RuntimeError("Bybit pagination limit reached before full range")
    return _validated_frame(rows, symbol, timeframe, "public:bybit-v5-spot")


def _gate(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
          timeout: float) -> pd.DataFrame:
    interval = GATE_INTERVALS[timeframe]
    step = DURATIONS[timeframe]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    # Gate's from/to mode returns at most 1000 bars; use non-overlapping pages.
    page_span = step * 900
    rows: list[dict] = []
    cursor = start_s
    hosts = ("https://api.gateio.ws/api/v4", "https://api.gate.us/api/v4")
    while cursor < end_s:
        page_end = min(end_s, cursor + page_span)
        query = urllib.parse.urlencode({
            "currency_pair": symbol[:-4] + "_USDT",
            "interval": interval, "from": cursor, "to": page_end,
        })
        last_error = None
        payload = None
        for host in hosts:
            try:
                payload = _get_json(host + "/spot/candlesticks?" + query, timeout)
                break
            except Exception as exc:
                last_error = exc
        if payload is None:
            raise RuntimeError(f"Gate endpoints failed: {last_error}")
        if not isinstance(payload, list):
            raise RuntimeError("unexpected Gate candlestick response")
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed Gate candlestick")
            stamp_s = int(item[0])
            if cursor <= stamp_s < page_end:
                rows.append({
                    "timestamp": pd.Timestamp(stamp_s, unit="s", tz="UTC"),
                    "open": float(item[5]), "high": float(item[3]),
                    "low": float(item[4]), "close": float(item[2]),
                    "volume": float(item[1]),
                })
        cursor = page_end
    return _validated_frame(rows, symbol, timeframe, "public:gateio-spot-v4")


def _kucoin(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
            timeout: float) -> pd.DataFrame:
    interval = KUCOIN_INTERVALS[timeframe]
    step = DURATIONS[timeframe]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    # KuCoin caps a response at 1500 candles. Keep each request below that cap.
    page_span = step * 1200
    rows: list[dict] = []
    cursor = start_s
    while cursor < end_s:
        page_end = min(end_s, cursor + page_span)
        query = urllib.parse.urlencode({
            "symbol": symbol[:-4] + "-USDT", "type": interval,
            "startAt": cursor, "endAt": page_end,
        })
        payload = _get_json("https://api.kucoin.com/api/v1/market/candles?" + query, timeout)
        if not isinstance(payload, dict) or payload.get("code") != "200000":
            raise RuntimeError(f"unexpected KuCoin response: {str(payload)[:240]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed KuCoin candle list")
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed KuCoin candle")
            stamp_s = int(item[0])
            if cursor <= stamp_s < page_end:
                rows.append({
                    "timestamp": pd.Timestamp(stamp_s, unit="s", tz="UTC"),
                    "open": float(item[1]), "high": float(item[3]),
                    "low": float(item[4]), "close": float(item[2]),
                    "volume": float(item[5]),
                })
        cursor = page_end
    return _validated_frame(rows, symbol, timeframe, "public:kucoin-spot")


def fetch_range(
    symbol: str,
    timeframe: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Try independent public spot providers in order; never mix their candles."""
    market = validate_crypto_symbol(symbol)
    tf = validate_timeframe(timeframe)
    start, end = _utc(start_utc, "start_utc"), _utc(end_utc, "end_utc")
    if start >= end:
        raise DataValidationError("start_utc must be earlier than end_utc")

    providers = (
        ("Bybit", lambda: _bybit(market, tf, start, end, timeout_seconds)),
        ("Gate.io", lambda: _gate(market, tf, start, end, timeout_seconds)),
        ("KuCoin", lambda: _kucoin(market, tf, start, end, timeout_seconds)),
        ("Binance", lambda: _binance(market, tf, start, end, timeout_seconds)),
    )
    errors: list[str] = []
    for name, fetch in providers:
        try:
            frame = fetch()
            frame.attrs["provider_attempts"] = len(errors) + 1
            frame.attrs["provider_failures"] = tuple(errors)
            return frame
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise CryptoProvidersUnavailableError(
        "All independent public crypto providers failed: " + " | ".join(errors)
    )


def _binance(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
             timeout: float) -> pd.DataFrame:
    from signal_engine.data.binance_source import fetch_range as fetch_binance
    return fetch_binance(
        symbol, timeframe, start, end,
        timeout_seconds=timeout, max_requests=MAX_REQUESTS,
    )
