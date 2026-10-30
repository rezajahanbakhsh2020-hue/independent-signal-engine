"""Fail-closed public crypto OHLCV provider rotation.

Provider priority is explicit and deterministic:
KuCoin -> Bitget -> HTX -> MEXC.

Each provider must independently return the complete requested candle grid.
Data from different providers is never mixed. Every HTTP request uses the
same bounded timeout; no credentials or trading endpoints are used.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from signal_engine.constants import validate_crypto_symbol, validate_timeframe
from signal_engine.data.validate import DataValidationError, validate_ohlcv

DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_REQUESTS = 300
LIMIT = 1000

DURATIONS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1H": 3600, "4H": 14400, "1D": 86400,
}
KUCOIN_INTERVALS = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "1hour", "4H": "4hour", "1D": "1day",
}
BITGET_GRANULARITIES = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "1h", "4H": "4h", "1D": "1day",
}
MEXC_INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "60m", "4H": "4h", "1D": "1d",
}
HTX_PERIODS = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "60min", "4H": "4hour", "1D": "1day",
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
        url,
        headers={"Accept": "application/json", "User-Agent": "signal-engine/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _query(base: str, params: dict) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def _validated_frame(
    rows: list[dict],
    symbol: str,
    timeframe: str,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("provider returned no candles in requested range")

    raw = pd.DataFrame(rows)
    frame, report = validate_ohlcv(raw, strict_gap_policy=True)

    # strict_gap_policy in the shared validator records gaps but does not
    # reject them. Provider failover must reject gaps and incomplete ranges.
    if report.gaps:
        raise DataValidationError(
            f"{source} returned {len(report.gaps)} candle gap(s)", report=report
        )
    if report.n_duplicate_timestamps_removed:
        raise DataValidationError(
            f"{source} returned duplicate candle timestamps", report=report
        )

    step = DURATIONS[timeframe]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    first_open = (start_s // step) * step
    last_closed_open = (end_s // step) * step - step
    if last_closed_open < first_open:
        raise DataValidationError("requested range contains no completed candle grid")

    expected = pd.date_range(
        pd.Timestamp(first_open, unit="s", tz="UTC"),
        pd.Timestamp(last_closed_open, unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=step),
    )
    actual = frame.index
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise DataValidationError(
            f"{source} incomplete candle coverage: expected={len(expected)}, "
            f"received={len(actual)}, missing={len(missing)}, extra={len(extra)}"
        )

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
        "complete_candle_grid": True,
    })
    return frame


def _kucoin(
    symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, timeout: float
) -> pd.DataFrame:
    interval = KUCOIN_INTERVALS[timeframe]
    step = DURATIONS[timeframe]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    page_span = step * 1200  # below KuCoin's 1500-candle response cap
    rows: list[dict] = []
    cursor = start_s

    for _ in range(MAX_REQUESTS):
        if cursor >= end_s:
            break
        page_end = min(end_s, cursor + page_span)
        query = _query("https://api.kucoin.com/api/v1/market/candles", {
            "symbol": symbol[:-4] + "-USDT",
            "type": interval,
            "startAt": cursor,
            "endAt": page_end,
        })
        payload = _get_json(query, timeout)
        if not isinstance(payload, dict) or payload.get("code") != "200000":
            raise RuntimeError(f"unexpected KuCoin response: {str(payload)[:220]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed KuCoin candle list")
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed KuCoin candle")
            stamp = int(item[0])
            if cursor <= stamp < page_end:
                # KuCoin: time, open, close, high, low, volume, turnover
                rows.append({
                    "timestamp": pd.Timestamp(stamp, unit="s", tz="UTC"),
                    "open": float(item[1]), "high": float(item[3]),
                    "low": float(item[4]), "close": float(item[2]),
                    "volume": float(item[5]),
                })
        cursor = page_end
    else:
        raise RuntimeError("KuCoin pagination limit reached")
    return _validated_frame(rows, symbol, timeframe, "public:kucoin-spot", start, end)


def _bitget(
    symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, timeout: float
) -> pd.DataFrame:
    granularity = BITGET_GRANULARITIES[timeframe]
    step = DURATIONS[timeframe]
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    page_span = step * 999 * 1000
    rows: list[dict] = []
    cursor = start_ms

    for _ in range(MAX_REQUESTS):
        if cursor >= end_ms:
            break
        page_end = min(end_ms, cursor + page_span)
        query = _query("https://api.bitget.com/api/v2/spot/market/candles", {
            "symbol": symbol,
            "granularity": granularity,
            "startTime": cursor,
            "endTime": page_end,
            "limit": LIMIT,
        })
        payload = _get_json(query, timeout)
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise RuntimeError(f"unexpected Bitget response: {str(payload)[:220]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed Bitget candle list")
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed Bitget candle")
            stamp_ms = int(item[0])
            if cursor <= stamp_ms < page_end:
                rows.append({
                    "timestamp": pd.Timestamp(stamp_ms, unit="ms", tz="UTC"),
                    "open": float(item[1]), "high": float(item[2]),
                    "low": float(item[3]), "close": float(item[4]),
                    "volume": float(item[5]),
                })
        cursor = page_end
    else:
        raise RuntimeError("Bitget pagination limit reached")
    return _validated_frame(rows, symbol, timeframe, "public:bitget-spot-v2", start, end)


def _htx(
    symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, timeout: float
) -> pd.DataFrame:
    # HTX history/kline exposes only the most recent N bars and has no
    # start/end pagination. Completeness validation rejects lookbacks beyond
    # the actual returned window instead of silently accepting partial data.
    query = _query("https://api.huobi.pro/market/history/kline", {
        "symbol": symbol.lower(),
        "period": HTX_PERIODS[timeframe],
        "size": 2000,
    })
    payload = _get_json(query, timeout)
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"unexpected HTX response: {str(payload)[:220]}")
    batch = payload.get("data") or []
    if not isinstance(batch, list):
        raise RuntimeError("malformed HTX candle list")
    rows = []
    for item in batch:
        if not isinstance(item, dict):
            raise RuntimeError("malformed HTX candle")
        stamp = int(item["id"])
        rows.append({
            "timestamp": pd.Timestamp(stamp, unit="s", tz="UTC"),
            "open": float(item["open"]), "high": float(item["high"]),
            "low": float(item["low"]), "close": float(item["close"]),
            "volume": float(item["amount"]),
        })
    return _validated_frame(rows, symbol, timeframe, "public:htx-spot", start, end)


def _mexc(
    symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, timeout: float
) -> pd.DataFrame:
    interval = MEXC_INTERVALS[timeframe]
    step = DURATIONS[timeframe]
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    page_span = step * 999 * 1000
    rows: list[dict] = []
    cursor = start_ms

    for _ in range(MAX_REQUESTS):
        if cursor >= end_ms:
            break
        page_end = min(end_ms, cursor + page_span)
        query = _query("https://api.mexc.com/api/v3/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": page_end,
            "limit": LIMIT,
        })
        payload = _get_json(query, timeout)
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected MEXC response: {str(payload)[:220]}")
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed MEXC candle")
            stamp_ms = int(item[0])
            if cursor <= stamp_ms < page_end:
                rows.append({
                    "timestamp": pd.Timestamp(stamp_ms, unit="ms", tz="UTC"),
                    "open": float(item[1]), "high": float(item[2]),
                    "low": float(item[3]), "close": float(item[4]),
                    "volume": float(item[5]),
                })
        cursor = page_end
    else:
        raise RuntimeError("MEXC pagination limit reached")
    return _validated_frame(rows, symbol, timeframe, "public:mexc-spot-v3", start, end)


def fetch_range(
    symbol: str,
    timeframe: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch a complete range from the first available provider; never mix."""
    market = validate_crypto_symbol(symbol)
    tf = validate_timeframe(timeframe)
    start, end = _utc(start_utc, "start_utc"), _utc(end_utc, "end_utc")
    step = DURATIONS[tf]
    start = pd.Timestamp((int(start.timestamp()) // step) * step, unit="s", tz="UTC")
    if start >= end:
        raise DataValidationError("start_utc must be earlier than end_utc")
    if timeout_seconds <= 0:
        raise DataValidationError("timeout_seconds must be positive")

    providers = (
        ("KuCoin", lambda: _kucoin(market, tf, start, end, timeout_seconds)),
        ("Bitget", lambda: _bitget(market, tf, start, end, timeout_seconds)),
        ("HTX", lambda: _htx(market, tf, start, end, timeout_seconds)),
        ("MEXC", lambda: _mexc(market, tf, start, end, timeout_seconds)),
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
        "All configured public crypto providers failed: " + " | ".join(errors)
    )
