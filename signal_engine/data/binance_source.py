"""Public Binance market-data adapter (BTCUSDT / ETHUSDT only).

Uses the public REST endpoint `api.binance.com` via stdlib urllib so no
extra dependency is required and no credentials exist anywhere.

Truth rules:
- Uses ONLY data actually retrieved.  A network failure raises
  BinanceUnavailableError; the caller must NOT substitute fabricated data.
- Adds provenance metadata: retrieval time (UTC), symbol mapping, source
  URL, and full coverage [first_ts_utc, last_ts_utc].
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from signal_engine.constants import BINANCE_SYMBOLS, TIMEFRAME_TO_BINANCE
from signal_engine.data.validate import DataValidationError, validate_ohlcv

BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # provider limit per request


class BinanceUnavailableError(RuntimeError):
    """Raised when the public API cannot be reached or returns bad data."""


def _headers() -> dict[str, str]:
    return {"Accept": "application/json", "User-Agent": "signal-engine/1.0 + research"}


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_seconds: int = 30,
) -> list[list]:
    """Fetch one page of klines.  Raises BinanceUnavailableError on failure."""
    if symbol not in BINANCE_SYMBOLS:
        raise DataValidationError(
            f"Symbol {symbol!r} not supported by the Binance adapter; "
            f"supported: {', '.join(BINANCE_SYMBOLS)}"
        )
    if interval not in TIMEFRAME_TO_BINANCE.values():
        raise DataValidationError(f"Unsupported interval {interval!r}")
    params = (
        f"?symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={MAX_LIMIT}"
    )
    url = BASE_URL + params
    try:
        req = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                raise BinanceUnavailableError(f"HTTP {resp.status} from Binance")
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BinanceUnavailableError(f"Binance request failed: {exc}") from exc
    if not isinstance(payload, list):
        raise BinanceUnavailableError("Unexpected Binance payload shape")
    return payload


def fetch_range(
    symbol: str,
    timeframe: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    timeout_seconds: int = 30,
    max_requests: int = 250,
) -> pd.DataFrame:
    """Fetch a full [start, end) range by paging, without inventing candles."""
    from signal_engine.constants import validate_market, validate_timeframe

    market = validate_market(symbol)
    tf = validate_timeframe(timeframe)
    symbol = BINANCE_SYMBOLS[market]
    interval = TIMEFRAME_TO_BINANCE[tf]
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    rows: list[list] = []
    fetched_expected_span = 0
    for _ in range(max_requests):
        chunk = fetch_klines(symbol, interval, start_ms, end_ms, timeout_seconds)
        if not chunk:
            break
        rows.extend(chunk)
        fetched_expected_span += len(chunk)
        last_close_ms = int(chunk[-1][6])
        if last_close_ms >= end_ms - 1 or len(chunk) < MAX_LIMIT:
            break
        start_ms = last_close_ms + 1
        time.sleep(0.25)  # stay polite under provider rate limits
    if not rows:
        raise BinanceUnavailableError(f"No klines retrieved from Binance for {market} {tf}")

    records = []
    for k in rows:
        open_ms = int(k[0])
        try:
            o, h, l_, c = (float(k[i]) for i in (1, 2, 3, 4))
            vol = float(k[5])
        except (TypeError, ValueError):
            continue  # malformed kline, documented by validation counts later
        if not (o > 0 and h > 0 and l_ > 0 and c > 0):
            continue
        records.append(
            {
                "timestamp": pd.Timestamp(open_ms, unit="ms", tz="UTC"),
                "open": o,
                "high": h,
                "low": l_,
                "close": c,
                "volume": vol,
            }
        )
    raw = pd.DataFrame(records)
    df, _rep = validate_ohlcv(raw, strict_gap_policy=True)

    retrieval_time = datetime.now(timezone.utc).isoformat()
    df.attrs["source"] = "public:binance-api-v3"
    df.attrs["symbol_mapping"] = f"{market}->{symbol}"
    df.attrs["timeframe"] = tf
    df.attrs["retrieval_time_utc"] = retrieval_time
    df.attrs["coverage_start_utc"] = df.index[0].isoformat()
    df.attrs["coverage_end_utc"] = df.index[-1].isoformat()
    df.attrs["timezone"] = "UTC"
    return df
