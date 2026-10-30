"""Public Binance market-data adapter for Binance USDT spot symbols.

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

from signal_engine.constants import BINANCE_SYMBOLS, TIMEFRAME_TO_BINANCE, validate_crypto_symbol
from signal_engine.data.validate import DataValidationError, validate_ohlcv

# Binance documents these equivalent Spot REST base endpoints. Try the
# primary and GCP routes, then the regional api1-api4 hosts; the market-data-
# only host remains the final fallback. Never substitute another data vendor.
BASE_URLS = (
    "https://api.binance.com/api/v3/klines",
    "https://api-gcp.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api4.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
)
MAX_LIMIT = 1000  # provider limit per request


class BinanceUnavailableError(RuntimeError):
    """Raised when the public API cannot be reached or returns bad data."""


def _headers() -> dict[str, str]:
    return {"Accept": "application/json", "User-Agent": "signal-engine/1.0 + research"}


def _fetch_klines_with_host(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_seconds: int,
) -> tuple[list[list], str]:
    """Fetch one page, falling back only across official public Binance hosts."""
    params = (
        f"?symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={MAX_LIMIT}"
    )
    errors: list[str] = []
    for base_url in BASE_URLS:
        host = base_url.split("/")[2]
        try:
            req = urllib.request.Request(base_url + params, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                if resp.status != 200:
                    errors.append(f"{host}: HTTP {resp.status}")
                    continue
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{host}: {exc}")
            continue
        if not isinstance(payload, list):
            errors.append(f"{host}: unexpected payload shape")
            continue
        return payload, host
    raise BinanceUnavailableError(
        "All official Binance public endpoints failed: " + "; ".join(errors)
    )


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_seconds: int = 30,
) -> list[list]:
    """Fetch one page of klines. Raises BinanceUnavailableError on failure."""
    try:
        symbol = validate_crypto_symbol(symbol)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    if interval not in TIMEFRAME_TO_BINANCE.values():
        raise DataValidationError(f"Unsupported interval {interval!r}")
    payload, _host = _fetch_klines_with_host(
        symbol, interval, start_ms, end_ms, timeout_seconds
    )
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
    from signal_engine.constants import validate_timeframe

    try:
        market = validate_crypto_symbol(symbol)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    tf = validate_timeframe(timeframe)

    def as_utc(value: pd.Timestamp, label: str) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise DataValidationError(f"{label} must be a valid timestamp")
        if stamp.tzinfo is None:
            return stamp.tz_localize("UTC")
        return stamp.tz_convert("UTC")

    start_utc = as_utc(start_utc, "start_utc")
    end_utc = as_utc(end_utc, "end_utc")
    if start_utc >= end_utc:
        raise DataValidationError("start_utc must be earlier than end_utc")
    symbol = BINANCE_SYMBOLS.get(market, market)
    interval = TIMEFRAME_TO_BINANCE[tf]
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    rows: list[list] = []
    fetched_expected_span = 0
    hosts_used: set[str] = set()
    for _ in range(max_requests):
        chunk, host = _fetch_klines_with_host(symbol, interval, start_ms, end_ms, timeout_seconds)
        hosts_used.add(host)
        if len(hosts_used) > 1:
            raise BinanceUnavailableError(
                "Official Binance hosts changed during pagination; refusing to mix sources: "
                + ", ".join(sorted(hosts_used))
            )
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
    if not raw.empty:
        raw = raw.loc[
            (raw["timestamp"] >= start_utc) & (raw["timestamp"] < end_utc)
        ].reset_index(drop=True)
    if raw.empty:
        raise BinanceUnavailableError(
            f"No klines within requested UTC range for {market} {tf}"
        )
    df, _rep = validate_ohlcv(raw, strict_gap_policy=True)

    retrieval_time = datetime.now(timezone.utc).isoformat()
    df.attrs["source"] = f"public:binance-api-v3:{next(iter(hosts_used))}"
    df.attrs["symbol_mapping"] = f"{market}->{symbol}"
    df.attrs["timeframe"] = tf
    df.attrs["retrieval_time_utc"] = retrieval_time
    df.attrs["coverage_start_utc"] = df.index[0].isoformat()
    df.attrs["coverage_end_utc"] = df.index[-1].isoformat()
    df.attrs["timezone"] = "UTC"
    return df
