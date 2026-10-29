"""Public BiQuote OHLC adapter for XAUUSD.

BiQuote exposes anonymous OHLC endpoints. The unfinished bar is excluded.
For gold/CFD feeds, the provider's traded-volume field may be zero; this
adapter uses tickVolume as the engine's volume field and records that
semantics explicitly. Tick count is not exchange-traded volume.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from signal_engine.constants import TIMEFRAMES, TIMEFRAME_TO_PANDAS, validate_market, validate_timeframe
from signal_engine.data.validate import DataValidationError, validate_ohlcv

BASE_URL = "https://biquote.io/api/XAUUSD/ohlc"
MAX_LIMIT = 1000
INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}


class BiQuoteUnavailableError(RuntimeError):
    """Raised when BiQuote cannot provide usable XAUUSD OHLC bars."""


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso_z(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _get_page(
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout_seconds: int,
) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "interval": interval,
            "limit": MAX_LIMIT,
            "from": _iso_z(start),
            "to": _iso_z(end),
        }
    )
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "signal-engine/1.0 + research"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                raise BiQuoteUnavailableError(f"HTTP {resp.status} from BiQuote")
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BiQuoteUnavailableError(f"BiQuote request failed: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("symbol") != "XAUUSD":
        raise BiQuoteUnavailableError("Unexpected BiQuote response envelope or symbol")
    bars = payload.get("bars")
    if not isinstance(bars, list):
        raise BiQuoteUnavailableError("BiQuote response has no bars list")
    if any(not isinstance(bar, dict) for bar in bars):
        raise BiQuoteUnavailableError("BiQuote response contains a malformed bar")
    return bars


def fetch_range(
    symbol: str,
    timeframe: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    timeout_seconds: int = 30,
    max_requests: int = 250,
) -> pd.DataFrame:
    """Fetch closed XAUUSD candles over a requested UTC range.

    Bars are paged backwards because BiQuote returns newest-first. A partial
    range is never padded: actual coverage and any detected gaps are recorded.
    """
    market = validate_market(symbol)
    if market != "XAUUSD":
        raise DataValidationError("BiQuote adapter supports XAUUSD only; it does not map BTCUSD to BTCUSDT")
    tf = validate_timeframe(timeframe)
    interval = INTERVALS[tf]
    start = _utc_timestamp(start_utc)
    end = _utc_timestamp(end_utc)
    if end <= start:
        raise DataValidationError("end_utc must be later than start_utc")
    if max_requests < 1:
        raise DataValidationError("max_requests must be >= 1")

    cursor_end = end
    records: dict[pd.Timestamp, dict] = {}
    for page_number in range(max_requests):
        bars = _get_page(interval, start, cursor_end, timeout_seconds)
        if not bars:
            break
        parsed: list[tuple[pd.Timestamp, dict]] = []
        for bar in bars:
            # Do not include the live, unfinished candle.
            if bar.get("isOpen") is True:
                continue
            ts = pd.to_datetime(bar.get("openTime"), utc=True, errors="coerce")
            if pd.isna(ts):
                raise BiQuoteUnavailableError("BiQuote returned a bar with an invalid openTime")
            ts = pd.Timestamp(ts)
            parsed.append((ts, bar))
            if start <= ts < end:
                try:
                    # FX/CFD traded volume may be zero; tickVolume is a tick
                    # count, not centralized exchange volume.
                    volume = bar.get("tickVolume", bar.get("volume", 0))
                    records[ts] = {
                        "timestamp": ts,
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(volume),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise BiQuoteUnavailableError(f"Malformed OHLC values at {ts.isoformat()}") from exc

        if len(bars) < MAX_LIMIT or not parsed:
            break
        oldest = min(ts for ts, _bar in parsed)
        if oldest <= start:
            break
        cursor_end = oldest - pd.Timedelta(milliseconds=1)
        if page_number + 1 < max_requests:
            time.sleep(0.1)

    if not records:
        raise BiQuoteUnavailableError(f"No closed XAUUSD bars retrieved for {tf} in requested range")
    raw = pd.DataFrame(list(records.values()))
    df, report = validate_ohlcv(raw, strict_gap_policy=True)
    df.attrs["source"] = "public:biquote-ohlc"
    df.attrs["provider"] = "BiQuote"
    df.attrs["symbol_mapping"] = "XAUUSD->XAUUSD"
    df.attrs["timeframe"] = tf
    df.attrs["provider_interval"] = interval
    df.attrs["retrieval_time_utc"] = datetime.now(timezone.utc).isoformat()
    df.attrs["requested_start_utc"] = start.isoformat()
    df.attrs["requested_end_utc"] = end.isoformat()
    df.attrs["coverage_start_utc"] = df.index[0].isoformat()
    df.attrs["coverage_end_utc"] = df.index[-1].isoformat()
    df.attrs["timezone"] = "UTC"
    df.attrs["volume_semantics"] = "tickVolume tick count; not exchange-traded volume"
    df.attrs["validation_report"] = report.as_dict()
    return df
