"""Explicit single-provider USDT-margined perpetual OHLCV adapters.

This module never falls back to Spot or to another exchange. Provider-specific
symbol formats and timestamp units are explicit; only completed trade-price
candles are returned. Mark/index candles are deliberately not substituted.
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

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1H": 3600, "4H": 14400, "1D": 86400,
}
BITGET_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1H", "4H": "4H", "1D": "1D",
}
KUCOIN_GRANULARITY = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "4H": 240, "1D": 1440,
}
HTX_PERIOD = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "60min", "4H": "4hour", "1D": "1day",
}
MEXC_INTERVAL = {
    "1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
    "1H": "Min60", "4H": "Hour4", "1D": "Day1",
}
PROVIDERS = ("Bitget", "KuCoin", "HTX", "MEXC")
MAX_PAGES = 300
MAX_PAGE_CANDLES = {"Bitget": 200, "KuCoin": 500, "HTX": 2000, "MEXC": 2000}


class PerpetualUnavailableError(RuntimeError):
    """The selected perpetual provider did not return valid candles."""


def _utc(value: pd.Timestamp, field: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise DataValidationError(f"{field} must be a valid timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _json(url: str, timeout: float) -> object:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "signal-engine/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _url(base: str, params: dict) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def _contract_symbol(provider: str, symbol: str) -> str:
    base = symbol[:-4]
    if provider == "Bitget":
        return symbol
    if provider == "KuCoin":
        # KuCoin names the BTC perpetual underlying XBT.
        return ("XBT" if base == "BTC" else base) + "USDTM"
    if provider == "HTX":
        return f"{base}-USDT"
    if provider == "MEXC":
        return f"{base}_USDT"
    raise ValueError(f"Unsupported perpetual provider: {provider}")


def _frame(rows: list[dict], provider: str, symbol: str, contract: str,
           timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not rows:
        raise PerpetualUnavailableError(
            f"{provider} returned no {timeframe} perpetual trade candles for {contract}"
        )
    raw = pd.DataFrame(rows)
    frame, report = validate_ohlcv(raw, strict_gap_policy=True)
    step = TIMEFRAME_SECONDS[timeframe]
    grid_ns = step * 1_000_000_000
    aligned = (frame.index.as_unit("ns").asi8 % grid_ns) == 0
    frame = frame.loc[
        aligned & (frame.index >= start.floor(f"{step}s")) & (frame.index < end)
    ]
    now = pd.Timestamp.now(tz="UTC")
    frame = frame.loc[(frame.index + pd.Timedelta(seconds=step)) <= now]
    if frame.empty:
        raise PerpetualUnavailableError(
            f"{provider} returned no completed, aligned perpetual candles for {contract}"
        )
    frame.attrs.update({
        "market_type": "PERPETUAL",
        "provider": provider,
        "contract_symbol": contract,
        "source": f"public:{provider.lower()}-usdt-perpetual",
        "symbol_mapping": f"{symbol}->{contract}",
        "timeframe": timeframe,
        "provider_interval": {
            "Bitget": BITGET_INTERVAL,
            "KuCoin": KUCOIN_GRANULARITY,
            "HTX": HTX_PERIOD,
            "MEXC": MEXC_INTERVAL,
        }[provider][timeframe],
        "timestamp_unit": {
            "Bitget": "milliseconds",
            "KuCoin": "milliseconds",
            "HTX": "seconds",
            "MEXC": "seconds",
        }[provider],
        "retrieval_time_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_start_utc": frame.index[0].isoformat(),
        "coverage_end_utc": frame.index[-1].isoformat(),
        "timezone": "UTC",
        "closed_candles_only": True,
        "pagination_truncated": False,
        "validation_report": report.as_dict(),
        "gap_count": len(report.gaps),
    })
    return frame


def _bitget(symbol: str, tf: str, start: pd.Timestamp, end: pd.Timestamp,
            timeout: float) -> pd.DataFrame:
    contract = _contract_symbol("Bitget", symbol)
    step = TIMEFRAME_SECONDS[tf]
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    page_span = step * 199 * 1000
    rows: list[dict] = []
    cursor = start_ms
    for _ in range(MAX_PAGES):
        if cursor >= end_ms:
            break
        page_end = min(end_ms, cursor + page_span)
        payload = _json(_url("https://api.bitget.com/api/v2/mix/market/history-candles", {
            "symbol": contract, "productType": "USDT-FUTURES",
            "granularity": BITGET_INTERVAL[tf], "startTime": cursor,
            "endTime": page_end, "limit": 200,
        }), timeout)
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise RuntimeError(f"unexpected Bitget futures response: {str(payload)[:240]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed Bitget futures candle list")
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed Bitget futures candle")
            stamp = pd.Timestamp(int(item[0]), unit="ms", tz="UTC")
            if cursor <= int(item[0]) < page_end:
                rows.append({"timestamp": stamp, "open": float(item[1]),
                    "high": float(item[2]), "low": float(item[3]),
                    "close": float(item[4]), "volume": float(item[5])})
        cursor = page_end
        if cursor < end_ms:
            time.sleep(0.06)
    else:
        raise RuntimeError("Bitget perpetual pagination limit reached")
    return _frame(rows, "Bitget", symbol, contract, tf, start, end)


def _kucoin(symbol: str, tf: str, start: pd.Timestamp, end: pd.Timestamp,
            timeout: float) -> pd.DataFrame:
    contract = _contract_symbol("KuCoin", symbol)
    step = TIMEFRAME_SECONDS[tf]
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    page_span = step * 499 * 1000
    rows: list[dict] = []
    cursor = start_ms
    for _ in range(MAX_PAGES):
        if cursor >= end_ms:
            break
        page_end = min(end_ms, cursor + page_span)
        payload = _json(_url("https://api-futures.kucoin.com/api/v1/kline/query", {
            "symbol": contract, "granularity": KUCOIN_GRANULARITY[tf],
            "from": cursor, "to": page_end,
        }), timeout)
        if not isinstance(payload, dict) or payload.get("code") != "200000":
            raise RuntimeError(f"unexpected KuCoin futures response: {str(payload)[:240]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed KuCoin futures candle list")
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                raise RuntimeError("malformed KuCoin futures candle")
            stamp_ms = int(item[0])
            if cursor <= stamp_ms < page_end:
                rows.append({"timestamp": pd.Timestamp(stamp_ms, unit="ms", tz="UTC"),
                    "open": float(item[1]), "high": float(item[2]),
                    "low": float(item[3]), "close": float(item[4]),
                    "volume": float(item[5])})
        cursor = page_end
        if cursor < end_ms:
            time.sleep(0.06)
    else:
        raise RuntimeError("KuCoin perpetual pagination limit reached")
    return _frame(rows, "KuCoin", symbol, contract, tf, start, end)


def _htx(symbol: str, tf: str, start: pd.Timestamp, end: pd.Timestamp,
         timeout: float) -> pd.DataFrame:
    contract = _contract_symbol("HTX", symbol)
    step = TIMEFRAME_SECONDS[tf]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    page_span = step * 1999
    rows: list[dict] = []
    cursor = start_s
    for _ in range(MAX_PAGES):
        if cursor >= end_s:
            break
        page_end = min(end_s, cursor + page_span)
        payload = _json(_url("https://api.hbdm.com/linear-swap-ex/market/history/kline", {
            "contract_code": contract, "period": HTX_PERIOD[tf],
            "size": 2000, "from": cursor, "to": page_end,
        }), timeout)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError(f"unexpected HTX USDT-swap response: {str(payload)[:240]}")
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise RuntimeError("malformed HTX USDT-swap candle list")
        for item in batch:
            if not isinstance(item, dict):
                raise RuntimeError("malformed HTX USDT-swap candle")
            stamp_s = int(item["id"])
            if cursor <= stamp_s < page_end:
                rows.append({"timestamp": pd.Timestamp(stamp_s, unit="s", tz="UTC"),
                    "open": float(item["open"]), "high": float(item["high"]),
                    "low": float(item["low"]), "close": float(item["close"]),
                    "volume": float(item.get("amount", item.get("vol", 0)))})
        cursor = page_end
        if cursor < end_s:
            time.sleep(0.06)
    else:
        raise RuntimeError("HTX perpetual pagination limit reached")
    return _frame(rows, "HTX", symbol, contract, tf, start, end)


def _mexc(symbol: str, tf: str, start: pd.Timestamp, end: pd.Timestamp,
          timeout: float) -> pd.DataFrame:
    contract = _contract_symbol("MEXC", symbol)
    step = TIMEFRAME_SECONDS[tf]
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    page_span = step * 1999
    rows: list[dict] = []
    cursor = start_s
    for _ in range(MAX_PAGES):
        if cursor >= end_s:
            break
        page_end = min(end_s, cursor + page_span)
        payload = _json(_url(
            f"https://api.mexc.com/api/v1/contract/kline/{urllib.parse.quote(contract, safe='')}",
            {"interval": MEXC_INTERVAL[tf], "start": cursor, "end": page_end},
        ), timeout)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RuntimeError(f"unexpected MEXC contract response: {str(payload)[:240]}")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise RuntimeError("malformed MEXC contract candle data")
        times = data.get("time") or []
        columns = [data.get(key) or [] for key in ("open", "high", "low", "close", "vol")]
        if any(len(col) != len(times) for col in columns):
            raise RuntimeError("MEXC contract candle columns have inconsistent lengths")
        for i, stamp in enumerate(times):
            stamp_s = int(stamp)
            if cursor <= stamp_s < page_end:
                rows.append({"timestamp": pd.Timestamp(stamp_s, unit="s", tz="UTC"),
                    "open": float(columns[0][i]), "high": float(columns[1][i]),
                    "low": float(columns[2][i]), "close": float(columns[3][i]),
                    "volume": float(columns[4][i])})
        cursor = page_end
        if cursor < end_s:
            time.sleep(0.06)
    else:
        raise RuntimeError("MEXC perpetual pagination limit reached")
    return _frame(rows, "MEXC", symbol, contract, tf, start, end)



def list_symbols(provider: str, timeout_seconds: float = 15.0) -> list[str]:
    """Return live, provider-specific USDT perpetual contracts as canonical BASEUSDT symbols.

    This is discovery only: the selected provider remains the sole source, and
    an unavailable discovery endpoint fails closed rather than using a static
    cross-exchange list or silently substituting BTC.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
    if timeout_seconds <= 0:
        raise DataValidationError("timeout_seconds must be positive")

    if provider == "Bitget":
        payload = _json(_url("https://api.bitget.com/api/v2/mix/market/contracts", {
            "productType": "USDT-FUTURES",
        }), timeout_seconds)
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise PerpetualUnavailableError("Bitget contract-list response was invalid")
        rows = payload.get("data") or []
        symbols = [
            str(row.get("symbol", "")).upper()
            for row in rows if isinstance(row, dict)
            and row.get("quoteCoin") == "USDT"
            and row.get("symbolType") == "perpetual"
            and row.get("symbolStatus") == "normal"
        ]
    elif provider == "KuCoin":
        payload = _json("https://api-futures.kucoin.com/api/v1/contracts/active", timeout_seconds)
        if not isinstance(payload, dict) or payload.get("code") != "200000":
            raise PerpetualUnavailableError("KuCoin contract-list response was invalid")
        rows = payload.get("data") or []
        symbols = [
            ("BTC" if str(row.get("baseCurrency", "")).upper() == "XBT"
             else str(row.get("baseCurrency", "")).upper()) + "USDT"
            for row in rows if isinstance(row, dict)
            and row.get("quoteCurrency") == "USDT"
            and row.get("settleCurrency") == "USDT"
            and row.get("status") == "Open"
            and row.get("expireDate") is None
        ]
    elif provider == "HTX":
        payload = _json("https://api.hbdm.com/linear-swap-api/v1/swap_contract_info", timeout_seconds)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise PerpetualUnavailableError("HTX contract-list response was invalid")
        rows = payload.get("data") or []
        symbols = [
            str(row.get("contract_code", "")).upper().replace("-", "")
            for row in rows if isinstance(row, dict)
            and str(row.get("contract_code", "")).upper().endswith("-USDT")
            and row.get("contract_status") == 1
        ]
    else:
        payload = _json("https://contract.mexc.com/api/v1/contract/detail", timeout_seconds)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise PerpetualUnavailableError("MEXC contract-list response was invalid")
        rows = payload.get("data") or []
        symbols = [
            str(row.get("symbol", "")).upper().replace("_", "")
            for row in rows if isinstance(row, dict)
            and row.get("quoteCoin") == "USDT"
            and row.get("settleCoin") == "USDT"
        ]

    import re
    valid = sorted({
        symbol for symbol in symbols
        if re.fullmatch(r"[A-Z0-9]{2,20}USDT", symbol)
    })
    if not valid:
        raise PerpetualUnavailableError(
            f"{provider} returned no active USDT perpetual contracts"
        )
    # Keep BTC easy to find without making it a fallback or requiring it to exist.
    return (["BTCUSDT"] if "BTCUSDT" in valid else []) + [
        symbol for symbol in valid if symbol != "BTCUSDT"
    ]

_FETCHERS = {"Bitget": _bitget, "KuCoin": _kucoin, "HTX": _htx, "MEXC": _mexc}


def fetch_range(provider: str, symbol: str, timeframe: str,
                start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                timeout_seconds: float = 15.0) -> pd.DataFrame:
    """Fetch one explicitly selected provider; never cross-provider or Spot fallback."""
    if provider not in _FETCHERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
    market = validate_crypto_symbol(symbol)
    tf = validate_timeframe(timeframe)
    start, end = _utc(start_utc, "start_utc"), _utc(end_utc, "end_utc")
    if start >= end:
        raise DataValidationError("start_utc must be earlier than end_utc")
    if timeout_seconds <= 0:
        raise DataValidationError("timeout_seconds must be positive")
    step = TIMEFRAME_SECONDS[tf]
    start = pd.Timestamp((int(start.timestamp()) // step) * step, unit="s", tz="UTC")
    try:
        return _FETCHERS[provider](market, tf, start, end, timeout_seconds)
    except Exception as exc:
        if isinstance(exc, (ValueError, DataValidationError)):
            raise
        raise PerpetualUnavailableError(
            f"{provider} PERPETUAL {market} {tf} unavailable: {type(exc).__name__}: {exc}"
        ) from exc
