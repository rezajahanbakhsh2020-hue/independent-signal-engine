"""Canonical market identifiers, timeframes, and the 21-combination matrix.

The engine deliberately restricts itself to the canonical timeframe
identifiers listed below.  Any other identifier is rejected explicitly
rather than coerced.
"""

from __future__ import annotations

import re

MARKETS = ("XAUUSD", "BTCUSDT", "ETHUSDT")

TIMEFRAMES = ("1m", "5m", "15m", "30m", "1H", "4H", "1D")

# Canonical identifier -> pandas offset alias (unambiguous long forms so
# resampling boundaries are exact and independent of pandas alias guesses).
TIMEFRAME_TO_PANDAS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}

# Binance kline interval strings for the public REST adapter.
TIMEFRAME_TO_BINANCE = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}

BINANCE_SYMBOLS = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
}


class UnsupportedTimeframeError(ValueError):
    """Raised when a timeframe identifier is not one of the canonical seven."""


class UnsupportedMarketError(ValueError):
    """Raised when a market identifier is not one of the canonical three."""


def validate_market(market: str) -> str:
    """Validate a canonical market or a Binance USDT spot symbol.

    The fixed 21-cell research matrix still uses MARKETS only. Interactive
    crypto commands may use any syntactically valid USDT symbol, e.g. SOLUSDT.
    Binance remains the source of truth for whether that pair actually exists.
    """
    m = str(market).strip().upper()
    if m in MARKETS:
        return m
    if re.fullmatch(r"[A-Z0-9]{2,20}USDT", m):
        return m
    raise UnsupportedMarketError(
        f"Unsupported market {market!r}; use one of {', '.join(MARKETS)} "
        "or a Binance USDT spot symbol such as SOLUSDT"
    )


def validate_timeframe(timeframe: str) -> str:
    tf = str(timeframe).strip()
    if tf not in TIMEFRAMES:
        raise UnsupportedTimeframeError(
            f"Unsupported timeframe {timeframe!r}; supported: {', '.join(TIMEFRAMES)}"
        )
    return tf


def market_timeframe_matrix() -> list[tuple[str, str]]:
    """All 21 canonical (market, timeframe) combinations, in fixed order."""
    return [(m, tf) for m in MARKETS for tf in TIMEFRAMES]
