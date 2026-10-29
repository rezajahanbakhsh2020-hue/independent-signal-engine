"""Deterministic indicators computed only from completed candle data.

All functions accept a Datetime-indexed OHLCV frame and return series
aligned to the input index (values at index i use data up to and
including candle i).  They are pure and deterministic.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rol_high(series: pd.Series, window: int) -> pd.Series:
    """Rolling max including the current bar (used with shift at signal time)."""
    return series.rolling(window=window, min_periods=window).max()


def rol_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Simple rolling mean of True Range (Wilder-agnostic, deterministic)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if window < 2:
        raise ValueError("window must be >= 2")
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower
