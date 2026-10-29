"""Trend pullback strategy: trade in the direction of a slow trend filter
when price momentarily pulls back toward/through a fast moving average
and then closes back in the trend direction.

Deterministic, uses only completed candles.
"""

from __future__ import annotations

import pandas as pd

from signal_engine.indicators import ema, sma
from signal_engine.strategies.base import Strategy, transition_direction


class PullbackStrategy(Strategy):
    name = "pullback"

    def __init__(
        self,
        trend_window: int = 50,
        pullback_window: int = 10,
        kind: str = "sma",
        min_trend_strength: float = 0.0,
    ) -> None:
        if trend_window <= 2 or pullback_window <= 1:
            raise ValueError("trend_window must be >= 3 and pullback_window >= 2")
        if pullback_window >= trend_window:
            raise ValueError("pullback_window must be < trend_window")
        if kind not in ("sma", "ema"):
            raise ValueError("kind must be 'sma' or 'ema'")
        if min_trend_strength < 0:
            raise ValueError("min_trend_strength must be >= 0")
        self.trend_window = trend_window
        self.pullback_window = pullback_window
        self.kind = kind
        self.min_trend_strength = min_trend_strength

    def _avg(self, series: pd.Series, window: int) -> pd.Series:
        return sma(series, window) if self.kind == "sma" else ema(series, window)

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend = self._avg(close, self.trend_window)
        pull = self._avg(close, self.pullback_window)
        lower = df["low"]
        upper = df["high"]

        up_trend = trend.diff() > self.min_trend_strength
        dn_trend = trend.diff() < -self.min_trend_strength
        strength_ok = pd.Series(True, index=df.index)
        strength_ok = trend.notna()

        touched = (lower <= pull) | (upper >= pull)
        closed_above = close > pull
        closed_below = close < pull

        long_ok = up_trend & touched & closed_above & strength_ok
        short_ok = dn_trend & touched & closed_below & strength_ok

        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[long_ok] = 1
        raw[short_ok] = -1
        raw[trend.isna() | pull.isna()] = 0
        return transition_direction(raw)
