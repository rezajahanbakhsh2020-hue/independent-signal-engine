"""Donchian-style price breakout strategy.

A long signal fires when the candle closes above the highest high of the
prior N completed candles; short when it closes below the lowest low of
the prior N candles.  Prior-N window explicitly excludes the current
candle to avoid self-referential breakout confirmation.

Default filter: minimum true-range-normalized candle-size filter is not
used by default (kept simple); an optional minimum breakout margin (in
price fraction of the prior range) can be configured.
"""

from __future__ import annotations

import pandas as pd

from signal_engine.strategies.base import Strategy, transition_direction


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self,lookback: int = 20, min_margin: float = 0.0) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if min_margin < 0:
            raise ValueError("min_margin must be >= 0")
        self.lookback = lookback
        self.min_margin = min_margin

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        prior_high = df["high"].shift(1).rolling(self.lookback, min_periods=self.lookback).max()
        prior_low = df["low"].shift(1).rolling(self.lookback, min_periods=self.lookback).min()
        close = df["close"]
        range_px = (prior_high - prior_low).abs()

        long_ok = (
            (close > prior_high)
            & ((close - prior_high) >= self.min_margin * range_px)
        )
        short_ok = (
            (close < prior_low)
            & ((prior_low - close) >= self.min_margin * range_px)
        )
        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[long_ok & prior_high.notna()] = 1
        raw[short_ok & prior_low.notna()] = -1
        return transition_direction(raw)
