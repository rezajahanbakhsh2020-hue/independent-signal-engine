"""Bollinger band re-entry reversal strategy.

BUY when the close crosses back above the lower band.
SELL when the close crosses back below the upper band.
Only completed candles are used; the engine acts on the next candle.
"""
from __future__ import annotations

import pandas as pd

from signal_engine.indicators import bollinger
from signal_engine.strategies.base import Strategy


class BollingerMeanReversionStrategy(Strategy):
    name = "Bollinger_Reversion"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        if window < 5:
            raise ValueError("window must be >= 5")
        if num_std <= 0:
            raise ValueError("num_std must be > 0")
        self.window = window
        self.num_std = num_std

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        upper, _mid, lower = bollinger(close, self.window, self.num_std)

        previous_close = close.shift(1)
        previous_lower = lower.shift(1)
        previous_upper = upper.shift(1)

        crossed_up_from_lower = (previous_close < previous_lower) & (close >= lower)
        crossed_down_from_upper = (previous_close > previous_upper) & (close <= upper)

        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[crossed_up_from_lower & previous_lower.notna() & lower.notna()] = 1
        raw[crossed_down_from_upper & previous_upper.notna() & upper.notna()] = -1
        return raw
