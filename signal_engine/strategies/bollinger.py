"""Bollinger-Band mean-reversion strategy.

Long when candle close crosses back under the lower band (i.e., closes
back above close[i-1] while prior close was under the lower band at the
prior candle); symmetric for shorts at the upper band.

Simplified deterministic variant: enter long when close[i-1] < lower[i-1]
and close[i] > lower[i] (a "band re-entry"); short when close[i-1] >
upper[i-1] and close[i] < upper[i].

This uses only completed candles at i-1 and i, so executing at open of
i+1 involves no lookahead.
"""

from __future__ import annotations

import pandas as pd

from signal_engine.indicators import bollinger
from signal_engine.strategies.base import Strategy


class BollingerMeanReversionStrategy(Strategy):
    name = "bollinger_mean_reversion"

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

        prev_close = close.shift(1)
        prev_lower = lower.shift(1)
        prev_upper = upper.shift(1)

        long_ok = (prev_close < prev_lower) & (close >= lower)
        short_ok = (prev_close > prev_upper) & (close <= upper)

        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[long_ok & prev_lower.notna() & lower.notna()] = 1
        raw[short_ok & prev_upper.notna() & upper.notna()] = -1
        return raw  # re-entry signals allowed; transition_direction not needed
