"""RSI reversal strategy using the original Wilder RSI(14) rules."""
from __future__ import annotations

import pandas as pd

from signal_engine.indicators import rsi_wilder
from signal_engine.strategies.base import Strategy


class RSIReversalStrategy(Strategy):
    name = "RSI_Reversal"

    def __init__(self, period: int = 14) -> None:
        if period != 14:
            raise ValueError("RSI_Reversal uses the fixed original period of 14")
        self.period = period

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        rsi = rsi_wilder(df["close"], self.period)
        previous = rsi.shift(1)

        # Preserve the original crossing rules: oversold recovery is BUY;
        # overbought rejection is SELL. Thresholds are intentionally fixed.
        buy = (previous <= 30.0) & (rsi > 30.0)
        sell = (previous >= 70.0) & (rsi < 70.0)

        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[buy & previous.notna() & rsi.notna()] = 1
        raw[sell & previous.notna() & rsi.notna()] = -1
        return raw
