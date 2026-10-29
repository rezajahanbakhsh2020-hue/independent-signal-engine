"""SMA/EMA crossover strategy."""

from __future__ import annotations

import pandas as pd

from signal_engine.indicators import ema, sma
from signal_engine.strategies.base import Strategy, transition_direction


class CrossoverStrategy(Strategy):
    name = "crossover"

    def __init__(self, fast: int = 10, slow: int = 30, kind: str = "sma") -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("require 0 < fast < slow")
        if kind not in ("sma", "ema"):
            raise ValueError("kind must be 'sma' or 'ema'")
        self.fast = fast
        self.slow = slow
        self.kind = kind

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        avg_fast = sma(close, self.fast) if self.kind == "sma" else ema(close, self.fast)
        avg_slow = sma(close, self.slow) if self.kind == "sma" else ema(close, self.slow)
        raw = pd.Series(0, index=df.index, dtype="int64")
        raw[avg_fast > avg_slow] = 1
        raw[avg_fast < avg_slow] = -1
        # warm-up rows with NaN averages must be NO TRADE
        raw[avg_fast.isna() | avg_slow.isna()] = 0
        return transition_direction(raw)
