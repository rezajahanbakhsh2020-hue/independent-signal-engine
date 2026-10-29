"""Strategy base class and explicit signal semantics.

Signal semantics (applies to every strategy in this package):
- A strategy computes a *raw direction* {-1, 0, +1} from data that ends
  at candle i (inclusive).
- The candle i is assumed closed when its close price is final.
- signal_ts  = open time of candle i  (the candle that produced evidence)
- decision_ts= open time of candle i+1 (when the decision is acted on)
- execution occurs at the open of candle i+1 using its own bid/ask with
  spread and slippage applied.  Nothing from candle i+1's interior or
  future candles is used to generate the decision.
- new-entry semantics: a signal above zero requests a long; below zero
  requests a short; zero means NO TRADE.  Strategies emit raw direction
  only when a state transition occurs (state change), so repeated
  signals are not re-entered continuously; the backtester also refuses
  to open a trade while a position is already open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of {-1, 0, +1} per candle, using only that candle
        and earlier data (no shift inside the strategy -- shifting to the
        execution candle is the backtester's responsibility)."""


def transition_direction(raw: pd.Series) -> pd.Series:
    """Emit +1/-1 only when raw direction changes from the prior candle."""
    out = pd.Series(0, index=raw.index, dtype="int64")
    prev = raw.shift(1).fillna(0)
    out[(raw > 0) & (prev <= 0)] = 1
    out[(raw < 0) & (prev >= 0)] = -1
    return out
