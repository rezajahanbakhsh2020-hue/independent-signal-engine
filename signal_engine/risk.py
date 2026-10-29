"""ATR-based risk geometry, position sizing, and signal objects.

Every emitted signal carries explicit:
- signal_ts:  open time of the candle whose close produced the signal
- decision_ts: open time of the candle after the signal candle (when
  the decision is acted upon)
- entry / stop / target prices with direction-coherence requirements
- invalidation rule: the stop; also a candle-close invalidation if
  configured (close_stop: stop triggered when close crosses, else
  intrabar touch of the stop or target ends the trade).

Direction coherence invariants (enforced, raise RiskError on violation):
- BUY:  stop < entry <= candidate execution upper bound; target > entry
- SELL: stop > entry; target < entry
- All of entry/stop/target finite numeric; stop distance > 0.

Zero ATR or insufficient history => NO TRADE (never fabricated levels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal[1, -1]


class RiskError(ValueError):
    """Raised for directionally incoherent or non-finite risk levels."""


@dataclass(frozen=True)
class Signal:
    market: str
    timeframe: str
    strategy: str
    side: Side
    signal_ts: pd.Timestamp
    decision_ts: pd.Timestamp
    entry: float
    stop: float
    target: float
    position_size: float
    invalidated_by: str = "touch_of_stop"

    def validate(self) -> None:
        vals = (self.entry, self.stop, self.target)
        if not all(np.isfinite(np.asarray(vals, dtype=float))):
            raise RiskError("entry/stop/target must be finite")
        if not np.isfinite(self.position_size) or self.position_size <= 0:
            raise RiskError("position_size must be positive finite")
        if self.side == 1:
            if not (self.stop < self.entry < self.target):
                raise RiskError(
                    f"BUY coherence violated: stop({self.stop}) < entry({self.entry}) < target({self.target})"
                )
        elif self.side == -1:
            if not (self.target < self.entry < self.stop):
                raise RiskError(
                    f"SELL coherence violated: target({self.target}) < entry({self.entry}) < stop({self.stop})"
                )
        else:
            raise RiskError("side must be 1 (BUY) or -1 (SELL)")
        if self.decision_ts <= self.signal_ts:
            raise RiskError("decision_ts must be after signal_ts")


@dataclass
class RiskConfig:
    atr_window: int = 14
    atr_multiplier: float = 2.0   # stop distance = atr * multiplier
    reward_risk_multiple: float = 2.0
    risk_fraction: float = 0.01    # fraction of equity risked per trade
    min_atr: float = 0.0           # if > 0, atr below this is treated as zero
    max_position_fraction: float = 1.0  # cap notional / equity
    close_stop: bool = False       # invalidation by candle close instead of touch

    def validate(self) -> None:
        if self.atr_window < 2:
            raise ValueError("atr_window must be >= 2")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if self.reward_risk_multiple <= 0:
            raise ValueError("reward_risk_multiple must be > 0")
        if not (0 < self.risk_fraction <= 0.5):
            raise ValueError("risk_fraction must be in (0, 0.5]")
        if not (0 < self.max_position_fraction <= 10):
            raise ValueError("max_position_fraction must be in (0, 10]")

    def __post_init__(self) -> None:
        self.validate()


def build_signal(
    side: Side,
    signal_idx: int,
    df: pd.DataFrame,
    risk_config: RiskConfig,
    atr_series: pd.Series,
    equity: float,
    market: str,
    timeframe: str,
    strategy_name: str,
) -> Signal | None:
    """Construct a Signal with ATR-based stop geometry, or NO TRADE (None).

    Fail-closed: zero/missing ATR, non-finite prices, incoherent geometry,
    or insufficient warm-up all produce None, not fabricated levels.
    """
    offset = 1  # execution candle index
    exec_idx = signal_idx + offset
    if exec_idx >= len(df):
        return None
    entry_ref = float(df["open"].iloc[exec_idx])
    a = atr_series.iloc[signal_idx]
    if (
        entry_ref <= 0
        or not np.isfinite(entry_ref)
        or a is None
        or not np.isfinite(a)
        or a <= 0
        or a < risk_config.min_atr
    ):
        return None
    stop_dist = risk_config.atr_multiplier * float(a)
    if stop_dist <= 0 or not np.isfinite(stop_dist):
        return None
    if side == 1:
        stop = entry_ref - stop_dist
        target = entry_ref + risk_config.reward_risk_multiple * stop_dist
    else:
        stop = entry_ref + stop_dist
        target = entry_ref - risk_config.reward_risk_multiple * stop_dist
    per_unit_risk = stop_dist
    if per_unit_risk <= 0 or not np.isfinite(per_unit_risk):
        return None
    qty = (risk_config.risk_fraction * equity) / per_unit_risk
    notional = qty * entry_ref
    if notional > risk_config.max_position_fraction * equity:
        qty = risk_config.max_position_fraction * equity / entry_ref
    if qty <= 0 or not np.isfinite(qty):
        return None
    try:
        sig = Signal(
            market=market,
            timeframe=timeframe,
            strategy=strategy_name,
            side=side,
            signal_ts=df.index[signal_idx],
            decision_ts=df.index[exec_idx],
            entry=entry_ref,
            stop=stop,
            target=target,
            position_size=qty,
            invalidated_by=("close_across_stop" if risk_config.close_stop else "touch_of_stop"),
        )
        sig.validate()
        return sig
    except RiskError:
        return None
