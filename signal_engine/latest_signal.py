"""Latest-decision check: BUY / SELL / NO TRADE on genuinely recent candles.

Supports LEVEL 3 (current-data signal verification): given a validated CSV
of closed candles ending at the most recently observed candles, compute the
decision the backtester's uniform execution semantics would most recently
have acted on, using only data available at that time:

- Every strategy exposes a per-candle direction series d(t) in {-1,0,+1}
  computed from candles up to and including t only (no lookahead).
- The decision acted upon at candle t derives from d(t-1): the most recent
  acted decision for candles 0..N-1 uses d(N-2) and executes at the open
  of candle N-1.  Nothing from candle N-1's interior is used for the
  decision; ATR/geometry provenance is also candle N-2.
- d(N-1) is reported as the pending direction, which would only be acted
  upon at a future candle that does not exist yet.  No future execution
  price is ever fabricated.

Fail-closed:
- too few candles, no transition, zero/insufficient ATR, or incoherent
  risk geometry all yield an explicit NO TRADE with a documented reason,
  never fabricated entry/stop/target levels.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from signal_engine.indicators import atr as atr_fn
from signal_engine.risk import RiskConfig, build_signal
from signal_engine.strategies import Strategy

ACTED_LABELS = {1: "BUY", -1: "SELL", 0: "NO TRADE"}


def _label(side: int) -> str:
    return ACTED_LABELS.get(side, str(side))


def latest_decision_report(
    df: pd.DataFrame,
    strategy: Strategy,
    risk_config: RiskConfig,
    market: str,
    timeframe: str,
    provenance: str,
    equity: float = 10_000.0,
) -> dict[str, Any]:
    """Build the JSON dict describing the latest acted decision and pending direction."""
    n = len(df)
    base: dict[str, Any] = {
        "kind": "latest_signal_check",
        "market": market,
        "timeframe": timeframe,
        "strategy": strategy.name,
        "provenance": provenance,
        "timezone": "UTC",
        "n_candles": n,
        "coverage_start_utc": df.index[0].isoformat() if n else None,
        "coverage_end_utc": df.index[-1].isoformat() if n else None,
        "equity_for_sizing": equity,
        "semantics": (
            "decision acted at candle t derives from strategy direction at t-1; "
            "latest acted decision uses the penultimate candle as the signal candle "
            "and the last candle's open as the execution reference"
        ),
        "research_only": "offline/verification output; not investment advice; no profitability is implied",
    }
    if n < 3:
        return {
            **base,
            "decision": {"label": "NO TRADE", "reason": f"insufficient_candles:{n}"},
            "pending_raw_direction": None,
        }

    direction = strategy.raw_direction(df)  # per-candle, causal by construction
    signal_idx = n - 2  # latest decision uses candle N-2 as signal candle
    acted_side = int(direction.iloc[signal_idx])
    pending_side = int(direction.iloc[n - 1])

    decision: dict[str, Any]
    if acted_side == 0:
        decision = {
            "label": "NO TRADE",
            "reason": "no_active_transition_signal",
            "signal_ts": df.index[signal_idx].isoformat(),
            "decision_ts": df.index[n - 1].isoformat(),
        }
    else:
        atr_series = atr_fn(df["high"], df["low"], df["close"], window=risk_config.atr_window)
        sig = build_signal(
            side=acted_side,
            signal_idx=signal_idx,
            df=df,
            risk_config=risk_config,
            atr_series=atr_series,
            equity=equity,
            market=market,
            timeframe=timeframe,
            strategy_name=strategy.name,
        )
        if sig is None:
            decision = {
                "label": "NO TRADE",
                "reason": "risk_geometry_rejected_zero_or_insufficient_atr",
                "signal_ts": df.index[signal_idx].isoformat(),
                "decision_ts": df.index[n - 1].isoformat(),
            }
        else:
            stop_dist = abs(sig.entry - sig.stop)
            decision = {
                "label": ACTED_LABELS[acted_side],
                "side": int(sig.side),
                "signal_ts": sig.signal_ts.isoformat(),
                "decision_ts": sig.decision_ts.isoformat(),
                "entry": sig.entry,
                "stop": sig.stop,
                "target": sig.target,
                "position_size": sig.position_size,
                "stop_distance_in_atr": risk_config.atr_multiplier,
                "reward_risk_per_unit": (sig.target - sig.entry) / stop_dist,
                "invalidated_by": sig.invalidated_by,
            }

    return {**base, "decision": decision, "pending_raw_direction": _label(pending_side)}
