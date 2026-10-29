"""Chronological, deterministic, fill-aware backtester.

Execution semantics (documented, enforced):
- Strategy signals are computed per candle i using data up to and
  including candle i only.  A signal is *acted upon* at candle i+1's
  open.  Nothing from candle i+1 (or later) is used to generate the
  decision.
- Fill model (worst-side fills):
    Long entry  = open * (1 + (spread/2 + slippage))
    Short entry = open * (1 - (spread/2 + slippage))
    Long exit at target  = target * (1 - spread/2)          [sell at bid]
    Long exit at stop    = stop  * (1 - spread/2 - slippage)
    Short exit at target = target * (1 + spread/2)          [buy at ask]
    Short exit at stop   = stop  * (1 + spread/2 + slippage)
- Candle ordering inside a trade: fill at open; for the remainder of
  the candle, STOP is evaluated before TARGET.  If both levels are hit
  in the same candle, the STOP is assumed to fill first (conservative).
- Fees: percentage fee charged on notional at every fill and deducted
  from equity immediately at fill time.
- Overlapping trades: never.  At most one open position at any moment.
- Equity is marked to market each candle's close for drawdown/exposure.
- Gross vs net: Trade.gross_pnl excludes all fees; Trade.net_pnl
  subtracts entry + exit fees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signal_engine.indicators import atr as atr_fn
from signal_engine.risk import RiskConfig, RiskError, Signal, build_signal
from signal_engine.strategies import Strategy


class BacktestConfigError(ValueError):
    """Raised on incoherent backtest configuration."""


@dataclass
class CostConfig:
    fee_bps: float = 2.0        # percent-of-notional fee per fill (in bps)
    spread_bps: float = 2.0     # total spread (in bps), half applied each side
    slippage_bps: float = 1.0   # adverse slippage per fill (in bps)

    def __post_init__(self) -> None:
        for k in ("fee_bps", "spread_bps", "slippage_bps"):
            v = float(getattr(self, k))
            if v < 0 or not np.isfinite(v):
                raise BacktestConfigError(f"{k} must be finite and >= 0")

    def _adj(self, price: float, bps: float) -> float:
        return price * (1.0 + bps / 10000.0)

    def enter_fill(self, side: int, open_price: float) -> float:
        if side == 1:
            return self._adj(open_price, self.spread_bps / 2 + self.slippage_bps)
        return self._adj(open_price, -(self.spread_bps / 2 + self.slippage_bps))

    def exit_target_fill(self, side: int, target_price: float) -> float:
        if side == 1:
            return self._adj(target_price, -self.spread_bps / 2)
        return self._adj(target_price, self.spread_bps / 2)

    def exit_stop_fill(self, side: int, stop_price: float) -> float:
        if side == 1:
            return self._adj(stop_price, -(self.spread_bps / 2 + self.slippage_bps))
        return self._adj(stop_price, self.spread_bps / 2 + self.slippage_bps)

    def fee_on(self, notional: float) -> float:
        return abs(notional) * self.fee_bps / 10000.0


@dataclass
class Trade:
    side: int
    market: str
    timeframe: str
    strategy: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    qty: float
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = "open"
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees


@dataclass
class BacktestResult:
    market: str
    timeframe: str
    strategy: str
    start_utc: str
    end_utc: str
    n_candles: int
    initial_equity: float
    final_equity: float
    equity: pd.Series
    trades: list[Trade]
    n_signals_rejected_zero_atr: int = 0
    n_signals_rejected_incoherent: int = 0
    config: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "market": self.market,
            "timeframe": self.timeframe,
            "strategy": self.strategy,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "n_candles": self.n_candles,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "n_trades": len(self.trades),
            "n_signals_rejected_zero_atr": self.n_signals_rejected_zero_atr,
            "n_signals_rejected_incoherent": self.n_signals_rejected_incoherent,
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }


def backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    market: str,
    timeframe: str,
    risk: RiskConfig,
    costs: CostConfig,
    initial_equity: float = 10_000.0,
    provenance: dict | None = None,
) -> BacktestResult:
    """Run a chronologically-simulated backtest.  Deterministic."""
    if initial_equity <= 0:
        raise BacktestConfigError("initial_equity must be > 0")
    empty = BacktestResult(
        market=market,
        timeframe=timeframe,
        strategy=strategy.name,
        start_utc="",
        end_utc="",
        n_candles=0,
        initial_equity=initial_equity,
        final_equity=initial_equity,
        equity=pd.Series(dtype=float),
        trades=[],
        config={"risk": vars(risk), "costs": vars(costs), "strategy_params": vars(strategy)},
        provenance=provenance or {"source": "unknown"},
        warnings=["empty_dataframe"],
    )
    if df.empty:
        return empty

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atr_arr = atr_fn(df["high"], df["low"], df["close"], risk.atr_window).to_numpy(dtype=float)

    raw = strategy.raw_direction(df)
    decision = raw.shift(1).fillna(0).to_numpy(dtype=np.int64)  # decide at t from signal at t-1
    index = df.index

    n = len(df)
    equity = float(initial_equity)
    eq_curve = np.full(n, np.nan, dtype=float)
    trades: list[Trade] = []
    t_open: Trade | None = None
    rejected_zero_atr = 0
    rejected_incoherent = 0

    def close_position(side: int, tr: Trade, idx: int, fill: float, reason: str) -> None:
        nonlocal equity, t_open
        gross = tr.qty * (fill - tr.entry_price) if side == 1 else tr.qty * (tr.entry_price - fill)
        fee = costs.fee_on(fill * tr.qty)
        tr.fees += fee
        tr.gross_pnl += gross
        equity += gross - fee
        tr.exit_ts = index[idx]
        tr.exit_price = fill
        tr.exit_reason = reason
        trades.append(tr)
        t_open = None

    for t in range(n):
        ts = index[t]

        # 1. manage an open position against this candle's range,
        #    stop-first (conservative) when both stop and target are hit.
        if t_open is not None:
            side = t_open.side
            if side == 1:
                stop_hit = lows[t] <= t_open.stop_price
                target_hit = highs[t] >= t_open.target_price
            else:
                stop_hit = highs[t] >= t_open.stop_price
                target_hit = lows[t] <= t_open.target_price
            if stop_hit:
                fill = costs.exit_stop_fill(side, t_open.stop_price)
                close_position(side, t_open, t, fill, "stop")
            elif target_hit:
                fill = costs.exit_target_fill(side, t_open.target_price)
                close_position(side, t_open, t, fill, "target")

        # 2. act on decision to enter (only if flat).  decision[t] is the
        #    raw signal computed from candle t-1 (already shifted).
        sig_side = int(decision[t])
        if t_open is None and sig_side != 0:
            stop_builder = getattr(strategy, "stop_for_signal", None)
            if callable(stop_builder):
                # Structural-stop strategies do not depend on ATR for entry geometry.
                try:
                    entry_ref = float(opens[t])
                    structural_stop = float(stop_builder(df, t - 1, sig_side))
                    stop_dist = abs(entry_ref - structural_stop)
                    if (not np.isfinite(structural_stop) or not np.isfinite(entry_ref)
                            or entry_ref <= 0 or stop_dist <= 0
                            or (sig_side == 1 and structural_stop >= entry_ref)
                            or (sig_side == -1 and structural_stop <= entry_ref)):
                        raise RiskError("invalid structural stop geometry")
                    target = entry_ref + sig_side * stop_dist * risk.reward_risk_multiple
                    qty = risk.risk_fraction * equity / stop_dist
                    qty = min(qty, risk.max_position_fraction * equity / entry_ref)
                    sig = Signal(
                        market=market, timeframe=timeframe, strategy=strategy.name,
                        side=sig_side, signal_ts=index[t - 1], decision_ts=index[t],
                        entry=entry_ref, stop=structural_stop, target=target,
                        position_size=qty,
                        invalidated_by=("close_across_stop" if risk.close_stop else "touch_of_stop"),
                    )
                    sig.validate()
                except (ValueError, TypeError, IndexError, RiskError, OverflowError):
                    rejected_incoherent += 1
                    sig = None
            else:
                atr_val = float(atr_arr[t - 1])
                if not (np.isfinite(atr_val) and atr_val > 0):
                    rejected_zero_atr += 1
                    sig = None
                else:
                    sig = build_signal(
                        side=sig_side,  # type: ignore[arg-type]
                        signal_idx=t - 1, df=df, risk_config=risk,
                        atr_series=pd.Series(atr_arr, index=index), equity=equity,
                        market=market, timeframe=timeframe, strategy_name=strategy.name,
                    )
                    if sig is None:
                        rejected_incoherent += 1
            if sig is not None:
                        entry_fill = costs.enter_fill(sig_side, float(opens[t]))
                        entry_fee = costs.fee_on(entry_fill * sig.position_size)
                        equity -= entry_fee
                        t_open = Trade(
                            side=sig_side,
                            market=market,
                            timeframe=timeframe,
                            strategy=strategy.name,
                            signal_ts=index[t - 1],
                            entry_ts=ts,
                            entry_price=entry_fill,
                            stop_price=sig.stop,
                            target_price=sig.target,
                            qty=sig.position_size,
                            fees=entry_fee,
                        )

        # 3. mark-to-market at close.
        if t_open is not None:
            unrealized = (
                t_open.qty * (closes[t] - t_open.entry_price)
                if t_open.side == 1
                else t_open.qty * (t_open.entry_price - closes[t])
            )
            eq_curve[t] = equity + unrealized
        else:
            eq_curve[t] = equity

    # close lingering position at the final candle close (documented).
    if t_open is not None:
        fill = (
            costs.exit_stop_fill(t_open.side, float(closes[-1]))
            if t_open.side == 1
            else costs.exit_target_fill(t_open.side, float(closes[-1]))
        )
        close_position(t_open.side, t_open, n - 1, fill, "end_of_data")
        eq_curve[-1] = equity

    eq_s = pd.Series(eq_curve, index=index, dtype=float)
    eq_s = eq_s.ffill().fillna(float(initial_equity)).bfill()

    return BacktestResult(
        market=market,
        timeframe=timeframe,
        strategy=strategy.name,
        start_utc=index[0].isoformat(),
        end_utc=index[-1].isoformat(),
        n_candles=int(n),
        initial_equity=float(initial_equity),
        final_equity=float(equity),
        equity=eq_s,
        trades=trades,
        n_signals_rejected_zero_atr=int(rejected_zero_atr),
        n_signals_rejected_incoherent=int(rejected_incoherent),
        config={
            "risk": vars(risk),
            "costs": vars(costs),
            "strategy_params": vars(strategy),
        },
        provenance=dict(provenance or {}),
        warnings=[],
    )
