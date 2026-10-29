"""Performance, risk, and concentration metrics with explicit denominators.

Concentration definition (documented, no ambiguous defaults):
- Traded-HHI concentration over TRADES on net realized monetary P&L:
    positive_pnl = sum(net_pnl_i for net_pnl_i > 0)
    if positive_pnl <= 0                         -> concentration = None
    if only one positive-trade P&L value exists  -> concentration = 1.0
    else concentration = sum((pnl_i/positive_pnl)^2 for pnl_i > 0)
- This measures how concentrated the GAINS are across trades.  Losses do
  not share the denominator; a set with mostly losses returns None (no
  meaningful concentration percentage can be produced).  Zero and
  negative denominators never produce a misleading percentage.
- Alternative dimension: concentration over STRATEGY net P&L (same
  formula, across strategies for a matrix report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine.backtest import BacktestResult


def hhi_concentration(values: list[float] | np.ndarray) -> float | None:
    """HHI concentration of positive values in [0, 1].

    Values are a share-of-sum numerator set.  Returns None when the
    denominator (sum of positive values) is <= 0 or values are empty.
    """
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return None
    positive = vals[vals > 0]
    if positive.size == 0:
        return None
    denom = positive.sum()
    if not np.isfinite(denom) or denom <= 0:
        return None
    return float(((positive / denom) ** 2).sum())


@dataclass
class PerformanceMetrics:
    n_trades: int
    final_equity: float
    initial_equity: float
    total_return_pct: float | None
    max_drawdown_pct: float | None
    win_rate: float | None
    profit_factor: float | None
    avg_net_pnl: float | None
    avg_win: float | None
    avg_loss: float | None
    expectancy_per_trade: float | None
    exposure_pct: float | None
    turnover_x_equity: float | None
    gross_total_pnl: float | None
    net_total_pnl: float | None
    total_fees: float | None
    worst_trade_pnl: float | None
    best_trade_pnl: float | None
    trade_hhi_concentration: float | None
    n_gains: int
    n_losses: int
    denominator_notes: str
    insufficient_samples: bool

    def as_dict(self) -> dict:
        d = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
        }
        return d


def drawdown_fraction(equity: pd.Series) -> pd.Series:
    """Fraction under running max: 0 to 1 (never negative, dimensionless)."""
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    dd = (equity - peak) / peak.abs()
    return dd.where(peak.abs() > 0, 0.0).clip(-np.inf, 0.0)


def compute_metrics(
    res: BacktestResult,
    concentration_dimension: str = "trades",
    min_samples_warn: int = 20,
) -> PerformanceMetrics:
    """Compute descriptive performance metrics from a backtest result.

    Denominators are explicit:
    - total_return_pct: (final - initial)/|initial| * 100  (initial always > 0
      in the engine; guarded anyway).
    - max_drawdown_pct: max of drawdown_fraction(equity) * 100, computed on
      the equity curve (marked-to-market).  None when equity empty.
    - win_rate: wins/(wins+losses) among closed trades, None if no trades.
    - profit_factor: sum(winning net P&L)/abs(sum(losing net P&L)) using the
      |loss| magnitude; None when no losses (documented, not set to inf),
      cap-free but reported as-is; 0 when no gains and some losses.
    - exposure_pct: fraction of candles where a trade was open.
    - turnover: total notional traded / initial equity (documented).
    """
    eq = res.equity
    init = res.initial_equity
    final = res.final_equity

    total_return = None
    if init and init > 0 and np.isfinite(init):
        total_return = (final - init) / abs(init) * 100.0

    dd = drawdown_fraction(eq)
    max_dd = None
    if not dd.empty and np.isfinite(dd.min()):
        max_dd = float(dd.min() * 100.0)

    trades = res.trades
    n_trades = len(trades)
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    gross_wins = float(sum(wins)) if wins else 0.0
    gross_losses = float(sum(losses)) if losses else 0.0

    win_rate = None
    if n_trades:
        win_rate = len(wins) / (len(wins) + len(losses))

    profit_factor = None
    if gross_losses < 0:
        magnitude = abs(gross_losses)
        if magnitude > 0 and np.isfinite(magnitude):
            profit_factor = gross_wins if gross_wins > 0 else 0.0
            profit_factor = float(gross_wins / magnitude) if gross_wins > 0 else 0.0

    avg_net_pnl = None
    if n_trades:
        avg_net_pnl = float(np.mean([t.net_pnl for t in trades]))
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    expectancy = avg_net_pnl

    exposure = None
    if res.n_candles:
        open_candles = 0
        for tr in trades:
            entry_idx = _idx_of(res.equity.index, tr.entry_ts)
            exit_idx = _idx_of(res.equity.index, tr.exit_ts) if tr.exit_ts else res.n_candles
            if entry_idx is not None and exit_idx is not None:
                open_candles += max(exit_idx - entry_idx, 1)
        exposure = 100.0 * open_candles / res.n_candles

    turnover = None
    if init and init > 0:
        notional = sum(t.entry_price * t.qty for t in trades)
        turnover = float(notional / init)

    ce_conc = hhi_concentration([t.net_pnl for t in trades])

    total_fees = float(sum(t.fees for t in trades)) if trades else 0.0
    gross_total = float(sum(t.gross_pnl for t in trades)) if trades else 0.0
    net_total = float(sum(t.net_pnl for t in trades)) if trades else 0.0

    n_gains = len(wins)
    n_losses = len(losses)
    notes = (
        "concentration dimension=trades; denominator = sum of positive net trade P&L; "
        "losses excluded from denominator; None returned when denominator <= 0"
    )
    return PerformanceMetrics(
        n_trades=n_trades,
        final_equity=float(final),
        initial_equity=float(init),
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_net_pnl=avg_net_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_per_trade=expectancy,
        exposure_pct=exposure,
        turnover_x_equity=turnover,
        gross_total_pnl=gross_total,
        net_total_pnl=net_total,
        total_fees=total_fees,
        worst_trade_pnl=float(min([t.net_pnl for t in trades])) if trades else None,
        best_trade_pnl=float(max([t.net_pnl for t in trades])) if trades else None,
        trade_hhi_concentration=ce_conc,
        n_gains=n_gains,
        n_losses=n_losses,
        denominator_notes=notes,
        insufficient_samples=n_trades < min_samples_warn,
    )


def _idx_of(index: pd.DatetimeIndex, ts: pd.Timestamp | None) -> int | None:
    if ts is None:
        return None
    if len(index) == 0:
        return None
    idx = index.get_indexer([ts], method="bfill")
    i = idx[0]
    return int(i) if i >= 0 else None
