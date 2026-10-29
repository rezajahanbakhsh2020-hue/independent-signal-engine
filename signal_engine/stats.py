"""Randomized-baseline statistical evaluation.

Test design (documented, reproducible):
- OBSERVED statistic: net total P&L of the strategy over the configured
  eval window (final_equity - initial_equity from the deterministic
  backtest).
- NULL model: identical dataset, identical timestamps, identical risk
  geometry and cost parameters, identical signal TIMING (same candle at
  which a state-change entry would be attempted), but the side of each
  entry opportunity drawn i.i.d. uniform from {+1, -1} with a
  reproducible numpy Generator (default_rng(seed)).  This destroys the
  directional information claimed by the strategy while preserving the
  timing cadence, cost structure, and risk geometry.
- M simulations, M chosen by the caller (must be >= 1).
- One-sided empirical p-value:
      p = (1 + (# sims with stat >= observed)) / (M + 1)
- Limitations (disclosed in every report):
  * the null is NOT a universal market null; it does not model regime
    persistence, volatility clustering, or realistic market microstructure;
  * the test requires a sufficient number of observed trades; with zero
    trades it returns no p-value (fail-closed) rather than a fake result;
  * per-configuration p-values do not correct for multiple testing.
    When K configurations are tested, apply a Bonferroni-style
    adjustment or report min-p * K as an upper bound on family-wise
    error, which this module surfaces as `bonferroni_min_p_bound`.
  * z-scores (where produced) are descriptive statistics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine.backtest import BacktestResult, CostConfig, backtest
from signal_engine.risk import RiskConfig
from signal_engine.strategies import Strategy


@dataclass
class BaselineResult:
    m_simulations: int
    seed: int
    observed_stat: float
    random_stats: list[float]
    p_value: float | None
    n_simulations_with_valid_stat: int
    statistic_name: str = "net_total_pnl"
    null_model: str = "random_entry_side_uniform_iid_same_timing_geometry"
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if self.p_value is not None and not (0.0 < self.p_value <= 1.0):
            raise ValueError("p-value out of (0, 1] range")

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic_name,
            "null_model": self.null_model,
            "M": self.m_simulations,
            "seed": self.seed,
            "observed_stat": self.observed_stat,
            "random_stats_summary": {
                "min": min(self.random_stats) if self.random_stats else None,
                "median": float(np.median(self.random_stats)) if self.random_stats else None,
                "max": max(self.random_stats) if self.random_stats else None,
            },
            "p_value": self.p_value,
            "n_simulations_with_valid_stat": self.n_simulations_with_valid_stat,
            "notes": list(self.notes),
        }


class RandomizedSideStrategy(Strategy):
    """Wraps a base strategy: same entry opportunities, side drawn at random."""

    name = "randomized_side"

    def __init__(self, base: Strategy, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        self._base = base
        self._rng = rng
        self.name = f"randomized_side({base.name})"

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        base = self._base.raw_direction(df)
        events = (base != 0).to_numpy()
        out = base.to_numpy().copy()
        n_events = int(events.sum())
        if n_events:
            sides = self._rng.integers(0, 2, size=n_events) * 2 - 1  # {-1, +1}
            out[events] = sides
        return pd.Series(out, index=df.index, dtype="int64")


def run_randomized_baseline(
    df: pd.DataFrame,
    strategy_factory,
    market: str,
    timeframe: str,
    risk: RiskConfig,
    costs: CostConfig,
    m_simulations: int,
    seed: int,
    initial_equity: float = 10_000.0,
    provenance: dict | None = None,
    cell_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> BaselineResult:
    """Compare observed strategy net P&L against M randomized-side sims.

    Optional cell_window restricts both observed and random evaluation to
    an identical window (used for out-of-sample/baseline comparability).
    """
    if m_simulations < 1:
        raise ValueError("M must be >= 1")
    evals_df = df
    if cell_window is not None:
        s, e = cell_window
        evals_df = df[(df.index >= s) & (df.index < e)]

    strat = strategy_factory()
    observed: BacktestResult = backtest(
        evals_df,
        strat,
        market,
        timeframe,
        risk,
        costs,
        initial_equity=initial_equity,
        provenance=provenance,
    )
    observed_stat = observed.final_equity - observed.initial_equity

    notes = [
        "null: identical signal timing, uniform iid side; not a universal market null",
        "p = (1 + count(random >= observed)) / (M + 1)",
    ]
    if len(observed.trades) == 0:
        notes.append(
            "observed strategy produced 0 trades in the eval window; p-value withheld (fail-closed)"
        )
        return BaselineResult(
            m_simulations=m_simulations,
            seed=seed,
            observed_stat=observed_stat,
            random_stats=[],
            p_value=None,
            n_simulations_with_valid_stat=0,
            notes=notes,
        )

    base = strat
    random_stats: list[float] = []
    for i in range(m_simulations):
        sim_seed = (seed + i) % (2**32 - 1)
        rng = np.random.default_rng(sim_seed)
        sim_strat = RandomizedSideStrategy(base, rng)
        sims_res = backtest(
            evals_df,
            sim_strat,
            market,
            timeframe,
            risk,
            costs,
            initial_equity=initial_equity,
            provenance=provenance,
        )
        random_stats.append(sims_res.final_equity - sims_res.initial_equity)

    count_ge = sum(1 for s_ in random_stats if s_ >= observed_stat)
    p_value = (1 + count_ge) / (m_simulations + 1)
    notes.append("per-config p-value: multiply by number of tested configs for Bonferroni bound")
    return BaselineResult(
        m_simulations=m_simulations,
        seed=seed,
        observed_stat=float(observed_stat),
        random_stats=[float(s_) for s_ in random_stats],
        p_value=float(p_value),
        n_simulations_with_valid_stat=len(random_stats),
        notes=notes,
    )
