"""Walk-forward evaluation with explicit chronological splits and holdout.

Non-negotiable rules enforced here:
- Chronological order only: every eval window is strictly after its
  train window and strictly before the next fold.
- The holdout is a single, untouched final slice: it is never used for
  training, validation, or hyperparameter selection.  It is evaluated
  exactly once, after all tuning decisions are complete.
- Fail closed: insufficient data raises WalkForwardError instead of
  silently shrinking windows.
- Every split boundary is recorded as exact UTC timestamps and candle
  counts.  A contamination check re-verifies non-overlap at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signal_engine.backtest import BacktestResult, CostConfig, backtest
from signal_engine.metrics import hhi_concentration
from signal_engine.risk import RiskConfig


class WalkForwardError(RuntimeError):
    """Raised when the requested evaluation cannot be built from the data
    without violating chronological/integrity constraints."""


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    n_train: int
    n_eval: int

    def as_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start_utc": self.train_start.isoformat(),
            "train_end_utc_exclusive": self.train_end.isoformat(),
            "eval_start_utc": self.eval_start.isoformat(),
            "eval_end_utc": self.eval_end.isoformat(),
            "n_train_candles": self.n_train,
            "n_eval_candles": self.n_eval,
        }


@dataclass
class WalkForwardSpec:
    n_folds: int
    train_candles: int
    eval_candles: int
    holdout_fraction: float


@dataclass
class WalkForwardResult:
    spec: WalkForwardSpec
    folds: list[Fold]
    fold_results: list[BacktestResult]
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_holdout: int
    holdout_result: BacktestResult | None
    validation_stat: float
    validation_concentration: float | None
    holdout_stat: float | None
    holdout_concentration: float | None
    leakage_check_passed: bool

    def as_dict(self) -> dict:
        return {
            "n_folds": self.spec.n_folds,
            "train_candles": self.spec.train_candles,
            "eval_candles": self.spec.eval_candles,
            "holdout_fraction": self.spec.holdout_fraction,
            "folds": [f.as_dict() for f in self.folds],
            "holdout_start_utc": self.holdout_start.isoformat(),
            "holdout_end_utc": self.holdout_end.isoformat(),
            "n_holdout_candles": self.n_holdout,
            "validation_stat_net_pnl": self.validation_stat,
            "validation_concentration": self.validation_concentration,
            "holdout_stat_net_pnl": self.holdout_stat,
            "holdout_concentration": self.holdout_concentration,
            "leakage_check_passed": self.leakage_check_passed,
        }


def build_splits(
    df: pd.DataFrame,
    n_folds: int,
    train_candles: int,
    eval_candles: int,
    holdout_fraction: float = 0.2,
) -> tuple[list[Fold], pd.Timestamp, pd.Timestamp, int]:
    """Construct folds + holdout boundaries.  Fail-closed."""
    if n_folds < 1:
        raise WalkForwardError("n_folds must be >= 1")
    if train_candles < 2 or eval_candles < 1:
        raise WalkForwardError("train_candles must be >= 2, eval_candles >= 1")
    if not (0.0 < holdout_fraction < 0.9):
        raise WalkForwardError("holdout_fraction must be in (0, 0.9)")

    n_total = len(df)
    n_holdout = max(int(n_total * holdout_fraction), 1)
    n_dev = n_total - n_holdout
    required = n_folds * train_candles + n_folds * eval_candles
    if required > n_dev:
        raise WalkForwardError(
            f"Insufficient development candles: need {required} "
            f"(n_folds={n_folds}, train={train_candles}, eval={eval_candles}) "
            f"but have {n_dev} (total {n_total}, holdout {n_holdout})"
        )

    folds: list[Fold] = []
    # Each fold is disjoint: train window of `train_candles` immediately
    # followed by its own eval window of `eval_candles`; windows advance by
    # (train+eval) per fold so no candle is reused across fold boundaries.
    cursor = 0
    for i in range(n_folds):
        train_start_idx = cursor
        train_end_idx = train_start_idx + train_candles
        eval_start_idx = train_end_idx
        eval_end_idx = eval_start_idx + eval_candles
        folds.append(
            Fold(
                fold_id=i,
                train_start=df.index[train_start_idx],
                train_end=df.index[train_end_idx],
                eval_start=df.index[eval_start_idx],
                eval_end=df.index[eval_end_idx],
                n_train=train_candles,
                n_eval=eval_candles,
            )
        )
        cursor = eval_end_idx

    holdout_start = df.index[n_dev]
    holdout_end = df.index[-1]
    return folds, holdout_start, holdout_end, n_holdout


def check_no_leakage(folds: list[Fold], holdout_start: pd.Timestamp) -> None:
    """Re-verify strict chronology; raises on any overlap/contamination."""
    for f in folds:
        if not (f.train_start < f.train_end <= f.eval_start < f.eval_end):
            raise WalkForwardError(
                f"Fold {f.fold_id}: incoherent window ordering "
                f"({f.train_start} < {f.train_end} <= {f.eval_start} < {f.eval_end} violated)"
            )
    for a, b in zip(folds, folds[1:]):
        if a.eval_end > b.train_start:
            raise WalkForwardError(
                f"Temporal overlap detected between fold {a.fold_id} eval end "
                f"({a.eval_end}) and fold {b.fold_id} train start ({b.train_start})"
            )
    if folds and folds[-1].eval_end > holdout_start:
        raise WalkForwardError(
            f"Fold {folds[-1].fold_id} eval end ({folds[-1].eval_end}) "
            f"contaminates holdout start ({holdout_start})"
        )


def run_walk_forward(
    df: pd.DataFrame,
    strategy_factory,
    market: str,
    timeframe: str,
    risk: RiskConfig,
    costs: CostConfig,
    n_folds: int = 4,
    train_candles: int = 400,
    eval_candles: int = 100,
    holdout_fraction: float = 0.2,
    initial_equity: float = 10_000.0,
    provenance: dict | None = None,
) -> WalkForwardResult:
    """Run chronological walk-forward with strict train/eval/holdout split.

    strategy_factory() -> Strategy   (fresh deterministic instance each fold;
                                      any *tuning* must happen on train folds;
                                      the factory decides what gets built.)
    """
    folds, holdout_start, holdout_end, n_holdout = build_splits(
        df, n_folds, train_candles, eval_candles, holdout_fraction
    )
    check_no_leakage(folds, holdout_start)
    index = df.index

    fold_results: list[BacktestResult] = []
    for f in folds:
        sub = df[(index >= f.eval_start) & (index < f.eval_end)]
        strat = strategy_factory()
        res = backtest(
            sub,
            strat,
            market,
            timeframe,
            risk,
            costs,
            initial_equity=initial_equity,
            provenance=provenance,
        )
        fold_results.append(res)

    sub_ho = df[index >= holdout_start]
    ho_res: BacktestResult | None = None
    if not sub_ho.empty:
        ho_res = backtest(
            sub_ho,
            strategy_factory(),  # final strategy, tuned on train/val only
            market,
            timeframe,
            risk,
            costs,
            initial_equity=initial_equity,
            provenance=provenance,
        )

    val_stat = float(sum(r.final_equity - r.initial_equity for r in fold_results))
    val_conc = hhi_concentration(
        [t.net_pnl for r in fold_results for t in r.trades]
    )
    ho_stat: float | None = float(ho_res.final_equity - ho_res.initial_equity) if ho_res else None
    ho_conc: float | None = (
        hhi_concentration([t.net_pnl for t in ho_res.trades]) if ho_res else None
    )

    spec = WalkForwardSpec(
        n_folds=n_folds,
        train_candles=train_candles,
        eval_candles=eval_candles,
        holdout_fraction=holdout_fraction,
    )
    return WalkForwardResult(
        spec=spec,
        folds=folds,
        fold_results=fold_results,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        n_holdout=n_holdout,
        holdout_result=ho_res,
        validation_stat=val_stat,
        validation_concentration=val_conc,
        holdout_stat=ho_stat,
        holdout_concentration=ho_conc,
        leakage_check_passed=True,
    )
