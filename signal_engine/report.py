"""Report generation for single cells and the full 21-cell matrix.

Every report cell carries explicit provenance:
- data source and provenance label ('synthetic' vs real file/api fetch)
- retrieval time (UTC) where applicable
- exact data coverage
- validation summary (rejected rows, duplicates, gaps)
- honest completion status for the cell:
    COMPLETE_OFFLINE / UNAVAILABLE_DATA / ERROR
Nothing is silently omitted from the matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine.backtest import CostConfig, backtest
from signal_engine.constants import market_timeframe_matrix, validate_market, validate_timeframe
from signal_engine.data.validate import ValidationReport
from signal_engine.metrics import compute_metrics
from signal_engine.risk import RiskConfig
from signal_engine.stats import run_randomized_baseline
from signal_engine.strategies import STRATEGY_REGISTRY
from signal_engine.walkforward import run_walk_forward

CELL_STATUS_COMPLETE_OFFLINE = "COMPLETE_OFFLINE"
CELL_STATUS_UNAVAILABLE = "UNAVAILABLE_DATA"
CELL_STATUS_ERROR = "ERROR"
CELL_STATUS_NO_TRADES = "COMPLETE_OFFLINE_NO_TRADES"


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def matrix_cells() -> list[tuple[str, str]]:
    return market_timeframe_matrix()


@dataclass
class CellReport:
    market: str
    timeframe: str
    status: str
    data_provenance: str
    coverage_start_utc: str | None = None
    coverage_end_utc: str | None = None
    n_candles: int | None = None
    validation: dict | None = None
    backtest: dict | None = None
    metrics: dict | None = None
    walkforward: dict | None = None
    baseline: dict | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "market": self.market,
            "timeframe": self.timeframe,
            "status": self.status,
            "data_provenance": self.data_provenance,
            "coverage_start_utc": self.coverage_start_utc,
            "coverage_end_utc": self.coverage_end_utc,
            "n_candles": self.n_candles,
            "validation": self.validation,
            "backtest": self.backtest,
            "metrics": self.metrics,
            "walkforward": self.walkforward,
            "baseline": self.baseline,
            "error": self.error,
        }


def evaluate_cell(
    df: pd.DataFrame,
    validation_report: ValidationReport,
    market: str,
    timeframe: str,
    strategy_name: str,
    strategy_params: dict | None,
    risk: RiskConfig,
    costs: CostConfig,
    provenance: str,
    run_walkforward_flag: bool = True,
    run_baseline_flag: bool = True,
    baseline_m: int = 999,
    baseline_seed: int = 20260101,
    wf_params: dict | None = None,
    initial_equity: float = 10_000.0,
) -> CellReport:
    """Evaluate one (market, timeframe) cell against one strategy."""
    validate_market(market)
    validate_timeframe(timeframe)
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy {strategy_name!r}; available: {', '.join(STRATEGY_REGISTRY)}"
        )
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    strategy_params = strategy_params or {}

    try:
        strat = strategy_cls(**strategy_params)
    except Exception as exc:  # invalid params are explicit errors, never silent
        return CellReport(
            market=market,
            timeframe=timeframe,
            status=CELL_STATUS_ERROR,
            data_provenance=provenance,
            error=f"invalid strategy parameters: {exc}",
        )

    res = backtest(
        df,
        strat,
        market,
        timeframe,
        risk,
        costs,
        initial_equity=initial_equity,
        provenance={"source": provenance, "provenance_label": "REAL" if "synthetic" not in provenance else "SYNTHETIC"},
    )
    metrics = compute_metrics(res)

    status = CELL_STATUS_COMPLETE_OFFLINE
    if len(res.trades) == 0:
        status = CELL_STATUS_NO_TRADES

    cell = CellReport(
        market=market,
        timeframe=timeframe,
        status=status,
        data_provenance=provenance,
        coverage_start_utc=df.index[0].isoformat(),
        coverage_end_utc=df.index[-1].isoformat(),
        n_candles=len(df),
        validation=validation_report.as_dict(),
        backtest=res.as_dict(),
        metrics=metrics.as_dict(),
    )

    if run_walkforward_flag:
        wf_params = wf_params or {}
        sub = df
        try:
            wf = run_walk_forward(
                sub,
                lambda: strategy_cls(**strategy_params),
                market,
                timeframe,
                risk,
                costs,
                initial_equity=initial_equity,
                provenance={"source": provenance},
                **wf_params,
            )
            cell.walkforward = wf.as_dict()
        except Exception as exc:
            cell.walkforward = {"error": str(exc)}

    if run_baseline_flag:
        try:
            base = run_randomized_baseline(
                df,
                lambda: strategy_cls(**strategy_params),
                market,
                timeframe,
                risk,
                costs,
                m_simulations=baseline_m,
                seed=baseline_seed,
                initial_equity=initial_equity,
                provenance={"source": provenance},
            )
            cell.baseline = base.as_dict()
        except Exception as exc:
            cell.baseline = {"error": str(exc)}
    return cell


def matrix_skeleton() -> list[tuple[str, str]]:
    """Always materialize all 21 combinations explicitly (never quietly)."""
    return matrix_cells()


def report_to_json(cells: list[CellReport], indent: int = 2) -> str:
    data = {"cells": [c.as_dict() for c in cells]}
    return json.dumps(data, indent=indent, default=_jsonable)
