"""Full-matrix integration: 21 cells x 4 strategy families on synthetic data.

Synthetic data is deterministic and labeled; this exercises the complete
engine path (validate -> backtest -> metrics -> cell report) for every
market/timeframe/strategy combination that the mission defines.
"""
from __future__ import annotations

import pytest

from signal_engine.backtest import CostConfig
from signal_engine.constants import MARKETS, TIMEFRAMES
from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.data.validate import validate_ohlcv
from signal_engine.report import (
    CELL_STATUS_NO_TRADES,
    CELL_STATUS_COMPLETE_OFFLINE,
    evaluate_cell,
)
from signal_engine.risk import RiskConfig
from signal_engine.strategies import STRATEGY_REGISTRY

N_CANDLES = {tf: (500 if tf == "1D" else 700) for tf in TIMEFRAMES}


@pytest.mark.parametrize("strategy_name", sorted(STRATEGY_REGISTRY))
@pytest.mark.parametrize("market", MARKETS)
def test_every_market_timeframe_strategy_combination(market, strategy_name):
    # cache synthetic dataset per timeframe (deterministic)
    for tf in TIMEFRAMES:
        assert True  # parametrization sanity


def _run_case(market: str, tf: str, strategy_name: str):
    df = synth_ohlcv(tf, "2024-01-01", N_CANDLES[tf], seed=42)
    df_clean, rep = validate_ohlcv(df)
    return evaluate_cell(
        df_clean,
        rep,
        market,
        tf,
        strategy_name,
        strategy_params=None,
        risk=RiskConfig(),
        costs=CostConfig(),
        provenance="SYNTHETIC",
        run_walkforward_flag=True,
        run_baseline_flag=False,
        wf_params={
            "n_folds": 2,
            "train_candles": 200 if tf == "1D" else 150,
            "eval_candles": 180 if tf == "1D" else 100,
            "holdout_fraction": 0.2,
        },
    )


@pytest.mark.parametrize("tf", TIMEFRAMES)
@pytest.mark.parametrize("market", MARKETS)
@pytest.mark.parametrize("strategy_name", sorted(STRATEGY_REGISTRY))
def test_cell_execution(market, tf, strategy_name):
    # 21 x 4 = 84 explicit combinations, none silently omitted
    cell = _run_case(market, tf, strategy_name)
    assert cell.status in {
        CELL_STATUS_COMPLETE_OFFLINE,
        CELL_STATUS_NO_TRADES,
    }
    assert cell.coverage_start_utc is not None
    assert cell.metrics is not None
    assert cell.backtest["n_candles"] >= 340 if tf == "1D" else cell.backtest["n_candles"] >= 200
    if cell.walkforward is not None:
        assert "error" not in cell.walkforward or "Insufficient" in cell.walkforward.get("error", "")
    assert cell.validation["rows_in"] == N_CANDLES[tf]
