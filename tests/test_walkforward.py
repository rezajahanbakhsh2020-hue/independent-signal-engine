"""Walk-forward tests: boundaries, holdout isolation, leakage fail-closed."""
from __future__ import annotations

import pandas as pd
import pytest

from signal_engine.backtest import CostConfig
from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.risk import RiskConfig
from signal_engine.strategies import CrossoverStrategy
from signal_engine.walkforward import (
    WalkForwardError,
    build_splits,
    check_no_leakage,
    run_walk_forward,
)


def test_exact_boundaries_and_counts():
    df = synth_ohlcv("1H", "2024-01-01", 1000, seed=42)
    folds, ho_start, ho_end, n_ho = build_splits(df, n_folds=3, train_candles=200,
                                                 eval_candles=50, holdout_fraction=0.2)
    assert len(folds) == 3
    assert folds[0].train_end == folds[0].eval_start
    for f in folds:
        assert f.n_train == 200 and f.n_eval == 50
        assert f.train_start < f.train_end <= f.eval_start < f.eval_end
    # disjoint folds: fold i+1 train start == fold i eval end
    for a, b in zip(folds, folds[1:]):
        assert b.train_start == a.eval_end
    assert n_ho == 200
    assert ho_start == df.index[800]
    assert ho_end == df.index[999]


def test_holdout_untouched_by_folds():
    df = synth_ohlcv("1H", "2024-01-01", 1000, seed=42)
    folds, ho_start, _, _ = build_splits(df, n_folds=3, train_candles=200,
                                         eval_candles=50, holdout_fraction=0.2)
    for f in folds:
        assert f.eval_end <= ho_start


def test_insufficient_data_fails_closed():
    df = synth_ohlcv("1H", "2024-01-01", 100, seed=42)
    with pytest.raises(WalkForwardError, match="Insufficient development candles"):
        build_splits(df, n_folds=5, train_candles=500, eval_candles=100)


def test_check_no_leakage_flags_overlap():
    df = synth_ohlcv("1H", "2024-01-01", 1000, seed=42)
    folds, ho_start, _, _ = build_splits(df, n_folds=3, train_candles=200,
                                         eval_candles=50, holdout_fraction=0.2)
    check_no_leakage(folds, ho_start)  # ok

    # corrupt fold 0 to overlap fold 1's train window
    bad = list(folds)
    bad[0].eval_end = bad[1].train_start + pd.Timedelta(hours=5)
    with pytest.raises(WalkForwardError, match="overlap"):
        check_no_leakage(bad, ho_start)


def test_check_no_leakage_flags_holdout_contamination():
    df = synth_ohlcv("1H", "2024-01-01", 1000, seed=42)
    folds, ho_start, _, _ = build_splits(df, n_folds=3, train_candles=200,
                                         eval_candles=50, holdout_fraction=0.2)
    with pytest.raises(WalkForwardError, match="contaminates holdout"):
        check_no_leakage(folds, ho_start - pd.Timedelta(hours=500))


def test_run_walk_forward_evaluates_each_fold_and_holdout():
    df = synth_ohlcv("1H", "2024-01-01", 1400, seed=42)
    wf = run_walk_forward(
        df,
        lambda: CrossoverStrategy(fast=5, slow=20),
        "BTCUSDT",
        "1H",
        RiskConfig(),
        CostConfig(),
        n_folds=3,
        train_candles=250,
        eval_candles=100,
    )
    assert len(wf.fold_results) == 3
    assert wf.leakage_check_passed
    assert wf.holdout_result is not None
    assert wf.holdout_stat is not None
    assert wf.n_holdout == int(1400 * 0.2)
    # recorded boundaries: last fold ends before holdout
    assert wf.folds[-1].eval_end <= wf.holdout_start


def test_run_walk_forward_rejects_impossible_geometry():
    df = synth_ohlcv("1H", "2024-01-01", 300, seed=42)
    with pytest.raises(WalkForwardError):
        run_walk_forward(
            df, lambda: CrossoverStrategy(), "BTCUSDT", "1H", RiskConfig(), CostConfig(),
            n_folds=5, train_candles=400, eval_candles=100,
        )
