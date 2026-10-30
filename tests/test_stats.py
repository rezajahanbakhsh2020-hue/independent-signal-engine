"""Randomized-baseline statistical tests."""
from __future__ import annotations

import numpy as np
import pytest

from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.backtest import CostConfig
from signal_engine.risk import RiskConfig
from tests._strategy_fixtures import CrossoverStrategy
from signal_engine.stats import RandomizedSideStrategy, run_randomized_baseline


def test_p_value_formula_properties(synth_1h):
    res = run_randomized_baseline(
        synth_1h,
        lambda: CrossoverStrategy(fast=5, slow=20),
        "BTCUSDT",
        "1H",
        RiskConfig(),
        CostConfig(),
        m_simulations=19,
        seed=123,
    )
    assert res.m_simulations == 19
    assert len(res.random_stats) == 19
    # p must be of the form (1+k)/(M+1) for integer k in [0, M]
    p = res.p_value
    k = p * (res.m_simulations + 1) - 1
    assert abs(k - round(k)) < 1e-9
    assert 0.0 < p <= 1.0


def test_p_value_deterministic_given_seed(synth_1h):
    args = (
        lambda: CrossoverStrategy(fast=5, slow=20),
        "BTCUSDT", "1H", RiskConfig(), CostConfig(), 10, 55,
    )
    r1 = run_randomized_baseline(synth_1h, *args)
    r2 = run_randomized_baseline(synth_1h, *args)
    assert r1.p_value == r2.p_value
    assert r1.random_stats == r2.random_stats


def test_zero_observed_trades_withheld_p_value():
    df = synth_ohlcv("1H", "2024-01-01", 40, seed=1)  # tiny: warmup barely exists
    res = run_randomized_baseline(
        df,
        lambda: CrossoverStrategy(fast=10, slow=30),
        "BTCUSDT",
        "1H",
        RiskConfig(),
        CostConfig(),
        m_simulations=10,
        seed=1,
    )
    if res.p_value is None:
        assert "0 trades" in " ".join(res.notes)
    else:
        assert 0.0 < res.p_value <= 1.0


def test_randomized_side_strategy_same_signal_mask():
    df = synth_ohlcv("1H", "2024-01-01", 600, seed=42)
    base = CrossoverStrategy(fast=5, slow=20)
    base_raw = base.raw_direction(df)
    wrapped = RandomizedSideStrategy(base, np.random.default_rng(0))
    raw = wrapped.raw_direction(df)
    mask_change = (base_raw != 0).to_numpy()
    # same non-zero positions
    assert ((raw != 0).to_numpy() == mask_change).all()
    # sides are drawn but preserved magnitudes
    assert set(np.unique(raw.to_numpy())) <= {-1, 0, 1}


def test_invalid_m_rejected(synth_1h):
    with pytest.raises(ValueError):
        run_randomized_baseline(
            synth_1h, lambda: CrossoverStrategy(), "BTCUSDT", "1H", RiskConfig(), CostConfig(),
            m_simulations=0, seed=1,
        )


def test_documented_notes_present(synth_1h):
    res = run_randomized_baseline(
        synth_1h, lambda: CrossoverStrategy(fast=5, slow=20), "BTCUSDT", "1H",
        RiskConfig(), CostConfig(), m_simulations=5, seed=7,
    )
    notes = " ".join(res.notes)
    assert "null" in notes and "(M + 1)" in notes
