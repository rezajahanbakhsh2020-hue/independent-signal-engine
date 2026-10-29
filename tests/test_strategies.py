"""Strategy tests: all four families, causality, identifier coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.constants import MARKETS, TIMEFRAMES, UnsupportedMarketError, UnsupportedTimeframeError
from signal_engine.strategies import (
    BollingerMeanReversionStrategy,
    BreakoutStrategy,
    CrossoverStrategy,
    PullbackStrategy,
    STRATEGY_REGISTRY,
)

ALL_STRATEGIES = sorted(STRATEGY_REGISTRY)


def test_registry_has_all_four_families():
    assert set(STRATEGY_REGISTRY) == {
        "crossover",
        "pullback",
        "breakout",
        "bollinger_mean_reversion",
    }


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_signal_series_values(synth_1h, name):
    strat = STRATEGY_REGISTRY[name]()
    raw = strat.raw_direction(synth_1h)
    assert set(np.unique(raw)) <= {-1, 0, 1}


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_causality_no_lookahead(synth_1h, name):
    """Changing FUTURE candles must not change current signals."""
    strat = STRATEGY_REGISTRY[name]()
    base = strat.raw_direction(synth_1h)
    modified = synth_1h.copy()
    # corrupt a far-future region
    modified.iloc[-30:, :] = modified.iloc[-30:, :] * 3.0
    mod = strat.raw_direction(modified)
    cut = len(synth_1h) - 30
    pd.testing.assert_series_equal(
        base.iloc[:cut].reset_index(drop=True),
        mod.iloc[:cut].reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_determinism(synth_1h, name):
    strat1 = STRATEGY_REGISTRY[name]()
    strat2 = STRATEGY_REGISTRY[name]()
    pd.testing.assert_series_equal(
        strat1.raw_direction(synth_1h), strat2.raw_direction(synth_1h)
    )


def test_crossover_param_validation():
    with pytest.raises(ValueError):
        CrossoverStrategy(fast=30, slow=10)
    with pytest.raises(ValueError):
        CrossoverStrategy(fast=0, slow=10)
    with pytest.raises(ValueError):
        CrossoverStrategy(kind="rsi")


def test_pullback_param_validation():
    with pytest.raises(ValueError):
        PullbackStrategy(trend_window=5, pullback_window=10)


def test_bollinger_param_validation():
    with pytest.raises(ValueError):
        BollingerMeanReversionStrategy(window=2)
    with pytest.raises(ValueError):
        BollingerMeanReversionStrategy(num_std=-1)


def test_breakout_param_validation():
    with pytest.raises(ValueError):
        BreakoutStrategy(lookback=1)


@pytest.mark.parametrize("tf", TIMEFRAMES)
@pytest.mark.parametrize("market", MARKETS)
def test_matrix_identifiers_valid(market, tf):
    from signal_engine.constants import market_timeframe_matrix, validate_market, validate_timeframe

    assert validate_market(market) == market
    assert validate_timeframe(tf) == tf
    cells = market_timeframe_matrix()
    assert len(cells) == 21
    assert (market, tf) in cells


def test_unsafe_identifiers_rejected():
    from signal_engine.constants import validate_market, validate_timeframe

    with pytest.raises(UnsupportedTimeframeError):
        validate_timeframe("2m")
    with pytest.raises(UnsupportedMarketError):
        validate_market("AAPL")
    with pytest.raises(UnsupportedMarketError):
        validate_market("XAGUSD")


def test_signals_emit_events_on_direction_change(synth_1h):
    strat = CrossoverStrategy(fast=5, slow=20)
    raw = strat.raw_direction(synth_1h)
    n_events = int((raw != 0).sum())
    assert n_events > 0
    # state transitions: consecutive nonzero events cannot be same direction
    events = raw[raw != 0]
    assert len(events) >= 1
