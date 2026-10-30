"""Strategy allow-list and causal signal contract checks (not executed in this change)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.constants import (
    MARKETS, TIMEFRAMES, UnsupportedMarketError, UnsupportedTimeframeError,
)
from signal_engine.strategies import (
    BollingerMeanReversionStrategy,
    RSIReversalStrategy,
    STRATEGY_REGISTRY,
)

ALL_STRATEGIES = sorted(STRATEGY_REGISTRY)


def test_registry_contains_only_the_two_selected_winners():
    assert set(STRATEGY_REGISTRY) == {"Bollinger_Reversion", "RSI_Reversal"}


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_signal_series_values(synth_1h, name):
    raw = STRATEGY_REGISTRY[name]().raw_direction(synth_1h)
    assert set(np.unique(raw)) <= {-1, 0, 1}


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_causality_no_lookahead(synth_1h, name):
    strategy = STRATEGY_REGISTRY[name]()
    base = strategy.raw_direction(synth_1h)
    modified = synth_1h.copy()
    modified.iloc[-30:, :] = modified.iloc[-30:, :] * 3.0
    changed = strategy.raw_direction(modified)
    cut = len(synth_1h) - 30
    pd.testing.assert_series_equal(
        base.iloc[:cut].reset_index(drop=True),
        changed.iloc[:cut].reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_determinism(synth_1h, name):
    first = STRATEGY_REGISTRY[name]()
    second = STRATEGY_REGISTRY[name]()
    pd.testing.assert_series_equal(
        first.raw_direction(synth_1h), second.raw_direction(synth_1h)
    )


def test_bollinger_parameter_validation():
    with pytest.raises(ValueError):
        BollingerMeanReversionStrategy(window=2)
    with pytest.raises(ValueError):
        BollingerMeanReversionStrategy(num_std=-1)


def test_rsi_reversal_uses_original_fixed_period():
    assert RSIReversalStrategy().period == 14
    with pytest.raises(ValueError):
        RSIReversalStrategy(period=10)


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
@pytest.mark.parametrize("market", MARKETS)
def test_matrix_identifiers_valid(market, timeframe):
    from signal_engine.constants import (
        market_timeframe_matrix, validate_market, validate_timeframe,
    )
    assert validate_market(market) == market
    assert validate_timeframe(timeframe) == timeframe
    cells = market_timeframe_matrix()
    assert len(cells) == 21
    assert (market, timeframe) in cells


def test_unsafe_identifiers_rejected():
    from signal_engine.constants import validate_market, validate_timeframe
    with pytest.raises(UnsupportedTimeframeError):
        validate_timeframe("2m")
    with pytest.raises(UnsupportedMarketError):
        validate_market("AAPL")
    with pytest.raises(UnsupportedMarketError):
        validate_market("XAGUSD")


def test_selected_strategy_signals_are_sparse_events(synth_1h):
    for strategy in (BollingerMeanReversionStrategy(), RSIReversalStrategy()):
        raw = strategy.raw_direction(synth_1h)
        assert set(np.unique(raw)) <= {-1, 0, 1}
