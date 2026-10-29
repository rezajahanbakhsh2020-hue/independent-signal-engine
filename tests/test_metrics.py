"""Metrics tests: explicit denominators, zero/negative denominators,
concentration definition, drawdown, exposure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.metrics import compute_metrics, drawdown_fraction, hhi_concentration


def test_hhi_concentration_single_positive_trade():
    assert hhi_concentration([10.0]) == pytest.approx(1.0)


def test_hhi_two_equal_gains():
    assert hhi_concentration([1.0, 1.0]) == pytest.approx(0.5)


def test_hhi_one_dominant_gain():
    assert hhi_concentration([9.0, 1.0]) == pytest.approx(0.81 + 0.01)


def test_hhi_losses_excluded_from_denominator():
    # documented: losses do not share the denominator
    assert hhi_concentration([1.0, 1.0, -5.0]) == pytest.approx(0.5)


def test_hhi_all_losses_returns_none():
    assert hhi_concentration([-1.0, -2.0]) is None


def test_hhi_empty_returns_none():
    assert hhi_concentration([]) is None


def test_hhi_zero_denominator_returns_none():
    assert hhi_concentration([0.0, 0.0]) is None


def test_drawdown_fraction_correct():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    eq = pd.Series([100.0, 110.0, 99.0, 121.0, 110.0], index=idx)
    dd = drawdown_fraction(eq)
    assert dd.iloc[2] == pytest.approx(99.0 / 110.0 - 1.0)
    assert dd.min() < 0
    assert dd.iloc[0] == pytest.approx(0.0)


def test_drawdown_on_negative_equity_no_fake_percentage():
    # equity crosses zero: dimensionless drawdown is guarded (no fake %)
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    eq = pd.Series([100.0, -50.0, -40.0, 10.0], index=idx)
    dd = drawdown_fraction(eq)
    assert np.isfinite(dd).all()
    assert (dd <= 0).all()


def test_compute_metrics_no_trades(synth_1h):
    from signal_engine.backtest import CostConfig, backtest
    from signal_engine.risk import RiskConfig

    # tiny data guaranteed to produce zero signals
    df = synth_1h.iloc[: 25]
    res = backtest(df, __import__("signal_engine.strategies", fromlist=["CrossoverStrategy"]).CrossoverStrategy(fast=10, slow=30), "BTCUSDT", "1H", RiskConfig(), CostConfig())
    m = compute_metrics(res)
    assert m.n_trades == 0
    assert m.win_rate is None
    assert m.profit_factor is None
    assert m.trade_hhi_concentration is None
    assert m.insufficient_samples is True


def test_compute_metrics_profit_factor_zero_when_no_gains():
    """A run with only losses must not fabricate an infinite profit factor."""
    from signal_engine.backtest import BacktestResult
    from signal_engine.backtest import Trade

    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    t1 = Trade(side=1, market="X", timeframe="1H", strategy="s", signal_ts=idx[0],
               entry_ts=idx[1], entry_price=100.0, stop_price=99.0, target_price=102.0,
               qty=1.0, exit_ts=idx[2], exit_price=99.0, exit_reason="stop",
               gross_pnl=-1.0, fees=0.1)
    eq = pd.Series([10000.0, 9999.0, 9998.9, 9998.9], index=idx)
    res = BacktestResult(
        market="X", timeframe="1H", strategy="s", start_utc="", end_utc="",
        n_candles=4, initial_equity=10000.0, final_equity=9998.9,
        equity=eq, trades=[t1])
    m = compute_metrics(res)
    assert m.profit_factor == 0.0
    assert m.n_losses == 1
    assert m.n_gains == 0
    assert m.win_rate == 0.0


def test_compute_metrics_trade_concentration_matches_hhi():
    from signal_engine.backtest import BacktestResult, Trade

    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")

    def mk(net, ts_i, ts_e, price_in, price_out):
        return Trade(side=1, market="X", timeframe="1H", strategy="s",
                     signal_ts=idx[ts_i], entry_ts=idx[ts_i + 1], entry_price=100.0,
                     stop_price=99.0, target_price=101.0, qty=1.0, exit_ts=idx[ts_e],
                     exit_price=price_out, exit_reason="target", gross_pnl=net, fees=0.0)

    trades = [mk(5.0, 0, 2, 100, 105), mk(5.0, 2, 4, 100, 105)]
    eq = pd.Series([10000.0] * 6, index=idx)
    res = BacktestResult(market="X", timeframe="1H", strategy="s", start_utc="", end_utc="",
                         n_candles=6, initial_equity=10000.0, final_equity=10010.0,
                         equity=eq, trades=trades)
    m = compute_metrics(res)
    assert m.trade_hhi_concentration == pytest.approx(0.5)
    assert m.n_gains == 2
    assert m.denominator_notes.startswith("concentration dimension=trades")
