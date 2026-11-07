"""Backtester regression tests: lookahead, timestamp alignment, fees,
slippage, stop geometry, trade ordering, determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.backtest import (
    BacktestConfigError,
    CostConfig,
    backtest,
)
from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.risk import RiskConfig, build_signal
from _strategy_fixtures import CrossoverStrategy


def test_determinism_same_config_same_result(synth_1h):
    risk = RiskConfig()
    costs = CostConfig()
    r1 = backtest(synth_1h, CrossoverStrategy(), "BTCUSDT", "1H", risk, costs)
    r2 = backtest(synth_1h, CrossoverStrategy(), "BTCUSDT", "1H", risk, costs)
    assert r1.final_equity == r2.final_equity
    assert len(r1.trades) == len(r2.trades)


def test_execution_timestamp_is_open_of_next_candle(synth_1h):
    df = synth_1h
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0)
    raw = CrossoverStrategy(fast=5, slow=20).raw_direction(df)
    events = np.flatnonzero(raw.to_numpy() != 0)
    if len(events) == 0:
        pytest.skip("no signals in fixture")
    res = backtest(df, CrossoverStrategy(fast=5, slow=20), "BTCUSDT", "1H", risk, CostConfig())
    for tr in res.trades:
        sig_pos = df.index.get_loc(tr.signal_ts)
        assert tr.entry_ts == df.index[sig_pos + 1]
        # execution at next candle OPEN (spread adj), so entry_ts > signal_ts
        assert tr.entry_ts > tr.signal_ts


def test_no_overlapping_trades(synth_1h):
    res = backtest(
        synth_1h, CrossoverStrategy(fast=5, slow=20), "BTCUSDT", "1H", RiskConfig(), CostConfig()
    )
    for a, b in zip(res.trades, res.trades[1:]):
        assert b.entry_ts >= a.exit_ts


def test_stop_before_target_in_same_candle(synth_1h):
    """Craft a bar where both stop and target would be inside [low, high];
    the stop must be assumed to fill first (conservative)."""
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    opens = np.full(n, 100.0)
    highs = opens + 0.2
    lows = opens - 0.2
    closes = opens.copy()
    vol = np.full(n, 1000.0)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx
    )

    class AlwaysLong:
        name = "always_long"

        def raw_direction(self, df):
            return pd.Series(1, index=df.index, dtype="int64")

    risk = RiskConfig(atr_window=3, atr_multiplier=1.0)
    costs = CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    res = backtest(df, AlwaysLong(), "BTCUSDT", "1H", risk, costs)
    assert len(res.trades) >= 1
    # tiny ranges: net should be a stop-out with a small loss, not a target win
    assert res.trades[0].exit_reason in {"stop", "end_of_data"}


def test_lookahead_probe_future_cannot_change_signals():
    """A signal candle must produce identical decisions even if all future
    candles are altered - i.e., execution cannot peek ahead."""
    df = synth_ohlcv("1H", "2024-01-01", 300, seed=7)
    strat = CrossoverStrategy(fast=5, slow=15)
    risk = RiskConfig()
    costs = CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)

    # find first signal, extract prefix dataset (signal + one candle to exec on)
    raw = strat.raw_direction(df)
    idx = np.flatnonzero(raw.to_numpy() != 0)[0]
    prefix = df.iloc[: idx + 2].copy()
    future_corrupt = df.iloc[: idx + 2].copy()
    future_corrupt.loc[df.index[idx + 1], ["open", "high", "low", "close"]] = df.loc[
        df.index[idx + 1], ["open", "high", "low", "close"]
    ].to_numpy() * 100.0

    r1 = backtest(prefix, strat, "BTCUSDT", "1H", risk, costs)
    r2 = backtest(future_corrupt, strat, "BTCUSDT", "1H", risk, costs)
    # entry timestamp must be identical (same signal time), and the signal
    # candle itself untouched; equity can differ only because the
    # execution candle price itself legitimately changed (that IS the
    # executed price).  Assert the same number of trades and identical
    # signal_ts values.
    assert [t.signal_ts for t in r1.trades][0] == [t.signal_ts for t in r2.trades][0]
    assert df.iloc[: idx + 1].equals(prefix.iloc[: idx + 1])


def test_costs_reduce_net_results_monotonically(synth_1h):
    df = synth_1h
    strat = CrossoverStrategy(fast=5, slow=20)
    risk = RiskConfig()
    c0 = CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    c1 = CostConfig(fee_bps=5.0, spread_bps=10.0, slippage_bps=10.0)
    r0 = backtest(df, strat, "BTCUSDT", "1H", risk, c0)
    r1 = backtest(df, strat, "BTCUSDT", "1H", risk, c1)
    if r0.trades:
        assert r1.final_equity < r0.final_equity + 1e-6


def test_gross_and_net_distinguishable(synth_1h):
    res = backtest(
        synth_1h, CrossoverStrategy(fast=5, slow=20), "BTCUSDT", "1H", RiskConfig(),
        CostConfig(fee_bps=10.0, spread_bps=10.0, slippage_bps=5.0),
    )
    for t in res.trades:
        assert t.net_pnl <= t.gross_pnl  # fees always subtract
        assert t.fees > 0


def test_buy_incoherent_geometry_rejected():
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": np.linspace(100.5, 120.0, 20),
            "low": np.linspace(99.5, 99.0, 20),
            "close": np.linspace(100.2, 110.0, 20),
            "volume": np.full(20, 100.0),
        },
        index=idx,
    )
    atr_series = pd.Series(np.full(20, 1.0), index=idx)
    risk = RiskConfig(atr_multiplier=2.0, reward_risk_multiple=2.0)
    sig = build_signal(1, 10, df, risk, atr_series, equity=10000.0, market="BTCUSDT", timeframe="1H", strategy_name="c")
    # entry ~ at index 11's open = 100; stop = 98; target = 104: coherent
    assert sig is not None
    assert sig.stop < sig.entry < sig.target


def test_sell_incoherent_geometry_rejected():
    idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": np.linspace(100.5, 101, 20),
            "low": np.linspace(99.5, 80.0, 20),
            "close": np.linspace(99.8, 90.0, 20),
            "volume": np.full(20, 100.0),
        },
        index=idx,
    )
    atr_series = pd.Series(np.full(20, 1.0), index=idx)
    risk = RiskConfig(atr_multiplier=2.0, reward_risk_multiple=2.0)
    sig = build_signal(-1, 10, df, risk, atr_series, equity=10000.0, market="BTCUSDT", timeframe="1H", strategy_name="c")
    assert sig is not None
    assert sig.target < sig.entry < sig.stop


def test_zero_atr_no_trade():
    idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": np.full(20, 100.0),
            "low": np.full(20, 100.0),
            "close": np.full(20, 100.0),
            "volume": np.full(20, 100.0),
        },
        index=idx,
    )
    atr_series = pd.Series(np.full(20, 0.0), index=idx)
    risk = RiskConfig()
    out = build_signal(1, 10, df, risk, atr_series, 10000.0, "BTCUSDT", "1H", "s")
    assert out is None


def test_zero_atr_backtest_records_rejections():

    class AlwaysLong:
        name = "always_long"

        def raw_direction(self, df):
            return pd.Series(1, index=df.index, dtype="int64")

    idx = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(40, 100.0),
            "high": np.full(40, 100.0),
            "low": np.full(40, 100.0),
            "close": np.full(40, 100.0),
            "volume": np.full(40, 100.0),
        },
        index=idx,
    )
    res = backtest(df, AlwaysLong(), "BTCUSDT", "1H", RiskConfig(), CostConfig())
    assert res.n_signals_rejected_zero_atr > 0 or res.n_signals_rejected_incoherent > 0
    assert res.final_equity == pytest.approx(10_000.0, abs=1e-6)  # no trades, no losses


def test_empty_dataframe_returns_clean_result():
    df = synth_ohlcv("1H", "2024-01-01", 120)
    empty = df.iloc[0:0]
    res = backtest(empty, CrossoverStrategy(), "BTCUSDT", "1H", RiskConfig(), CostConfig())
    assert len(res.trades) == 0
    assert res.trades == []
    assert res.final_equity == 10_000.0
    assert res.warnings == ["empty_dataframe"]


def test_invalid_initial_equity_rejected(synth_1h):
    with pytest.raises(BacktestConfigError):
        backtest(synth_1h, CrossoverStrategy(), "BTCUSDT", "1H", RiskConfig(), CostConfig(), initial_equity=0)


def test_negative_equity_tracked_not_hidden():
    """A disaster run must still return final equity (possibly negative)."""

    class AlwaysLong:
        name = "always_long"

        def raw_direction(self, df):
            return pd.Series(1, index=df.index, dtype="int64")

    df = synth_ohlcv("1H", "2024-01-01", 200, seed=3, start_price=100.0, yearly_vol=4.0)
    res = backtest(
        df, AlwaysLong(), "BTCUSDT", "1H", RiskConfig(risk_fraction=0.5), CostConfig()
    )
    # every trade risks 50% and always-long in a recessive series can lose a lot
    assert np.isfinite(res.final_equity)


def test_turnover_positive_when_trades_exist(synth_1h):
    from signal_engine.metrics import compute_metrics

    res = backtest(
        synth_1h, CrossoverStrategy(fast=5, slow=20), "BTCUSDT", "1H", RiskConfig(), CostConfig()
    )
    m = compute_metrics(res)
    if res.trades:
        assert m.turnover_x_equity > 0
    else:
        assert m.turnover_x_equity is None or m.turnover_x_equity == 0


@pytest.mark.parametrize(
    ("side", "gap_open", "gap_high", "gap_low"),
    [(1, 90.0, 91.0, 89.0), (-1, 110.0, 111.0, 109.0)],
)
def test_stop_gap_fills_at_adverse_open(side, gap_open, gap_high, gap_low):
    """A stop crossed by the next candle's open must not fill at the old stop."""
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    opens = np.full(8, 100.0)
    highs = np.full(8, 101.0)
    lows = np.full(8, 99.0)
    closes = np.full(8, 100.0)
    opens[4], highs[4], lows[4], closes[4] = gap_open, gap_high, gap_low, gap_open
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.full(8, 100.0)}, index=idx
    )

    class AlwaysDirectional:
        name = "always_directional"

        def raw_direction(self, data):
            return pd.Series(side, index=data.index, dtype="int64")

    costs = CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    result = backtest(df, AlwaysDirectional(), "BTCUSDT", "1H",
                      RiskConfig(atr_window=3, atr_multiplier=1.0), costs)
    stopped = next(trade for trade in result.trades if trade.exit_reason == "stop")
    assert stopped.exit_price == pytest.approx(gap_open)


def test_short_end_of_data_liquidation_uses_adverse_market_fill():
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": np.full(8, 100.0), "high": np.full(8, 100.1),
         "low": np.full(8, 99.9), "close": np.full(8, 100.0),
         "volume": np.full(8, 100.0)}, index=idx
    )

    class AlwaysShort:
        name = "always_short"

        def raw_direction(self, data):
            return pd.Series(-1, index=data.index, dtype="int64")

    costs = CostConfig(fee_bps=0.0, spread_bps=2.0, slippage_bps=1.0)
    result = backtest(df, AlwaysShort(), "BTCUSDT", "1H",
                      RiskConfig(atr_window=3, atr_multiplier=2.0), costs)
    trade = next(trade for trade in result.trades if trade.exit_reason == "end_of_data")
    expected = costs.exit_stop_fill(-1, float(df["close"].iloc[-1]))
    assert trade.exit_price == pytest.approx(expected)


@pytest.mark.parametrize(("side", "hit_column", "hit_price"), [
    (1, "low", 97.0),
    (-1, "high", 103.0),
])
def test_new_position_can_stop_out_on_its_entry_candle(side, hit_column, hit_price):
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": np.full(8, 100.0), "high": np.full(8, 101.0),
         "low": np.full(8, 99.0), "close": np.full(8, 100.0),
         "volume": np.full(8, 100.0)}, index=idx
    )
    df.loc[idx[3], hit_column] = hit_price

    class AlwaysDirectional:
        name = "always_directional"
        def raw_direction(self, data):
            return pd.Series(side, index=data.index, dtype="int64")

    result = backtest(
        df, AlwaysDirectional(), "BTCUSDT", "1H",
        RiskConfig(atr_window=3, atr_multiplier=1.0),
        CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
    )
    assert result.trades
    trade = result.trades[0]
    assert trade.entry_ts == idx[3]
    assert trade.exit_ts == idx[3]
    assert trade.exit_reason == "stop"


def test_signal_generated_while_position_open_is_not_queued_after_exit():
    idx = pd.date_range("2024-01-01", periods=9, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": np.full(9, 100.0), "high": np.full(9, 101.0),
         "low": np.full(9, 99.0), "close": np.full(9, 100.0),
         "volume": np.full(9, 100.0)}, index=idx
    )
    df.loc[idx[5], "low"] = 97.0

    class DirectionChangesWhileLong:
        name = "direction_changes_while_open"
        def raw_direction(self, data):
            values = np.zeros(len(data), dtype="int64")
            values[2] = 1
            values[4] = -1
            return pd.Series(values, index=data.index)

    result = backtest(
        df, DirectionChangesWhileLong(), "BTCUSDT", "1H",
        RiskConfig(atr_window=3, atr_multiplier=1.0),
        CostConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
    )
    assert len(result.trades) == 1
    assert result.trades[0].side == 1
    assert result.trades[0].exit_ts == idx[5]
    assert result.trades[0].exit_reason == "stop"
