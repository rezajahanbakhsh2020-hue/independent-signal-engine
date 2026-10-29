"""Tests for the latest-decision (current-signal) check.

Covers: BUY/SELL/NO TRADE reachability with real strategy data, decision
semantics (act at t from the direction at t-1, execute at the last
candle's open), no-lookahead guarantees, fail-closed geometry reasons,
and CLI wiring including rejection of invalid CSVs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from signal_engine.cli import main as cli_main
from signal_engine.data.csv_source import read_csv_ohlcv
from signal_engine.latest_signal import latest_decision_report
from signal_engine.risk import RiskConfig, build_signal
from signal_engine.strategies import STRATEGY_REGISTRY, Strategy

LABEL = {-1: "SELL", 0: "NO TRADE", 1: "BUY"}


def _candles(prices: list[float], period="60min", start="2025-06-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq=period, tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p + 1.0, "low": p - 1.0, "close": p, "volume": 100.0},
        index=idx,
    )


def _crossover(fast=2, slow=4):
    return STRATEGY_REGISTRY["crossover"](fast=fast, slow=slow)


def _report(df, strat=None, risk=None, market="BTCUSDT"):
    return latest_decision_report(
        df, strat or _crossover(), risk or RiskConfig(), market, "1H", provenance="csv:test"
    )


class FixedDirectionStrategy(Strategy):
    """Test stub: returns a pre-set causal direction series (no prior knowledge)."""

    name = "fixed_direction"

    def __init__(self, directions: list[int]) -> None:
        self._directions = list(directions)

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        assert len(self._directions) == len(df), "stub must be sized to the frame"
        return pd.Series(self._directions, index=df.index, dtype="int64")


def test_buy_reachable_after_real_crossover_flip():
    df = _candles(list(np.linspace(160, 100, 60)) + list(np.linspace(100, 170, 3)))
    rep = _report(df, _crossover())
    d = rep["decision"]
    assert d["label"] == "BUY"
    assert d["side"] == 1
    assert d["stop"] < d["entry"] < d["target"]
    assert d["signal_ts"] < d["decision_ts"]
    assert d["signal_ts"] == df.index[-2].isoformat()
    assert d["decision_ts"] == df.index[-1].isoformat()
    assert d["entry"] == pytest.approx(df["open"].iloc[-1])
    assert d["reward_risk_per_unit"] == pytest.approx(2.0)


def test_sell_reachable_after_real_crossover_flip():
    df = _candles(list(np.linspace(100, 200, 60)) + list(np.linspace(200, 90, 3)))
    rep = _report(df, _crossover())
    d = rep["decision"]
    assert d["label"] == "SELL"
    assert d["side"] == -1
    assert d["target"] < d["entry"] < d["stop"]
    assert d["signal_ts"] == df.index[-2].isoformat()
    assert d["decision_ts"] == df.index[-1].isoformat()
    assert d["entry"] == pytest.approx(df["open"].iloc[-1])


def test_no_trade_without_active_transition():
    # Persistent up-trend: the crossover fires once at warm-up end only,
    # far from the end; the penultimate signal candle carries no transition.
    df = _candles(list(np.linspace(100, 160, 40)) + [161.0])
    rep = _report(df)
    d = rep["decision"]
    assert d["label"] == "NO TRADE"
    assert d["reason"] == "no_active_transition_signal"
    assert d["decision_ts"] == df.index[-1].isoformat()
    assert rep["pending_raw_direction"] == "NO TRADE"


def test_flip_on_last_candle_is_pending_only():
    # The transition lands on the final candle: it would be acted on by a
    # *future* candle that does not exist yet, so the acted decision is NO
    # TRADE and only the pending direction is reported.
    df = _candles(list(np.linspace(100, 200, 60)) + list(np.linspace(200, 90, 2)))
    assert LABEL[int(_crossover().raw_direction(df).iloc[-1])] == "SELL"
    rep = _report(df)
    assert rep["pending_raw_direction"] == "SELL"
    assert rep["decision"]["label"] == "NO TRADE"
    assert rep["decision"]["reason"] == "no_active_transition_signal"


def test_no_lookahead_last_candle_does_not_change_decision():
    # The acted decision must derive only from candles up to N-2.  A wild
    # final candle changes only the execution reference (its open), never
    # the side, ATR provenance, or stop distance.
    n = 40
    dirs = [-1] * (n - 2) + [1, 0]  # BUY transition on the penultimate candle
    risk = RiskConfig()
    df_a = _candles(list(np.linspace(100, 160, n - 1)) + [105.0])
    df_b = _candles(list(np.linspace(100, 160, n - 1)) + [1000.0])
    strat = FixedDirectionStrategy(dirs)
    rep_a = _report(df_a, strat, risk)
    rep_b = _report(df_b, strat, risk)
    for rep, last_open in ((rep_a, 105.0), (rep_b, 1000.0)):
        d = rep["decision"]
        assert d["label"] == "BUY"
        assert d["entry"] == pytest.approx(last_open)
        assert d["stop_distance_in_atr"] == 2.0
    assert rep_a["decision"]["signal_ts"] == rep_b["decision"]["signal_ts"]
    # identical ATR provenance -> identical stop/target distance from entry
    assert rep_a["decision"]["stop"] - rep_a["decision"]["entry"] == pytest.approx(
        rep_b["decision"]["stop"] - rep_b["decision"]["entry"]
    )


def test_risk_geometry_rejection_fail_closed():
    # min_atr above every achievable ATR forces the fail-closed branch:
    # explicit NO TRADE and never fabricated entry/stop/target levels.
    df = _candles(list(np.linspace(100, 200, 60)) + list(np.linspace(200, 90, 3)))
    rep = _report(df, _crossover(), RiskConfig(atr_window=3, min_atr=1e12))
    d = rep["decision"]
    assert d["label"] == "NO TRADE"
    assert d["reason"] == "risk_geometry_rejected_zero_or_insufficient_atr"
    assert "entry" not in d and "stop" not in d and "target" not in d


def test_insufficient_candles_fail_closed():
    rep = _report(_candles([100.0]))
    d = rep["decision"]
    assert d["label"] == "NO TRADE"
    assert d["reason"].startswith("insufficient_candles")
    assert rep["pending_raw_direction"] is None


def test_bollinger_raw_semantics_supported():
    df = _candles(list(np.linspace(100.0, 100.0, 60)) + [80.0, 90.0])
    strat = STRATEGY_REGISTRY["bollinger_mean_reversion"](window=20, num_std=2.0)
    risk = RiskConfig(atr_window=5)
    rep = _report(df, strat, risk)
    assert rep["decision"]["label"] in {"BUY", "SELL", "NO TRADE"}
    assert rep["research_only"].startswith("offline")


def test_build_signal_zero_atr_is_single_geometry_authority():
    df = _candles(list(np.linspace(100, 160, 40)))
    df.iloc[-1, df.columns.get_loc("open")] = -5.0  # impossible price
    risk = RiskConfig()
    atr_series = pd.Series(0.0, index=df.index)
    sig = build_signal(1, len(df) - 2, df, risk, atr_series, 10_000.0, "BTCUSDT", "1H", "crossover")
    assert sig is None


def test_read_csv_matches_direct_call(tmp_path):
    prices = list(np.linspace(100, 200, 60)) + list(np.linspace(200, 90, 3))
    df = _candles(prices, period="60min")
    df.index.name = "timestamp"
    path = tmp_path / "x.csv"
    df.reset_index().to_csv(path, index=False)
    loaded, _rep = read_csv_ohlcv(str(path), expected_spacing="1H")
    strat, risk = _crossover(), RiskConfig()
    direct = _report(df, strat, risk)
    via_csv = _report(loaded, strat, risk)
    assert json.dumps(via_csv["decision"], sort_keys=True) == json.dumps(
        direct["decision"], sort_keys=True
    )


def test_cli_signal_on_real_csv_exit_zero(tmp_path, capsys):
    prices = list(np.linspace(100, 200, 60)) + list(np.linspace(200, 90, 3))
    df = _candles(prices, period="60min")
    df.index.name = "timestamp"
    path = tmp_path / "XAUUSD_1H.csv"
    df.reset_index().to_csv(path, index=False)
    rc = cli_main(
        [
            "signal", "--market", "XAUUSD", "--timeframe", "1H",
            "--csv", str(path), "--strategy", "crossover",
            "--param", "fast=2", "--param", "slow=4",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["decision"]["label"] == "SELL"
    assert payload["provenance"] == f"csv:{path}"


def test_cli_signal_rejects_bad_csv_fail_closed(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,open,high,low,close,volume\nnot-a-time,1,1,1,1,0\n")
    rc = cli_main(
        ["signal", "--market", "BTCUSDT", "--timeframe", "1H", "--csv", str(bad), "--strategy", "crossover"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert "NO TRADE" in payload["status"]


def test_cli_signal_missing_source_fails(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli_main(["signal", "--market", "BTCUSDT", "--timeframe", "1H", "--strategy", "crossover"])
