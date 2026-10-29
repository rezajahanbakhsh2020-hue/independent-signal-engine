"""Risk module: signal geometry validation, sizing, fail-closed behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.risk import RiskConfig, build_signal


def _df(n=30, open_px=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(n, open_px),
            "high": np.linspace(open_px + 1, open_px + 10, n),
            "low": np.linspace(open_px - 1, open_px - 10, n),
            "close": np.linspace(open_px + 0.5, open_px + 5, n),
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_buy_geometry_coherent():
    df = _df()
    atr = pd.Series(np.full(30, 1.0), index=df.index)
    sig = build_signal(1, 10, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s")
    assert sig is not None
    assert sig.stop < sig.entry < sig.target
    assert np.isfinite([sig.entry, sig.stop, sig.target]).all()
    assert sig.decision_ts > sig.signal_ts


def test_sell_geometry_coherent():
    df = _df()
    atr = pd.Series(np.full(30, 1.0), index=df.index)
    sig = build_signal(-1, 10, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s")
    assert sig is not None
    assert sig.target < sig.entry < sig.stop


def test_sizing_matches_risk_fraction():
    df = _df()
    atr = pd.Series(np.full(30, 1.0), index=df.index)
    risk = RiskConfig(atr_multiplier=2.0, risk_fraction=0.01)
    sig = build_signal(1, 10, df, risk, atr, 50_000.0, "BTCUSDT", "1H", "s")
    expected_qty = (0.01 * 50_000.0) / (2.0 * atr.iloc[10])
    assert sig.position_size == pytest.approx(expected_qty)


def test_zero_atr_returns_none():
    df = _df()
    atr = pd.Series(np.full(30, 0.0), index=df.index)
    assert build_signal(1, 10, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s") is None


def test_nan_atr_returns_none():
    df = _df()
    atr = pd.Series(np.full(30, np.nan), index=df.index)
    assert build_signal(1, 10, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s") is None


def test_negative_price_rejected():
    df = _df()
    df["open"] = -df["open"].abs()
    atr = pd.Series(np.full(30, 1.0), index=df.index)
    assert build_signal(1, 10, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s") is None


def test_signal_requires_exec_candle():
    df = _df(n=12)
    atr = pd.Series(np.full(12, 1.0), index=df.index)
    # signal on the final candle: no next candle to execute on
    assert build_signal(1, 11, df, RiskConfig(), atr, 10_000.0, "BTCUSDT", "1H", "s") is None


def test_invalid_risk_config_rejected():
    with pytest.raises(ValueError):
        RiskConfig(atr_multiplier=0)
    with pytest.raises(ValueError):
        RiskConfig(risk_fraction=0.99)
    with pytest.raises(ValueError):
        RiskConfig(reward_risk_multiple=0)


def test_min_atr_floor_forces_no_trade():
    df = _df()
    atr = pd.Series(np.full(30, 0.5), index=df.index)
    risk = RiskConfig(min_atr=1.0)
    assert build_signal(1, 10, df, risk, atr, 10_000.0, "BTCUSDT", "1H", "s") is None


def test_max_position_fraction_caps_notional():
    df = _df(open_px=10.0)
    atr = pd.Series(np.full(30, 0.1), index=df.index)
    risk = RiskConfig(risk_fraction=0.1, max_position_fraction=0.5)
    sig = build_signal(1, 10, df, risk, atr, 1_000.0, "BTCUSDT", "1H", "s")
    assert sig is not None
    # even after capping, geometry holds; notional cannot exceed cap
    assert sig.position_size * sig.entry <= 0.5 * 1_000.0 + 1e-9
