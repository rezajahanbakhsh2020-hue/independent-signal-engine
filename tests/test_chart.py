from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.chart import build_chart_html
from signal_engine.constants import validate_crypto_symbol
from signal_engine.strategies import STRATEGY_REGISTRY


def test_crypto_symbol_validation_accepts_more_than_btc_and_eth():
    assert validate_crypto_symbol(" solusdt ") == "SOLUSDT"
    assert validate_crypto_symbol("ADAUSDT") == "ADAUSDT"
    assert validate_crypto_symbol("DOGEUSDT") == "DOGEUSDT"


def test_crypto_symbol_validation_rejects_non_usdt_and_invalid_values():
    import pytest
    from signal_engine.constants import UnsupportedMarketError

    for symbol in ("XAUUSD", "SOLBTC", "USDT", "", "SOL/USDT", "SOLUSDT;"):
        with pytest.raises(UnsupportedMarketError):
            validate_crypto_symbol(symbol)


def test_chart_html_contains_market_timeframe_and_decision():
    idx = pd.date_range("2025-01-01", periods=80, freq="1h", tz="UTC")
    close = np.r_[np.linspace(100, 80, 40), np.linspace(80, 130, 40)]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(len(close), 100.0),
        },
        index=idx,
    )
    strategy = STRATEGY_REGISTRY["crossover"](fast=3, slow=8)
    report = {
        "decision": {"label": "BUY", "entry": float(close[-1])},
        "pending_raw_direction": "NO TRADE",
    }
    page = build_chart_html(
        df,
        market="SOLUSDT",
        timeframe="1H",
        strategy_name="crossover",
        strategy_params={"fast": 3, "slow": 8},
        strategy=strategy,
        latest_decision=report,
    )
    assert "<!doctype html>" in page.lower()
    assert "SOLUSDT" in page
    assert "1H" in page
    assert "Latest decision" in page
    assert "BUY" in page
    assert "<svg" in page
    assert "not orders" in page
