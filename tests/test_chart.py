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



def test_chart_crypto_cli_fetches_selected_symbol_and_writes_html(monkeypatch, tmp_path, capsys):
    from argparse import Namespace
    from signal_engine import cli
    from signal_engine.data import binance_source

    idx = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=80),
                        periods=80, freq="h", tz="UTC")
    close = np.linspace(100, 120, len(idx))
    candles = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(len(close), 100.0),
    }, index=idx)
    candles.attrs["source"] = "test:mock-binance"
    calls = []

    def fake_fetch_range(symbol, timeframe, start, end):
        calls.append((symbol, timeframe, start, end))
        return candles.copy()

    monkeypatch.setattr(binance_source, "fetch_range", fake_fetch_range)
    out = tmp_path / "charts" / "SOLUSDT_1H.html"
    args = Namespace(
        market="SOLUSDT", timeframe="1H",
        start=idx[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=idx[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_age_hours=None, max_candles=300, out=str(out),
        strategy="crossover", param=["fast=3", "slow=8"],
        atr_window=14, atr_multiplier=2.0, reward_risk=2.0,
        risk_fraction=0.01, fee_bps=2.0, spread_bps=2.0,
        slippage_bps=1.0, initial_equity=10000.0,
    )

    assert cli.cmd_chart_crypto(args) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert calls and calls[0][0:2] == ("SOLUSDT", "1H")
    assert str(calls[0][2].tz) == "UTC" and str(calls[0][3].tz) == "UTC"
    assert payload["ok"] is True
    assert payload["market"] == "SOLUSDT"
    assert out.is_file()
    html_page = out.read_text(encoding="utf-8")
    assert "SOLUSDT" in html_page
    assert "Latest decision" in html_page


def test_chart_crypto_cli_rejects_invalid_symbol_without_fetch(monkeypatch, capsys):
    from argparse import Namespace
    from signal_engine import cli

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("invalid symbol must not fetch or substitute a market")

    from signal_engine.data import binance_source
    monkeypatch.setattr(binance_source, "fetch_range", unexpected_fetch)
    args = Namespace(market="SOLBTC", timeframe="1H", max_candles=300)

    assert cli.cmd_chart_crypto(args) == 2
    assert '"ok": false' in capsys.readouterr().err.lower()
