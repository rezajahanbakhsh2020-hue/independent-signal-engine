"""Backtest the confirmed Ichimoku cloud + RSI + next-candle strategy on four real Binance pairs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from signal_engine.backtest import CostConfig, backtest
from signal_engine.data.binance_source import fetch_range
from signal_engine.metrics import compute_metrics
from signal_engine.risk import RiskConfig
from signal_engine.strategies import IchimokuRsiConfirmedStrategy

MARKETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")
TIMEFRAME = "1H"
LOOKBACK_DAYS = 60
INITIAL_EQUITY = 10_000.0


def rounded(value):
    return None if value is None else round(float(value), 6)


def main() -> int:
    now = pd.Timestamp.now(tz="UTC")
    end = now.floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0,
                      reward_risk_multiple=2.0, risk_fraction=0.01)
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    rows, errors = [], []

    for market in MARKETS:
        try:
            df = fetch_range(market, TIMEFRAME, start, end)
            df = df.loc[df.index + pd.Timedelta(hours=1) <= end]
            if len(df) < 200:
                raise RuntimeError(f"insufficient closed candles: {len(df)}")
            provenance = {
                "source": df.attrs.get("source", "unknown"),
                "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
                "coverage_start_utc": df.index[0].isoformat(),
                "coverage_end_utc": df.index[-1].isoformat(),
                "requested_start_utc": start.isoformat(),
                "requested_end_exclusive_utc": end.isoformat(),
                "closed_candles_only": True,
            }
            strategy = IchimokuRsiConfirmedStrategy()
            result = backtest(df, strategy, market, TIMEFRAME, risk, costs,
                              initial_equity=INITIAL_EQUITY, provenance=provenance)
            m = compute_metrics(result)
            rows.append({
                "market": market, "timeframe": TIMEFRAME,
                "strategy": strategy.name, "candles": result.n_candles,
                "start_utc": result.start_utc, "end_utc": result.end_utc,
                "source": provenance["source"],
                "total_return_pct": rounded(m.total_return_pct),
                "max_drawdown_pct": rounded(m.max_drawdown_pct),
                "net_total_pnl": rounded(m.net_total_pnl),
                "profit_factor": rounded(m.profit_factor),
                "win_rate_pct": rounded(None if m.win_rate is None else m.win_rate * 100),
                "trades": m.n_trades, "total_fees": rounded(m.total_fees),
                "insufficient_samples": m.insufficient_samples,
            })
        except Exception as exc:
            errors.append({"market": market, "error": f"{type(exc).__name__}: {exc}"})

    output = {
        "kind": "ichimoku_rsi_confirmed_crypto_backtest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_days_requested": LOOKBACK_DAYS, "timeframe": TIMEFRAME,
        "markets": list(MARKETS), "strategy": "ichimoku_rsi_confirmed",
        "strategy_parameters": {
            "tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52,
            "cloud_displacement": 26, "rsi_window": 14, "rsi_midline": 50,
            "confirmation": "next closed candle close beyond breakout candle high/low",
        },
        "initial_equity_per_test": INITIAL_EQUITY,
        "risk": vars(risk), "costs_bps_per_fill": vars(costs),
        "results": rows, "errors": errors,
        "limitations": [
            "Historical simulation only; not a guarantee of future returns.",
            "Only closed 1H candles are used; entries execute at the following candle open.",
            "Ichimoku spans are shifted back by their 26-candle display displacement to use the cloud actually visible at each candle, avoiding future-data leakage.",
            "RSI uses Wilder-style exponential smoothing; long requires RSI > 50 and short requires RSI < 50 on the confirmation candle.",
            "Low trade counts make performance estimates unreliable.",
        ],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/ichimoku_crypto_backtest.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv("artifacts/ichimoku_crypto_backtest.csv", index=False)
    print(json.dumps({"ok": not errors and len(rows) == len(MARKETS),
                      "markets_completed": [r["market"] for r in rows],
                      "tests_completed": len(rows), "errors": errors,
                      "output_json": "artifacts/ichimoku_crypto_backtest.json",
                      "output_csv": "artifacts/ichimoku_crypto_backtest.csv"}, indent=2))
    return 0 if not errors and len(rows) == len(MARKETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
