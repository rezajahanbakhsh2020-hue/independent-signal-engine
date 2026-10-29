"""Run a fixed, recent-data comparison of all registered strategies on four Binance spot pairs.

No parameter tuning is performed. All strategies use their documented defaults and
the same closed 1H candle window, risk configuration, and conservative transaction costs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from signal_engine.backtest import CostConfig, backtest
from signal_engine.data.binance_source import fetch_range
from signal_engine.metrics import compute_metrics
from signal_engine.risk import RiskConfig
from signal_engine.strategies import STRATEGY_REGISTRY

MARKETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")
TIMEFRAME = "1H"
LOOKBACK_DAYS = 60
INITIAL_EQUITY = 10_000.0


def main() -> int:
    now = pd.Timestamp.now(tz="UTC")
    end = now.floor("h")  # exclude the currently forming 1H candle
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0,
                      reward_risk_multiple=2.0, risk_fraction=0.01)
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    rows = []
    errors = []

    for market in MARKETS:
        try:
            df = fetch_range(market, TIMEFRAME, start, end)
            # Defensive closed-candle filter; do not backtest a partial candle.
            df = df.loc[df.index + pd.Timedelta(hours=1) <= end]
            if len(df) < 100:
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
            for name, cls in STRATEGY_REGISTRY.items():
                strategy = cls()  # fixed defaults; no tuning on the evaluation window
                result = backtest(
                    df, strategy, market, TIMEFRAME, risk, costs,
                    initial_equity=INITIAL_EQUITY, provenance=provenance,
                )
                m = compute_metrics(result)
                rows.append({
                    "market": market,
                    "timeframe": TIMEFRAME,
                    "strategy": name,
                    "candles": result.n_candles,
                    "start_utc": result.start_utc,
                    "end_utc": result.end_utc,
                    "source": provenance["source"],
                    "total_return_pct": _round(m.total_return_pct),
                    "max_drawdown_pct": _round(m.max_drawdown_pct),
                    "net_total_pnl": _round(m.net_total_pnl),
                    "profit_factor": _round(m.profit_factor),
                    "win_rate_pct": _round(None if m.win_rate is None else m.win_rate * 100),
                    "trades": m.n_trades,
                    "total_fees": _round(m.total_fees),
                    "insufficient_samples": m.insufficient_samples,
                    "n_signals_rejected_zero_atr": result.n_signals_rejected_zero_atr,
                    "n_signals_rejected_incoherent": result.n_signals_rejected_incoherent,
                })
        except Exception as exc:
            errors.append({"market": market, "error": f"{type(exc).__name__}: {exc}"})

    output = {
        "kind": "recent_crypto_strategy_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_days_requested": LOOKBACK_DAYS,
        "timeframe": TIMEFRAME,
        "markets": list(MARKETS),
        "strategies": list(STRATEGY_REGISTRY),
        "initial_equity_per_test": INITIAL_EQUITY,
        "strategy_parameters": "registered defaults; no optimization",
        "risk": vars(risk),
        "costs_bps_per_fill": vars(costs),
        "results": rows,
        "errors": errors,
        "interpretation": [
            "Descriptive historical simulation only; not a guarantee of future returns.",
            "Transaction costs include configured fees, spread and slippage.",
            "Treat results with fewer than 20 trades as low-sample evidence.",
            "The four assets and four strategies form 16 comparisons; choosing the best after comparison introduces selection bias.",
            "Only closed 1H candles are used; entry is at the next candle open under the engine's conservative fill rules.",
        ],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/recent_crypto_backtest.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv("artifacts/recent_crypto_backtest.csv", index=False)
    print(json.dumps({
        "ok": not errors and len(rows) == len(MARKETS) * len(STRATEGY_REGISTRY),
        "markets_completed": sorted({r["market"] for r in rows}),
        "tests_completed": len(rows),
        "errors": errors,
        "output_json": "artifacts/recent_crypto_backtest.json",
        "output_csv": "artifacts/recent_crypto_backtest.csv",
    }, indent=2))
    return 0 if not errors and len(rows) == len(MARKETS) * len(STRATEGY_REGISTRY) else 1


def _round(value):
    return None if value is None else round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
