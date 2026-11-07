#!/usr/bin/env python3
"""Offline ACB hypothesis check against the two existing signal rules.

Research-only entry point. Does not modify/register production strategies,
dashboard, adapters, risk logic, or the production backtester.
Crypto spot only: BTCUSDT and ETHUSDT, long-only, Binance public klines.
Uses the existing engine's execution/risk/cost model for comparability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.backtest import CostConfig, backtest
from signal_engine.data.binance_source import fetch_range
from signal_engine.indicators import atr, bollinger, rsi_wilder
from signal_engine.risk import RiskConfig
from signal_engine.strategies.bollinger import BollingerMeanReversionStrategy
from signal_engine.strategies.rsi_reversal import RSIReversalStrategy


@dataclass
class LongOnlyRule:
    name: str
    kind: str

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        raw = pd.Series(0, index=df.index, dtype="int64")
        if self.kind == "acb":
            # Prior-bar channel only: breakout is confirmed at close, entered next open.
            mid = close.rolling(20, min_periods=20).mean()
            channel_at_signal = mid + 2.0 * atr(df["high"], df["low"], close, 14)
            prev_close = close.shift(1)
            prev_channel = channel_at_signal.shift(1)
            buy = (prev_close <= prev_channel) & (close > channel_at_signal)
            raw[buy & channel_at_signal.notna() & prev_channel.notna()] = 1
        elif self.kind == "bollinger":
            raw = BollingerMeanReversionStrategy().raw_direction(df).clip(lower=0)
        elif self.kind == "rsi":
            raw = RSIReversalStrategy().raw_direction(df).clip(lower=0)
        else:
            raise ValueError(self.kind)
        return raw


def metrics(result, oos_start: pd.Timestamp, initial_equity: float) -> dict:
    trades = [t for t in result.trades if t.signal_ts >= oos_start]
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [-t.net_pnl for t in trades if t.net_pnl < 0]
    eq = result.equity.loc[result.equity.index >= oos_start]
    if len(eq):
        eq = eq / float(eq.iloc[0]) * initial_equity
        peak = eq.cummax()
        dd = (eq / peak - 1.0).min() * 100.0
        ending = float(eq.iloc[-1])
    else:
        dd, ending = 0.0, initial_equity
    gross_profit, gross_loss = sum(wins), sum(losses)
    return {
        "trades": len(trades),
        "net_return_pct": round((ending / initial_equity - 1) * 100, 3),
        "max_drawdown_pct": round(float(dd), 3),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (None if not wins else "inf"),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else 0.0,
        "net_pnl": round(sum(t.net_pnl for t in trades), 4),
    }


def main() -> None:
    now = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    start = now - pd.Timedelta(days=3 * 365)
    markets = ("BTCUSDT", "ETHUSDT")
    timeframes = ("1H", "4H")
    rules = (
        LongOnlyRule("ACB_ATR_Channel_Breakout", "acb"),
        LongOnlyRule("Bollinger_Reversion", "bollinger"),
        LongOnlyRule("RSI_Reversal", "rsi"),
    )
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0, reward_risk_multiple=2.0,
                      risk_fraction=0.01, max_position_fraction=1.0)
    initial = 10_000.0
    rows = []
    for market in markets:
        for timeframe in timeframes:
            print(f"FETCH {market} {timeframe} {start.isoformat()} -> {now.isoformat()}", flush=True)
            df = fetch_range(market, timeframe, start, now)
            if len(df) < 1000:
                raise RuntimeError(f"Insufficient verified history: {market} {timeframe}: {len(df)} candles")
            split = int(len(df) * 0.70)
            if split < 250 or len(df) - split < 100:
                raise RuntimeError(f"Insufficient chronological OOS sample: {market} {timeframe}")
            oos_start = df.index[split]
            # Keep pre-OOS candles solely for indicator warm-up; suppress all IS entries.
            for rule in rules:
                class OOSOnly:
                    name = rule.name
                    def raw_direction(self, frame):
                        signals = rule.raw_direction(frame)
                        return signals.where(frame.index >= oos_start, 0)
                result = backtest(
                    df, OOSOnly(), market, timeframe, risk, costs,
                    initial_equity=initial,
                    provenance={
                        "source": df.attrs.get("source", "unknown"),
                        "provider": "Binance Spot",
                        "symbol_mapping": df.attrs.get("symbol_mapping"),
                        "timeframe": timeframe,
                        "coverage_start_utc": df.index[0].isoformat(),
                        "coverage_end_utc": df.index[-1].isoformat(),
                        "oos_start_utc": oos_start.isoformat(),
                        "split": "first 70% indicator warmup only; final 30% OOS",
                    },
                )
                row = {
                    "market": market, "timeframe": timeframe,
                    "strategy": rule.name, "candles": len(df),
                    "oos_start_utc": oos_start.isoformat(),
                    **metrics(result, oos_start, initial),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path("research/results")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out / f"acb_comparison_{stamp}.csv"
    json_path = out / f"acb_comparison_{stamp}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "3y Binance Spot; 70% warmup/IS, final 30% OOS; long-only; next-open entries; stop-first; 2R target; 2x ATR stop; 2bps fee + 2bps spread + 1bps slippage per fill",
        "results": rows,
        "warning": "One fixed hypothesis test, not proof of durable edge. No parameter search; OOS is still a single historical period, not a guarantee.",
    }, indent=2), encoding="utf-8")
    print(f"\nSAVED {csv_path}\nSAVED {json_path}")
    print("No orders placed. No production files changed.")


if __name__ == "__main__":
    main()
