"""BTC-only 60-day Ichimoku confirmation comparison using real Binance 1H candles."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from signal_engine.backtest import CostConfig, backtest
from signal_engine.data.binance_source import fetch_range
from signal_engine.metrics import compute_metrics
from signal_engine.risk import RiskConfig
from signal_engine.strategies import IchimokuRsiConfirmedStrategy

MARKET, TIMEFRAME, LOOKBACK_DAYS = "BTCUSDT", "1H", 60
INITIAL_EQUITY = 10_000.0

class FilteredIchimoku(IchimokuRsiConfirmedStrategy):
    def __init__(self, use_macd=False, use_adx=False):
        super().__init__()
        self.use_macd, self.use_adx = use_macd, use_adx
        self.name = "ichimoku_rsi" + ("_macd" if use_macd else "") + ("_adx" if use_adx else "")

    def raw_direction(self, df):
        signals = super().raw_direction(df).copy()
        close = df["close"]
        if self.use_macd:
            macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
            signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
            hist = macd - signal
            signals = signals.where(((signals == 1) & (hist > 0)) | ((signals == -1) & (hist < 0)), 0)
        if self.use_adx:
            high, low = df["high"], df["low"]
            up = high.diff()
            down = -low.diff()
            plus_dm = up.where((up > down) & (up > 0), 0.0)
            minus_dm = down.where((down > up) & (down > 0), 0.0)
            prev_close = close.shift(1)
            tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
            dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
            adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            signals = signals.where(((signals == 1) & (adx >= 20) & (plus_di > minus_di)) | ((signals == -1) & (adx >= 20) & (minus_di > plus_di)), 0)
        return signals

def main():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    df = fetch_range(MARKET, TIMEFRAME, start, end)
    df = df.loc[df.index + pd.Timedelta(hours=1) <= end]
    if len(df) < 200:
        raise RuntimeError(f"insufficient closed candles: {len(df)}")
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0, reward_risk_multiple=2.0, risk_fraction=0.01)
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    variants = [
        ("RSI baseline", False, False),
        ("RSI + MACD", True, False),
        ("RSI + ADX", False, True),
        ("RSI + MACD + ADX", True, True),
    ]
    rows = []
    provenance = {"source": df.attrs.get("source", "unknown"), "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
        "coverage_start_utc": df.index[0].isoformat(), "coverage_end_utc": df.index[-1].isoformat(),
        "requested_start_utc": start.isoformat(), "requested_end_exclusive_utc": end.isoformat(), "closed_candles_only": True}
    for label, macd, adx in variants:
        strategy = FilteredIchimoku(macd, adx)
        result = backtest(df, strategy, MARKET, TIMEFRAME, risk, costs, initial_equity=INITIAL_EQUITY, provenance=provenance)
        m = compute_metrics(result)
        rows.append({"variant": label, "market": MARKET, "timeframe": TIMEFRAME, "candles": result.n_candles,
            "start_utc": result.start_utc, "end_utc": result.end_utc, "total_return_pct": round(m.total_return_pct, 6),
            "max_drawdown_pct": round(m.max_drawdown_pct, 6), "net_total_pnl": round(m.net_total_pnl, 6),
            "profit_factor": None if m.profit_factor is None else round(m.profit_factor, 6),
            "win_rate_pct": None if m.win_rate is None else round(m.win_rate*100, 6), "trades": m.n_trades,
            "total_fees": round(m.total_fees, 6), "insufficient_samples": m.insufficient_samples})
    output = {"kind":"btc_ichimoku_confirmation_comparison", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "window_days_requested":LOOKBACK_DAYS, "timeframe":TIMEFRAME, "market":MARKET,
        "base_conditions":["Ichimoku cloud breakout/direction","next completed candle closes beyond breakout candle high/low","RSI(14) direction >50 long / <50 short"],
        "additional_filters":{"MACD":"12/26/9 histogram agrees with direction","ADX":"ADX(14) >= 20 and +DI/-DI agrees with direction"},
        "initial_equity_per_test":INITIAL_EQUITY,"risk":vars(risk),"costs_bps_per_fill":vars(costs),
        "provenance":provenance,"results":rows,
        "limitations":["Same 60-day window as prior comparison; results are in-sample and not proof of future profitability.",
        "RSI is part of the existing implementation; this run compares adding MACD and/or ADX to it.",
        "Low trade counts are inconclusive. Fees, spread, slippage, and conservative stop-first fill logic are included."]}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/btc_ichimoku_confirmation_comparison.json").write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(rows).to_csv("artifacts/btc_ichimoku_confirmation_comparison.csv",index=False)
    print(json.dumps({"ok":True,"market":MARKET,"window_days":LOOKBACK_DAYS,"candles":len(df),"results":rows,
        "output_json":"artifacts/btc_ichimoku_confirmation_comparison.json","output_csv":"artifacts/btc_ichimoku_confirmation_comparison.csv"},indent=2))
if __name__ == "__main__":
    main()
