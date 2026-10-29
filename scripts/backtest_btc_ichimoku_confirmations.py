"""60-day BTCUSDT 1H Ichimoku confirmation grid with the requested structural stop."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.backtest import CostConfig, backtest
from signal_engine.data.binance_source import fetch_range
from signal_engine.metrics import compute_metrics
from signal_engine.risk import RiskConfig
from signal_engine.strategies.base import Strategy

MARKET, TIMEFRAME = "BTCUSDT", "1H"
BAR = pd.Timedelta(hours=1)
LOOKBACK_DAYS = 60
INITIAL_EQUITY = 10_000.0
FILTERS = ("rsi", "macd", "ema_trend", "adx_direction", "volume_expansion")
COMBINATIONS = ((),) + tuple(c for n in (1, 2) for c in itertools.combinations(FILTERS, n))


def midpoint(df: pd.DataFrame, window: int) -> pd.Series:
    return (df["high"].rolling(window, min_periods=window).max()
            + df["low"].rolling(window, min_periods=window).min()) / 2.0


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.where(loss != 0)
    rsi = (100 - 100 / (1 + rs)).mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return pd.DataFrame({
        "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
        "ema20": ema20, "ema50": ema50, "adx": adx,
        "plus_di": plus_di, "minus_di": minus_di,
        "volume_ratio": df["volume"] / df["volume"].rolling(20, min_periods=20).mean(),
    }, index=df.index)


class IchimokuCombo(Strategy):
    def __init__(self, filters: tuple[str, ...]):
        self.filters = filters
        self.name = "ichimoku_cloud_confirm" + ("_base" if not filters else "_" + "_".join(filters))
        self._ind = None

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tenkan, kijun = midpoint(df, 9), midpoint(df, 26)
        # Cloud values are shifted back to where they are displayed; no future cloud.
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = midpoint(df, 52).shift(26)
        top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        ind = indicators(df)
        self._ind = ind
        out = pd.Series(0, index=df.index, dtype="int64")
        state, pending_side = "neutral", 0
        pending_high = pending_low = float("nan")

        def passes(side: int, i: int) -> bool:
            r = ind.iloc[i]
            for key in self.filters:
                if key == "rsi":
                    ok = r.rsi > 50 if side == 1 else r.rsi < 50
                elif key == "macd":
                    ok = r.macd > r.macd_signal if side == 1 else r.macd < r.macd_signal
                elif key == "ema_trend":
                    ok = (close.iloc[i] > r.ema50 and r.ema20 > r.ema50) if side == 1 else (close.iloc[i] < r.ema50 and r.ema20 < r.ema50)
                elif key == "adx_direction":
                    ok = (r.adx >= 20 and r.plus_di > r.minus_di) if side == 1 else (r.adx >= 20 and r.minus_di > r.plus_di)
                else:
                    ok = r.volume_ratio >= 1.2
                if pd.isna(ok) or not bool(ok):
                    return False
            return True

        for i in range(len(df)):
            if pd.isna(top.iloc[i]) or pd.isna(bottom.iloc[i]):
                continue
            c, h, l = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
            ct, cb = float(top.iloc[i]), float(bottom.iloc[i])
            # Confirmation candle closes beyond breakout candle high/low.
            if pending_side:
                side = pending_side
                candle_ok = c > pending_high if side == 1 else c < pending_low
                if candle_ok and passes(side, i):
                    out.iloc[i] = side
                pending_side = 0
            if c > ct and state == "inside_from_below":
                pending_side, pending_high, pending_low = 1, h, l
                state = "above"
            elif c < cb and state == "inside_from_above":
                pending_side, pending_high, pending_low = -1, h, l
                state = "below"
            elif c < cb:
                state = "below"
            elif c > ct:
                state = "above"
            else:
                if state == "below" and h >= cb:
                    state = "inside_from_below"
                elif state == "above" and l <= ct:
                    state = "inside_from_above"
                elif state not in ("inside_from_below", "inside_from_above"):
                    state = "neutral"
        return out

    def stop_for_signal(self, df: pd.DataFrame, signal_idx: int, side: int) -> float:
        # signal_idx = confirmation candle; signal_idx-1 = cloud breakout;
        # signal_idx-2 = candle immediately preceding the breakout candle.
        anchor_idx = signal_idx - 2
        if anchor_idx < 0:
            raise ValueError("insufficient candles for structural stop")
        return float(df["open"].iloc[anchor_idx])


def main() -> int:
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    df = fetch_range(MARKET, TIMEFRAME, start, end)
    df = df.loc[df.index + BAR <= end].copy()
    if len(df) < 100:
        raise RuntimeError(f"insufficient closed candles for 60-day run: {len(df)}")
    provenance = {
        "source": df.attrs.get("source", "unknown"),
        "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
        "coverage_start_utc": df.index[0].isoformat(),
        "coverage_end_utc": df.index[-1].isoformat(),
        "requested_start_utc": start.isoformat(),
        "requested_end_exclusive_utc": end.isoformat(),
        "closed_candles_only": True,
    }
    # ATR is not used to place stops; it is only part of the generic engine's
    # existing warm-up gate. Structural stop geometry determines risk and size.
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0, reward_risk_multiple=2.0,
                      risk_fraction=0.01, max_position_fraction=1.0)
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    rows = []
    for combo in COMBINATIONS:
        strategy = IchimokuCombo(combo)
        result = backtest(df, strategy, MARKET, TIMEFRAME, risk, costs,
                          initial_equity=INITIAL_EQUITY, provenance=provenance)
        m = compute_metrics(result)
        rows.append({
            "filters": list(combo), "strategy": strategy.name,
            "total_return_pct": round(m.total_return_pct, 6) if m.total_return_pct is not None else None,
            "max_drawdown_pct": round(m.max_drawdown_pct, 6) if m.max_drawdown_pct is not None else None,
            "net_total_pnl": round(m.net_total_pnl, 6) if m.net_total_pnl is not None else None,
            "profit_factor": round(m.profit_factor, 6) if m.profit_factor is not None else None,
            "win_rate_pct": round(m.win_rate * 100, 6) if m.win_rate is not None else None,
            "trades": m.n_trades, "fees": round(m.total_fees, 6) if m.total_fees is not None else None,
            "rejected_incoherent": result.n_signals_rejected_incoherent,
            "insufficient_samples": m.insufficient_samples,
        })
    output = {
        "kind": "btc_ichimoku_structural_stop_confirmation_grid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": MARKET, "timeframe": TIMEFRAME, "window_days_requested": LOOKBACK_DAYS,
        "initial_equity": INITIAL_EQUITY, "risk": vars(risk), "costs_bps_per_fill": vars(costs),
        "fixed_conditions": [
            "Ichimoku 9/26/52 cloud structure and direction; enter cloud from opposite side then close beyond cloud",
            "following completed candle closes beyond breakout candle high (long) / low (short)",
            "execute at next candle open",
            "structural stop = open of candle immediately preceding the cloud-breakout candle",
            "invalid stop geometry rejects trade; no ATR or other fallback stop",
            "target = 2R from entry using structural-stop distance; size risks 1% equity capped at 100% notional",
        ],
        "filters": {
            "rsi": "RSI(14) >50 long / <50 short",
            "macd": "MACD(12,26) above/below signal(9)",
            "ema_trend": "price and EMA20 above EMA50 long; both below short",
            "adx_direction": "ADX(14)>=20 with +DI/-DI direction",
            "volume_expansion": "confirmation volume >=1.2x trailing 20-bar mean",
        },
        "data_provenance": provenance, "results": rows,
        "limitations": [
            "Exploratory in-sample grid; selecting the highest return among many combinations introduces selection bias.",
            "Two months and low trade counts cannot establish robust profitability.",
            "No live trading; this is a historical research simulation with modeled costs.",
        ],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/btc_ichimoku_confirmation_comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv("artifacts/btc_ichimoku_confirmation_comparison.csv", index=False)
    print(json.dumps({"ok": True, "market": MARKET, "timeframe": TIMEFRAME,
                      "candles": len(df), "provenance": provenance, "results": rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
