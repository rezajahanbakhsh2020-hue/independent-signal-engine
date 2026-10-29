"""BTC-only Ichimoku confirmation-combination experiment on one year of closed Binance 1H candles."""
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

MARKET = "BTCUSDT"
TIMEFRAME = "1H"
LOOKBACK_DAYS = 365
INITIAL_EQUITY = 10_000.0
CONFIRMATIONS = ("rsi", "macd", "ema_trend", "adx_direction")
COMBINATIONS = tuple(itertools.combinations(CONFIRMATIONS, 2))


def midpoint(df: pd.DataFrame, window: int) -> pd.Series:
    return (df["high"].rolling(window, min_periods=window).max()
            + df["low"].rolling(window, min_periods=window).min()) / 2.0


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.where(loss != 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return pd.DataFrame({
        "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
        "ema20": ema20, "ema50": ema50, "adx": adx,
        "plus_di": plus_di, "minus_di": minus_di,
    }, index=df.index)


class IchimokuComboStrategy(Strategy):
    """Fixed cloud/candle setup; choose two extra confirmations per experiment."""
    def __init__(self, confirmations: tuple[str, str]) -> None:
        self.confirmations = confirmations
        self.name = "ichimoku_" + "_".join(confirmations)

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tenkan = midpoint(df, 9)
        kijun = midpoint(df, 26)
        # Shift back to the candle where the cloud is visibly plotted; no lookahead.
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = midpoint(df, 52).shift(26)
        top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        ind = indicator_frame(df)

        out = pd.Series(0, index=df.index, dtype="int64")
        state = "neutral"
        pending_side = 0
        pending_high = pending_low = float("nan")

        def passes(side: int, i: int) -> bool:
            row = ind.iloc[i]
            for key in self.confirmations:
                if key == "rsi":
                    ok = row["rsi"] > 50 if side == 1 else row["rsi"] < 50
                elif key == "macd":
                    ok = row["macd"] > row["macd_signal"] if side == 1 else row["macd"] < row["macd_signal"]
                elif key == "ema_trend":
                    ok = (close.iloc[i] > row["ema50"] and row["ema20"] > row["ema50"]) if side == 1 else (close.iloc[i] < row["ema50"] and row["ema20"] < row["ema50"])
                else:  # ADX strength plus DI direction
                    ok = (row["adx"] >= 20 and row["plus_di"] > row["minus_di"]) if side == 1 else (row["adx"] >= 20 and row["minus_di"] > row["plus_di"])
                if not bool(ok) or pd.isna(ok):
                    return False
            return True

        for i in range(len(df)):
            if pd.isna(top.iloc[i]) or pd.isna(bottom.iloc[i]):
                continue
            c, h, l = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
            cloud_top, cloud_bottom = float(top.iloc[i]), float(bottom.iloc[i])
            if pending_side:
                side = pending_side
                candle_ok = c > pending_high if side == 1 else c < pending_low
                if candle_ok and passes(side, i):
                    out.iloc[i] = side
                pending_side = 0

            if c > cloud_top and state == "inside_from_below":
                pending_side, pending_high, pending_low = 1, h, l
                state = "above"
            elif c < cloud_bottom and state == "inside_from_above":
                pending_side, pending_high, pending_low = -1, h, l
                state = "below"
            elif c < cloud_bottom:
                state = "below"
            elif c > cloud_top:
                state = "above"
            else:
                if state == "below" and h >= cloud_bottom:
                    state = "inside_from_below"
                elif state == "above" and l <= cloud_top:
                    state = "inside_from_above"
                elif state not in ("inside_from_below", "inside_from_above"):
                    state = "neutral"
        return out


class WindowedStrategy(Strategy):
    """Apply a strategy only to a specified trading window, retaining prior indicator context."""
    def __init__(self, inner: Strategy, start: pd.Timestamp, end: pd.Timestamp) -> None:
        self.inner = inner
        self.start = start
        self.end = end
        self.name = f"{inner.name}_{start.date()}_{end.date()}"

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        raw = self.inner.raw_direction(df).copy()
        # A raw signal is executed one candle later; suppress signals outside the
        # window and the final candle before its exclusive end boundary.
        mask = (df.index >= self.start) & (df.index < self.end - pd.Timedelta(hours=1))
        return raw.where(mask, 0)


def summarize(df: pd.DataFrame, strategy: Strategy, market: str, risk: RiskConfig,
              costs: CostConfig, provenance: dict) -> dict:
    result = backtest(df, strategy, market, TIMEFRAME, risk, costs,
                      initial_equity=INITIAL_EQUITY, provenance=provenance)
    m = compute_metrics(result)
    return {
        "total_return_pct": m.total_return_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "net_total_pnl": m.net_total_pnl,
        "profit_factor": m.profit_factor,
        "win_rate_pct": None if m.win_rate is None else m.win_rate * 100,
        "trades": m.n_trades,
        "total_fees": m.total_fees,
        "insufficient_samples": m.insufficient_samples,
        "n_gains": m.n_gains,
        "n_losses": m.n_losses,
    }

def main() -> int:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    end = now
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    df = fetch_range(MARKET, TIMEFRAME, start, end)
    df = df.loc[df.index + pd.Timedelta(hours=1) <= end].copy()
    if len(df) < 5000:
        raise RuntimeError(f"Expected roughly one year of closed BTC candles; received only {len(df)}")
    provenance = {
        "source": df.attrs.get("source", "unknown"),
        "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
        "coverage_start_utc": df.index[0].isoformat(),
        "coverage_end_utc": df.index[-1].isoformat(),
        "requested_start_utc": start.isoformat(),
        "requested_end_exclusive_utc": end.isoformat(),
        "closed_candles_only": True,
    }
    risk = RiskConfig(atr_window=14, atr_multiplier=2.0,
                      reward_risk_multiple=2.0, risk_fraction=0.01)
    costs = CostConfig(fee_bps=2.0, spread_bps=2.0, slippage_bps=1.0)
    split = end - pd.Timedelta(days=120)
    rows = []
    for combo in COMBINATIONS:
        strategy = IchimokuComboStrategy(combo)
        full = summarize(df, strategy, MARKET, risk, costs, provenance)
        train = summarize(df, WindowedStrategy(strategy, start, split), MARKET, risk, costs, provenance)
        test = summarize(df, WindowedStrategy(strategy, split, end), MARKET, risk, costs, provenance)
        rows.append({
            "confirmations": list(combo), "strategy": strategy.name,
            "candles": len(df), "start_utc": provenance["coverage_start_utc"],
            "end_utc": provenance["coverage_end_utc"],
            "full_year": full, "first_245_days_selection_window": train,
            "final_120_days_out_of_sample": test,
        })
    rows.sort(key=lambda x: (
        x["first_245_days_selection_window"]["total_return_pct"] is not None,
        x["first_245_days_selection_window"]["total_return_pct"]
        if x["first_245_days_selection_window"]["total_return_pct"] is not None else -1e9
    ), reverse=True)
    selected = rows[0]
    output = {
        "kind": "btc_ichimoku_confirmation_grid_backtest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": MARKET, "timeframe": TIMEFRAME, "window_days_requested": LOOKBACK_DAYS,
        "selection_window_days": 245, "untouched_out_of_sample_days": 120,
        "selection_rule": "rank combinations on first 245 days only; inspect final 120 days separately",
        "initial_equity": INITIAL_EQUITY, "risk": vars(risk), "costs_bps_per_fill": vars(costs),
        "fixed_base_rules": {
            "ichimoku": "9/26/52; use cloud visibly plotted at current candle without future leakage",
            "cloud_direction": "long only after entering cloud from below and closing above; short only after entering from above and closing below",
            "candle_confirmation": "next completed candle closes beyond breakout candle high/low; execute at following candle open",
        },
        "extra_confirmation_candidates": {
            "rsi": "RSI(14) above 50 for long, below 50 for short",
            "macd": "MACD(12,26) above/below signal(9)",
            "ema_trend": "price and EMA20 above EMA50 for long; both below for short",
            "adx_direction": "ADX(14) >= 20 and DI direction agrees with side",
        },
        "data_provenance": provenance, "results": rows,
        "best_by_selection_window": {"confirmations": selected["confirmations"], "strategy": selected["strategy"], "first_245_days_selection_window": selected["first_245_days_selection_window"], "final_120_days_out_of_sample": selected["final_120_days_out_of_sample"]},
        "limitations": [
            "Exploratory grid search; choosing the best of six combinations creates selection bias.",
            "One year of historical BTC 1H data is not evidence of future profitability.",
            "All results include the same modeled fees, spread, slippage, ATR stop/target and next-open execution.",
            "A candidate is not considered robust from positive total return alone; trade count, drawdown and an untouched out-of-sample period must also be reviewed.",
        ],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/btc_ichimoku_confirmation_grid.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv("artifacts/btc_ichimoku_confirmation_grid.csv", index=False)
    print(json.dumps({"ok": True, "market": MARKET, "candles": len(df),
                      "coverage_start_utc": provenance["coverage_start_utc"],
                      "coverage_end_utc": provenance["coverage_end_utc"],
                      "combinations_tested": len(rows), "results": rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
