"""BTCUSDT 60-day Ichimoku confirmation grid using the requested structural stop."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.data.binance_source import fetch_range

MARKET, TIMEFRAME = "BTCUSDT", "1H"
DAYS, INITIAL_EQUITY, RISK_FRACTION, RR = 60, 10_000.0, 0.01, 2.0
FEE_BPS, SPREAD_BPS, SLIPPAGE_BPS = 2.0, 2.0, 1.0
FILTERS = ("rsi", "macd", "adx", "ema_trend", "stochastic")
COMBOS = [()] + list(itertools.combinations(FILTERS, 1)) + list(itertools.combinations(FILTERS, 2))


def midpoint(df, n):
    return (df.high.rolling(n, min_periods=n).max() + df.low.rolling(n, min_periods=n).min()) / 2


def indicators(df):
    c, h, l = df.close, df.high, df.low
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss != 0))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    e12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    e26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = e12 - e26
    macds = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up, down = h.diff(), -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    low14, high14 = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    stoch = 100 * (c-low14) / (high14-low14).replace(0, np.nan)
    return pd.DataFrame({"rsi":rsi, "macd_hist":macd-macds, "pdi":pdi, "mdi":mdi,
                         "adx":adx, "ema20":ema20, "ema50":ema50, "stoch":stoch}, index=df.index)


def raw_signals(df, filters):
    hi, lo, close = df.high, df.low, df.close
    tenkan, kijun = midpoint(df, 9), midpoint(df, 26)
    top = pd.concat([((tenkan+kijun)/2).shift(26), midpoint(df,52).shift(26)], axis=1).max(axis=1)
    bottom = pd.concat([((tenkan+kijun)/2).shift(26), midpoint(df,52).shift(26)], axis=1).min(axis=1)
    ind = indicators(df)
    out = np.zeros(len(df), dtype=np.int64)
    state, pending_side, pending_high, pending_low = "neutral", 0, np.nan, np.nan
    for i in range(len(df)):
        if not np.isfinite(top.iloc[i]) or not np.isfinite(bottom.iloc[i]):
            continue
        c, h, l, ct, cb = float(close.iloc[i]), float(hi.iloc[i]), float(lo.iloc[i]), float(top.iloc[i]), float(bottom.iloc[i])
        if pending_side:
            side = pending_side
            confirm = (c > pending_high) if side == 1 else (c < pending_low)
            ok = confirm
            for f in filters:
                row = ind.iloc[i]
                if f == "rsi": ok &= row.rsi > 50 if side == 1 else row.rsi < 50
                elif f == "macd": ok &= row.macd_hist > 0 if side == 1 else row.macd_hist < 0
                elif f == "adx": ok &= (row.adx >= 20 and row.pdi > row.mdi) if side == 1 else (row.adx >= 20 and row.mdi > row.pdi)
                elif f == "ema_trend": ok &= (row.ema20 > row.ema50) if side == 1 else (row.ema20 < row.ema50)
                elif f == "stochastic": ok &= row.stoch > 50 if side == 1 else row.stoch < 50
            if bool(ok): out[i] = side
            pending_side = 0
        if c > ct and state == "inside_from_below":
            pending_side, pending_high, pending_low, state = 1, h, l, "above"
        elif c < cb and state == "inside_from_above":
            pending_side, pending_high, pending_low, state = -1, h, l, "below"
        elif c < cb: state = "below"
        elif c > ct: state = "above"
        elif state == "below" and h >= cb: state = "inside_from_below"
        elif state == "above" and l <= ct: state = "inside_from_above"
        elif state not in ("inside_from_below", "inside_from_above"): state = "neutral"
    return pd.Series(out, index=df.index, dtype="int64")


def run_backtest(df, filters):
    raw = raw_signals(df, filters).to_numpy()
    o, h, l, c = (df[k].to_numpy(float) for k in ("open","high","low","close"))
    idx = df.index
    equity, peak, max_dd = INITIAL_EQUITY, INITIAL_EQUITY, 0.0
    trades, wins, losses, fees_paid, rejected = [], [], [], 0.0, 0
    position = None
    eq_curve = []
    def fee(notional): return abs(notional) * FEE_BPS / 10000
    def enter_fill(side, price): return price * (1 + side*(SPREAD_BPS/2+SLIPPAGE_BPS)/10000)
    def stop_fill(side, price): return price * (1 - side*(SPREAD_BPS/2+SLIPPAGE_BPS)/10000)
    def target_fill(side, price): return price * (1 - side*(SPREAD_BPS/2)/10000)
    for t in range(len(df)):
        if position is not None:
            s, qty, entry, stop, target, entry_fee, entry_ts, signal_ts = position
            stop_hit = l[t] <= stop if s == 1 else h[t] >= stop
            target_hit = h[t] >= target if s == 1 else l[t] <= target
            if stop_hit or target_hit:
                reason = "stop" if stop_hit else "target"
                fill = stop_fill(s, stop) if stop_hit else target_fill(s, target)
                exit_fee = fee(fill*qty)
                pnl = qty*s*(fill-entry) - entry_fee - exit_fee
                equity += qty*s*(fill-entry) - exit_fee
                fees_paid += exit_fee
                trades.append({"pnl":pnl,"reason":reason,"entry":entry,"stop":stop,"target":target,"entry_ts":str(entry_ts),"exit_ts":str(idx[t])})
                (wins if pnl > 0 else losses if pnl < 0 else []).append(pnl)
                position = None
        # Signal is known at t-1; entry at this candle's open. Structural anchor
        # is the open of the candle before the breakout candle: signal index - 2.
        side = int(raw[t-1]) if t > 0 else 0
        if position is None and side:
            signal_i = t-1
            anchor_i = signal_i-2
            if anchor_i < 0: rejected += 1
            else:
                entry_ref = float(o[t])
                stop = float(o[anchor_i])
                distance = (entry_ref-stop)*side
                if not np.isfinite(distance) or distance <= 0:
                    rejected += 1
                else:
                    target = entry_ref + side*RR*distance
                    entry = enter_fill(side, entry_ref)
                    # Keep sizing based on structural risk distance; cap notional at equity.
                    qty = min((equity*RISK_FRACTION)/distance, equity/entry)
                    if not (np.isfinite(qty) and qty > 0 and ((side == 1 and stop < entry < target) or (side == -1 and target < entry < stop))):
                        rejected += 1
                    else:
                        entry_fee = fee(entry*qty)
                        equity -= entry_fee
                        fees_paid += entry_fee
                        position = (side,qty,entry,stop,target,entry_fee,idx[t],idx[signal_i])
        eq_curve.append(equity if position is None else equity + position[1]*position[0]*(c[t]-position[2]))
        peak = max(peak, eq_curve[-1])
        max_dd = min(max_dd, (eq_curve[-1]-peak)/peak if peak else 0)
    if position is not None:
        s,qty,entry,stop,target,entry_fee,entry_ts,signal_ts = position
        fill = target_fill(s, float(c[-1]))
        exit_fee = fee(fill*qty)
        pnl = qty*s*(fill-entry)-entry_fee-exit_fee
        equity += qty*s*(fill-entry)-exit_fee
        fees_paid += exit_fee
        trades.append({"pnl":pnl,"reason":"end_of_data","entry":entry,"stop":stop,"target":target,"entry_ts":str(entry_ts),"exit_ts":str(idx[-1])})
        (wins if pnl > 0 else losses if pnl < 0 else []).append(pnl)
        eq_curve[-1] = equity
    gross_wins = sum(x for x in wins if x>0)
    gross_losses = abs(sum(x for x in losses if x<0))
    return {"filters":"+".join(filters) if filters else "cloud+next-candle only",
            "net_return_pct":round((equity/INITIAL_EQUITY-1)*100,4),
            "max_drawdown_pct":round(max_dd*100,4),"profit_factor":round(gross_wins/gross_losses,4) if gross_losses else None,
            "trades":len(trades),"wins":len(wins),"losses":len(losses),"net_pnl":round(equity-INITIAL_EQUITY,2),
            "fees_paid":round(fees_paid,2),"rejected_invalid_structural_stop":rejected,
            "avg_trade_pnl":round(float(np.mean([x["pnl"] for x in trades])),2) if trades else None,
            "trades_detail":trades}


def main():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=DAYS)
    df = fetch_range(MARKET, TIMEFRAME, start, end)
    # Explicitly remove any candle that is not fully closed.
    df = df.loc[df.index + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")]
    if len(df) < 200: raise RuntimeError(f"Insufficient closed candles: {len(df)}")
    results = [run_backtest(df, combo) for combo in COMBOS]
    results.sort(key=lambda x: (x["net_return_pct"], x["profit_factor"] if x["profit_factor"] is not None else -1), reverse=True)
    meta = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"market":MARKET,"timeframe":TIMEFRAME,
            "requested_window_days":DAYS,"actual_candles":len(df),"coverage_start_utc":df.index[0].isoformat(),
            "coverage_end_utc":df.index[-1].isoformat(),"data_source":df.attrs.get("source"),
            "retrieval_time_utc":df.attrs.get("retrieval_time_utc"),"initial_equity":INITIAL_EQUITY,
            "risk_fraction":RISK_FRACTION,"reward_risk":RR,"costs_bps_per_fill":{"fee":FEE_BPS,"spread":SPREAD_BPS,"slippage":SLIPPAGE_BPS},
            "structural_stop_rule":"open of candle immediately before breakout candle; breakout candle is signal candle minus one; anchor index = confirmation signal index minus two",
            "base_conditions":["Ichimoku cloud exit after entering from opposite side","next candle closes beyond breakout candle high/low"],
            "filter_combinations_tested":len(COMBOS),"results":results,
            "caveats":["One recent 60-day in-sample period; no guarantee of future performance.","Trade counts may be too small for inference.",
                       "Stops use the exact requested anchor and invalid stop geometry rejects the trade without fallback."]}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/btc_ichimoku_structural_stop.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in r.items() if k!="trades_detail"} for r in results]).to_csv("artifacts/btc_ichimoku_structural_stop.csv",index=False)
    print(json.dumps({"ok":True,**{k:meta[k] for k in ("market","timeframe","requested_window_days","actual_candles","coverage_start_utc","coverage_end_utc","data_source","filter_combinations_tested","structural_stop_rule")},"results":[{k:v for k,v in r.items() if k!="trades_detail"} for r in results]},indent=2))

if __name__ == "__main__":
    main()
