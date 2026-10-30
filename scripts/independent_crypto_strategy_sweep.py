"""Independent two-month crypto strategy sweep.

Long-only spot simulation. Signals use completed bars; entries/exits execute at
the next bar open. Taker fee, spread and slippage are charged on both sides.
No leverage, overlapping positions, fabricated candles, or parameter fitting.
"""
from __future__ import annotations
import json, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
INTERVALS = {"15m": 15, "1h": 60}
FEE_BPS, SPREAD_BPS, SLIPPAGE_BPS = 10.0, 2.0, 2.0  # per side: fee + half spread + slippage
INITIAL_EQUITY = 10_000.0
START = pd.Timestamp.now(tz="UTC").floor("h") - pd.DateOffset(months=2)
END = pd.Timestamp.now(tz="UTC").floor("h")

def fetch(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    base = "https://data-api.binance.vision/api/v3/klines"
    step_ms = INTERVALS[interval] * 60_000
    cursor, end_ms, rows = int(start.timestamp()*1000), int(end.timestamp()*1000), []
    while cursor < end_ms:
        q = urllib.parse.urlencode({"symbol":symbol,"interval":interval,"startTime":cursor,"endTime":end_ms,"limit":1000})
        req = urllib.request.Request(base+"?"+q,headers={"User-Agent":"strategy-research/1.0"})
        with urllib.request.urlopen(req,timeout=30) as response:
            page = json.loads(response.read().decode())
        if not isinstance(page,list): raise RuntimeError(f"Unexpected Binance response: {page}")
        if not page: break
        rows.extend(page)
        last_open=int(page[-1][0])
        nxt=last_open+step_ms
        if nxt<=cursor: raise RuntimeError("Pagination did not advance")
        cursor=nxt
        if len(page)<1000: break
        time.sleep(.15)
    if not rows: raise RuntimeError(f"No data: {symbol} {interval}")
    df=pd.DataFrame([{"open_time":int(x[0]),"open":float(x[1]),"high":float(x[2]),"low":float(x[3]),"close":float(x[4]),"volume":float(x[5]),"close_time":int(x[6])} for x in rows])
    df=df.drop_duplicates("open_time").sort_values("open_time")
    now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
    df=df[(df.open_time>=int(start.timestamp()*1000))&(df.open_time<end_ms)&(df.close_time<now_ms)]
    idx=pd.to_datetime(df.open_time,unit="ms",utc=True)
    df=df.set_index(idx)[["open","high","low","close","volume","close_time"]]
    expected=pd.date_range(df.index[0],df.index[-1],freq=f"{INTERVALS[interval]}min",tz="UTC")
    if not df.index.equals(expected):
        missing=len(expected.difference(df.index))
        raise RuntimeError(f"Missing candle intervals: {symbol} {interval} missing={missing}")
    df.attrs.update(source="Binance public Vision API",retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    return df

def indicators(d):
    c,h,l=d.close,d.high,d.low
    delta=c.diff()
    gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    rs=gain.ewm(alpha=.5,adjust=False,min_periods=2).mean()/loss.ewm(alpha=.5,adjust=False,min_periods=2).mean().replace(0,np.nan)
    rsi2=100-100/(1+rs)
    up=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    dn=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi14=100-100/(1+up/dn.replace(0,np.nan))
    ema20=c.ewm(span=20,adjust=False,min_periods=20).mean()
    ema50=c.ewm(span=50,adjust=False,min_periods=50).mean()
    ema200=c.ewm(span=200,adjust=False,min_periods=200).mean()
    sma5=c.rolling(5,min_periods=5).mean()
    mid=c.rolling(20,min_periods=20).mean()
    std=c.rolling(20,min_periods=20).std(ddof=0)
    upper,lower=mid+2*std,mid-2*std
    donchian_high=h.shift(1).rolling(20,min_periods=20).max()
    donchian_low=l.shift(1).rolling(20,min_periods=20).min()
    macd=c.ewm(span=12,adjust=False,min_periods=26).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
    sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    return pd.DataFrame({"rsi2":rsi2,"rsi14":rsi14,"ema20":ema20,"ema50":ema50,"ema200":ema200,
                         "sma5":sma5,"mid":mid,"upper":upper,"lower":lower,"donchian_high":donchian_high,
                         "macd_hist":macd-sig},index=d.index)

def make_rules(d):
    x=indicators(d); c=d.close; prev=c.shift(1)
    rules={}
    # Fixed, literature-standard forms; no tuning on the test window.
    rules["donchian20_breakout"]=(c>x.donchian_high, c<x.ema20)
    rules["ema20_50_pullback"]=( (x.ema20>x.ema50)&(prev<=x.ema20.shift(1))&(c>x.ema20), c<x.ema20)
    rules["bollinger20_2_mean_reversion"]=( (prev>=x.lower.shift(1))&(c<x.lower), c>=x.mid)
    rules["rsi2_10_mean_reversion"]=( (x.rsi2.shift(1)>=10)&(x.rsi2<10), (x.rsi2>60)|(c>x.sma5))
    rules["macd_ema200_trend"]=( (x.macd_hist.shift(1)<=0)&(x.macd_hist>0)&(c>x.ema200), x.macd_hist<0)
    rules["rsi14_30_reversal"]=( (x.rsi14.shift(1)>=30)&(x.rsi14<30), (x.rsi14>55))
    return rules

def run_one(d, name, entry, exit_sig, max_hold):
    # one position at a time; entry and signal are only known after candle close.
    o=d.open.to_numpy(float); c=d.close.to_numpy(float)
    en=entry.fillna(False).to_numpy(bool); ex=exit_sig.fillna(False).to_numpy(bool)
    equity=INITIAL_EQUITY; peak=equity; maxdd=0.0; pos=None; trades=[]; gross_turnover=0.0
    fee_rate=FEE_BPS/10000; adverse=(SPREAD_BPS/2+SLIPPAGE_BPS)/10000
    for i in range(len(d)-1):
        if pos is not None:
            side,qty,fill,entry_i=pos
            held=i-entry_i+1
            if ex[i] or held>=max_hold:
                exit_px=o[i+1]*(1-adverse)
                fee=abs(exit_px*qty)*fee_rate
                pnl=qty*(exit_px-fill)-fee
                equity+=qty*(exit_px-fill)-fee
                gross_turnover+=fill*qty+exit_px*qty
                trades.append({"entry_time":d.index[entry_i].isoformat(),"exit_time":d.index[i+1].isoformat(),
                               "entry":fill,"exit":exit_px,"qty":qty,"net_pnl":pnl,
                               "return_pct":pnl/(fill*qty)*100,"bars_held":held,
                               "exit_reason":"signal" if ex[i] else "max_hold"})
                pos=None
        if pos is None and en[i] and equity>0:
            fill=o[i+1]*(1+adverse)
            qty=equity/fill
            fee=fill*qty*fee_rate
            equity-=fee
            pos=(1,qty,fill,i+1)
        mtm=equity if pos is None else equity+pos[1]*(c[i]-pos[2])
        peak=max(peak,mtm); maxdd=min(maxdd,(mtm/peak-1) if peak else 0)
    if pos is not None:
        side,qty,fill,entry_i=pos
        exit_px=c[-1]*(1-adverse); fee=exit_px*qty*fee_rate
        pnl=qty*(exit_px-fill)-fee; equity+=qty*(exit_px-fill)-fee
        gross_turnover+=fill*qty+exit_px*qty
        trades.append({"entry_time":d.index[entry_i].isoformat(),"exit_time":d.index[-1].isoformat(),
                       "entry":fill,"exit":exit_px,"qty":qty,"net_pnl":pnl,
                       "return_pct":pnl/(fill*qty)*100,"bars_held":len(d)-entry_i,
                       "exit_reason":"end_of_data"})
    wins=[t["net_pnl"] for t in trades if t["net_pnl"]>0]
    losses=[t["net_pnl"] for t in trades if t["net_pnl"]<0]
    rets=np.array([t["net_pnl"]/(INITIAL_EQUITY if i==0 else max(1,INITIAL_EQUITY)) for i,t in enumerate(trades)])
    return {"strategy":name,"max_hold_bars":max_hold,"trades":len(trades),"wins":len(wins),"losses":len(losses),
            "win_rate_pct":round(100*len(wins)/len(trades),3) if trades else None,
            "net_return_pct":round((equity/INITIAL_EQUITY-1)*100,4),
            "profit_factor":round(sum(wins)/abs(sum(losses)),4) if losses else (None if not wins else "inf"),
            "max_drawdown_pct":round(maxdd*100,4),
            "avg_trade_net_pct":round(float(np.mean([t["return_pct"] for t in trades])),4) if trades else None,
            "median_trade_net_pct":round(float(np.median([t["return_pct"] for t in trades])),4) if trades else None,
            "gross_turnover_usdt":round(gross_turnover,2),"trades_detail":trades}

def main():
    out=Path("artifacts/crypto_strategy_sweep"); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; failures=[]; metadata=[]
    for symbol in SYMBOLS:
      for tf in INTERVALS:
        try:
          d=fetch(symbol,tf,START,END)
          metadata.append({"symbol":symbol,"timeframe":tf,"candles":len(d),"start_utc":d.index[0].isoformat(),
                           "end_utc":d.index[-1].isoformat(),"source":d.attrs["source"],
                           "retrieved_at_utc":d.attrs["retrieved_at_utc"],
                           "missing_intervals":0})
          for name,(entry,exit_sig) in make_rules(d).items():
            # Fixed holding caps are predeclared, not optimized: 8/24 bars.
            for hold in (8,24):
              row=run_one(d,name,entry,exit_sig,hold)
              row.update(symbol=symbol,timeframe=tf)
              all_rows.append(row)
        except Exception as e:
          failures.append({"symbol":symbol,"timeframe":tf,"error":repr(e)})
    summary=[]
    for row in all_rows:
      detail=row.pop("trades_detail")
      summary.append(row)
      safe=symbol=row["symbol"]; tf=row["timeframe"]; name=row["strategy"]; hold=row["max_hold_bars"]
      pd.DataFrame(detail).to_csv(out/f"trades_{safe}_{tf}_{name}_{hold}.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    payload={"created_at_utc":datetime.now(timezone.utc).isoformat(),"requested_start_utc":START.isoformat(),
             "requested_end_utc":END.isoformat(),"symbols":SYMBOLS,"timeframes":list(INTERVALS),
             "cost_model":{"taker_fee_bps_per_side":FEE_BPS,"spread_bps_roundtrip":SPREAD_BPS,
                           "slippage_bps_per_side":SLIPPAGE_BPS},
             "execution":"close signal -> next open entry; next-open exits; long-only spot; one position at a time; 100% equity sizing; no leverage",
             "strategies":[ "Donchian 20 breakout","EMA20/EMA50 pullback","Bollinger(20,2) lower-band mean reversion",
                            "RSI(2)<10 mean reversion","MACD(12,26,9) cross with EMA200 trend","RSI14<30 reversal"],
             "hold_caps_bars":[8,24],"data":metadata,"failures":failures,"results":summary,
             "limitations":["Two-month screening window is not proof of persistent edge.",
                            "Parameter values were predeclared; no optimization on this test window.",
                            "Long-only spot; results do not include funding or shorting.",
                            "100% notional exposure creates substantial drawdown risk.",
                            "A strategy with fewer than 100 trades is not eligible for a high-confidence claim.",
                            "Multiple comparisons require a separate chronological holdout and adjusted inference."]}
    (out/"report.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"window":[START.isoformat(),END.isoformat()],"datasets":metadata,"failures":failures,
                      "results":summary},indent=2,allow_nan=False))
    if failures: raise SystemExit(f"Data failures for {len(failures)} symbol/timeframe combinations")

if __name__=="__main__": main()
