"""Local live signal dashboard with a persistent, deduplicated event feed."""
from __future__ import annotations
import argparse, html, json, sqlite3, webbrowser
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import pandas as pd
from signal_engine.chart import build_chart_html
from signal_engine.constants import TIMEFRAMES, validate_market_or_crypto_symbol, validate_timeframe
from signal_engine.latest_signal import latest_decision_report
from signal_engine.backtest import CostConfig
from signal_engine.indicators import atr as atr_fn
from signal_engine.risk import RiskConfig, build_signal
from signal_engine.strategies import STRATEGY_REGISTRY
from signal_engine.strategies.base import transition_direction

TIMEFRAME_DURATION = {"1m":timedelta(minutes=1),"5m":timedelta(minutes=5),"15m":timedelta(minutes=15),"30m":timedelta(minutes=30),"1H":timedelta(hours=1),"4H":timedelta(hours=4),"1D":timedelta(days=1)}
POLL_SECONDS = 10
CRYPTO_LOOKBACK_CANDLES = 300
RISK_CONFIG = RiskConfig()
DISPLAY_COST_CONFIG = CostConfig()
RISK_TARGET_MULTIPLES = (1.0, 2.0, 3.0)
RISK_MODEL = (
    "Backtest-aligned RiskConfig: SMA ATR(14) x 2 stop; "
    "net 1R/2R/3R targets; fee 2 bps/fill, spread 2 bps, slippage 1 bp/fill"
)


def _risk_levels_for_event(df, signal_idx, side, atr_values, market, timeframe):
    """Return cost-aware display levels using the same risk/fill model as backtest."""
    signal = build_signal(
        side=side,
        signal_idx=signal_idx,
        df=df,
        risk_config=RISK_CONFIG,
        atr_series=atr_values,
        equity=10_000.0,
        market=market,
        timeframe=timeframe,
        strategy_name="dashboard-display",
    )
    if signal is None:
        return None

    costs = DISPLAY_COST_CONFIG
    fee_rate = costs.fee_bps / 10_000.0
    entry_fill = costs.enter_fill(side, signal.entry)
    if side > 0:
        entry_effective = entry_fill * (1.0 + fee_rate)
        stop_fill = costs.exit_stop_fill(side, signal.stop)
        stop_effective = stop_fill * (1.0 - fee_rate)
        risk_distance = entry_effective - stop_effective
        if risk_distance <= 0:
            return None
        targets = []
        for multiple in RISK_TARGET_MULTIPLES:
            desired_net_exit = entry_effective + risk_distance * multiple
            target_fill = desired_net_exit / (1.0 - fee_rate)
            trigger = target_fill / (1.0 - costs.spread_bps / 20_000.0)
            targets.append(trigger)
    elif side < 0:
        entry_effective = entry_fill * (1.0 - fee_rate)
        stop_fill = costs.exit_stop_fill(side, signal.stop)
        stop_effective = stop_fill * (1.0 + fee_rate)
        risk_distance = stop_effective - entry_effective
        if risk_distance <= 0:
            return None
        targets = []
        for multiple in RISK_TARGET_MULTIPLES:
            desired_net_exit = entry_effective - risk_distance * multiple
            if desired_net_exit <= 0:
                return None
            target_fill = desired_net_exit / (1.0 + fee_rate)
            trigger = target_fill / (1.0 + costs.spread_bps / 20_000.0)
            targets.append(trigger)
    else:
        return None

    values = [entry_effective, signal.stop, *targets, risk_distance]
    if not all(pd.notna(value) and float(value) > 0 for value in values):
        return None
    if side > 0 and not (signal.stop < signal.entry < targets[0] < targets[1] < targets[2]):
        return None
    if side < 0 and not (targets[2] < targets[1] < targets[0] < signal.entry < signal.stop):
        return None

    return {
        "entry_price": float(entry_effective),
        "stop_loss": float(signal.stop),
        "tp1": float(targets[0]),
        "tp2": float(targets[1]),
        "tp3": float(targets[2]),
        "risk_distance": float(risk_distance),
        "risk_atr": float(atr_values.iloc[signal_idx]),
        "risk_model": RISK_MODEL,
    }


def _database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(DB_PATH,timeout=5)
    db.execute("""CREATE TABLE IF NOT EXISTS signal_events (
      event_key TEXT PRIMARY KEY, market TEXT NOT NULL, timeframe TEXT NOT NULL,
      strategy TEXT NOT NULL, side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
      signal_time_utc TEXT NOT NULL, candle_open_utc TEXT NOT NULL,
      close_price REAL NOT NULL, recorded_at_utc TEXT NOT NULL,
      entry_price REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL,
      risk_distance REAL, risk_atr REAL, risk_model TEXT)""")
    existing = {row[1] for row in db.execute("PRAGMA table_info(signal_events)")}
    migrations = (
      ("entry_price","ALTER TABLE signal_events ADD COLUMN entry_price REAL"),
      ("stop_loss","ALTER TABLE signal_events ADD COLUMN stop_loss REAL"),
      ("tp1","ALTER TABLE signal_events ADD COLUMN tp1 REAL"),
      ("tp2","ALTER TABLE signal_events ADD COLUMN tp2 REAL"),
      ("tp3","ALTER TABLE signal_events ADD COLUMN tp3 REAL"),
      ("risk_distance","ALTER TABLE signal_events ADD COLUMN risk_distance REAL"),
      ("risk_atr","ALTER TABLE signal_events ADD COLUMN risk_atr REAL"),
      ("risk_model","ALTER TABLE signal_events ADD COLUMN risk_model TEXT"),
    )
    for name, statement in migrations:
        if name not in existing:
            db.execute(statement)
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_events_scope_time ON signal_events(market,timeframe,strategy,signal_time_utc DESC)")
    return db

def _market_data(market,timeframe,start,end):
    if market=="XAUUSD":
        from signal_engine.data.biquote_source import fetch_range
    else:
        from signal_engine.data.crypto_source import fetch_range
    return fetch_range(market,timeframe,start,end)

def _record_events(df,market,timeframe):
    duration=pd.Timedelta(TIMEFRAME_DURATION[timeframe]); now=pd.Timestamp.now(tz="UTC").isoformat()
    records=[]
    atr_values=atr_fn(df["high"],df["low"],df["close"],RISK_CONFIG.atr_window)
    for strategy_name,strategy_type in STRATEGY_REGISTRY.items():
        strategy=strategy_type()
        transitions=transition_direction(strategy.raw_direction(df))
        for stamp,side in transitions.items():
            side=int(side)
            if side not in {-1,1}: continue
            signal_idx=int(df.index.get_indexer([stamp])[0])
            if signal_idx < 0: continue
            candle_open=pd.Timestamp(stamp); signal_time=candle_open+duration
            key="|".join((market,timeframe,strategy_name,str(side),signal_time.isoformat()))
            levels=_risk_levels_for_event(df,signal_idx,side,atr_values,market,timeframe) or {}
            records.append((key,market,timeframe,strategy_name,"BUY" if side>0 else "SELL",
                signal_time.isoformat(),candle_open.isoformat(),float(df.loc[stamp,"close"]),now,
                levels.get("entry_price"),levels.get("stop_loss"),levels.get("tp1"),levels.get("tp2"),
                levels.get("tp3"),levels.get("risk_distance"),levels.get("risk_atr"),levels.get("risk_model")))
    if records:
        with closing(_database()) as db:
            db.executemany("""INSERT INTO signal_events
            (event_key,market,timeframe,strategy,side,signal_time_utc,candle_open_utc,close_price,
             recorded_at_utc,entry_price,stop_loss,tp1,tp2,tp3,risk_distance,risk_atr,risk_model)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_key) DO UPDATE SET
              entry_price=CASE WHEN excluded.entry_price IS NOT NULL THEN excluded.entry_price ELSE signal_events.entry_price END,
              stop_loss=CASE WHEN excluded.stop_loss IS NOT NULL THEN excluded.stop_loss ELSE signal_events.stop_loss END,
              tp1=CASE WHEN excluded.tp1 IS NOT NULL THEN excluded.tp1 ELSE signal_events.tp1 END,
              tp2=CASE WHEN excluded.tp2 IS NOT NULL THEN excluded.tp2 ELSE signal_events.tp2 END,
              tp3=CASE WHEN excluded.tp3 IS NOT NULL THEN excluded.tp3 ELSE signal_events.tp3 END,
              risk_distance=CASE WHEN excluded.risk_distance IS NOT NULL THEN excluded.risk_distance ELSE signal_events.risk_distance END,
              risk_atr=CASE WHEN excluded.risk_atr IS NOT NULL THEN excluded.risk_atr ELSE signal_events.risk_atr END,
              risk_model=CASE WHEN excluded.risk_model IS NOT NULL THEN excluded.risk_model ELSE signal_events.risk_model END""",records)
            db.commit()

def _history(market,timeframe):
    with closing(_database()) as db:
        rows=db.execute("""SELECT side,signal_time_utc,close_price,recorded_at_utc,strategy,market,timeframe,
        entry_price,stop_loss,tp1,tp2,tp3,risk_distance,risk_atr,risk_model
        FROM signal_events WHERE market=? AND timeframe=?
        ORDER BY signal_time_utc DESC, recorded_at_utc DESC""",(market,timeframe)).fetchall()
    now=pd.Timestamp.now(tz="UTC"); result=[]
    for row in rows:
        (side,signal_time,price,recorded_at,strategy_name,event_market,event_timeframe,
         entry,stop,tp1,tp2,tp3,risk_distance,risk_atr,risk_model)=row
        stamp=pd.Timestamp(signal_time); age=max(0,int((now-stamp).total_seconds()))
        age_text=f"{age//60} دقیقه قبل" if age<3600 else f"{age//3600} ساعت و {(age%3600)//60} دقیقه قبل"
        result.append({"side":side,"signal_time_utc":stamp.isoformat(),
            "signal_time_local":stamp.to_pydatetime().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "age_seconds":age,"age_text":age_text,"close_price":price,"recorded_at_utc":recorded_at,
            "strategy":strategy_name,"market":event_market,"timeframe":event_timeframe,
            "entry_price":entry,"stop_loss":stop,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "risk_distance":risk_distance,"risk_atr":risk_atr,"risk_model":risk_model})
    return result

def _poll(market,timeframe,strategy_name,days):
    now=pd.Timestamp.now(tz="UTC")
    duration=pd.Timedelta(TIMEFRAME_DURATION[timeframe])
    if market=="XAUUSD":
        start=now-pd.Timedelta(days=days)-duration*50
    else:
        # Crypto uses a bounded 300-candle window, including indicator warm-up.
        start=now-duration*CRYPTO_LOOKBACK_CANDLES
    df=_market_data(market,timeframe,start,now)
    df=df.loc[(df.index+duration)<=now]
    if df.empty: raise ValueError("داده کندل بسته‌شده برای این نماد و تایم‌فریم دریافت نشد.")
    age_seconds=max(0.0,(now-(df.index[-1]+duration)).total_seconds())
    fresh=age_seconds<=duration.total_seconds()*3
    strategy=STRATEGY_REGISTRY[strategy_name]()
    decision=latest_decision_report(df,strategy,RiskConfig(),market,timeframe,provenance=str(df.attrs.get("source","public market data")))
    if not fresh: decision["decision"]={"label":"NO TRADE","reason":"stale_data"}
    _record_events(df,market,timeframe)
    detail=decision.get("decision",{}); label=detail.get("label","NO TRADE")
    transitions=transition_direction(strategy.raw_direction(df))
    positions=[i for i,value in enumerate(transitions.tolist()) if int(value) in {-1,1}]
    latest=None
    if positions:
        pos=positions[-1]; stamp=pd.Timestamp(df.index[pos])+duration
        latest={"side":"BUY" if int(transitions.iloc[pos])>0 else "SELL","signal_time_utc":stamp.isoformat(),"signal_time_local":stamp.to_pydatetime().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),"age_seconds":max(0,int((now-stamp).total_seconds()))}
    chart=build_chart_html(df,market=market,timeframe=timeframe,strategy_name=strategy_name,strategy_params={},strategy=strategy,latest_decision=decision)
    return {"market":market,"timeframe":timeframe,"strategy":strategy_name,"decision":detail,"decision_label":label,"fresh":fresh,"data_age_seconds":int(age_seconds),"source":df.attrs.get("source","public market data"),"provider":df.attrs.get("provider","BiQuote" if market=="XAUUSD" else "unknown"),"provider_attempts":df.attrs.get("provider_attempts",1),"provider_failures":list(df.attrs.get("provider_failures",())),"pagination_truncated":bool(df.attrs.get("pagination_truncated",False)),"fetch_elapsed_seconds":df.attrs.get("fetch_elapsed_seconds"),"retrieval_time_utc":df.attrs.get("retrieval_time_utc",""),"closed_candles":len(df),"latest_closed_candle_utc":df.index[-1].isoformat(),"latest_signal":latest,"history":_history(market,timeframe),"chart":chart,"polled_at_utc":now.isoformat()}

def _page():
    market_options=("BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","XAUUSD")
    markets="".join(f'<option value="{s}"{" selected" if s=="XAUUSD" else ""}>{s}</option>' for s in market_options)
    timeframes="".join(f'<option value="{tf}"{" selected" if tf=="1m" else ""}>{tf}</option>' for tf in TIMEFRAMES)
    strategies="".join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in STRATEGY_REGISTRY)
    return f"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>داشبورد آنلاین سیگنال</title>
<style>
:root{{color-scheme:dark;--bg:#0b1218;--line:#263946;--muted:#9aabb7;--text:#edf4f7;--green:#21c58e;--red:#ff6177;--gold:#f3c76a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(160deg,#101e28,var(--bg) 55%);color:var(--text);font:15px system-ui,sans-serif}}
main{{width:min(1180px,100%);margin:auto;padding:22px 16px 60px}}h1{{font-size:25px;margin:0 0 5px}}.intro,.muted{{color:var(--muted)}}.intro{{margin:0 0 18px}}
.panel{{background:linear-gradient(145deg,#172833,#121e27);border:1px solid var(--line);border-radius:18px;padding:16px;margin:14px 0;box-shadow:0 10px 28px #0003}}
.controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}}label{{display:block;color:#b9c8d1;font-size:13px;margin-bottom:6px}}
select,input,button{{width:100%;padding:12px;border-radius:10px;border:1px solid #3a505d;background:#0c171e;color:var(--text);font:inherit}}
.actions{{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}}button{{background:#137b68;border:0;font-weight:750;cursor:pointer}}
.statusline{{display:flex;align-items:center;gap:9px;flex-wrap:wrap;color:var(--muted);font-size:13px}}.dot{{width:9px;height:9px;border-radius:50%;background:#71818b}}.dot.on{{background:var(--green);box-shadow:0 0 12px #21c58e99}}
.current{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}.big{{font-size:32px;font-weight:900;line-height:1.15}}.current-timeframe{{font-size:32px;font-weight:950;line-height:1.1;color:#b99aff}}.buy{{color:var(--green)}}.sell{{color:var(--red)}}.no-trade{{color:var(--gold)}}
.meta{{font-size:12px;color:var(--muted);line-height:1.8;margin-top:8px}}.section-title{{font-size:18px;margin:22px 2px 10px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px}}.event{{border:1px solid var(--line);border-radius:16px;padding:15px 16px;background:#15232d;min-width:0;transition:transform .15s,border-color .15s}}
.event:hover{{transform:translateY(-2px);border-color:#526d7b}}.event.old{{opacity:.72}}
.event-heading{{display:flex;align-items:center;justify-content:space-between;gap:10px}}
.event .side{{font-size:32px;font-weight:950;line-height:1.1}}
.event .timeframe-chip{{font-size:32px;font-weight:950;line-height:1.1;color:#b99aff}}
.event .scope{{font-size:12px;color:#aab9c2;margin-top:9px;line-height:1.7}}
.event .time{{font-size:12px;color:#aab9c2;margin-top:6px;line-height:1.7}}
.event .price{{font-size:13px;color:#d4dfe5;margin-top:7px}}
.risk-levels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}}
.risk-item{{background:#0d1921;border:1px solid #293d49;border-radius:10px;padding:8px 9px;min-width:0}}
.risk-label{{font-size:11px;color:#aab9c2;display:block;margin-bottom:4px}}
.risk-value{{font-size:14px;font-weight:800;overflow-wrap:anywhere}}
.risk-entry .risk-value{{color:var(--text)}}.risk-stop .risk-value{{color:var(--red)}}
.risk-target .risk-value{{color:var(--green)}}.risk-na{{grid-column:1/-1;color:#aab9c2;font-size:12px}}.risk-model-note{{grid-column:1/-1;color:#9aabb7;font-size:10px;line-height:1.5}}
#events:empty:after{{content:"هنوز سیگنالی در این بازه ثبت نشده است.";display:block;color:var(--muted);padding:18px}}#chart-frame{{width:100%;height:690px;border:0;border-radius:14px;background:#0b1117}}
.error{{color:#ff8999;white-space:pre-wrap}}.fresh{{color:var(--green)}}.stale{{color:var(--gold)}}
@media(max-width:560px){{main{{padding:16px 11px 40px}}.big{{font-size:29px}}#chart-frame{{height:520px}}.actions{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>داشبورد آنلاین سیگنال</h1><p class="intro">پایش خودکار بازار، تصمیم فعلی و آرشیو کارت‌های BUY / SELL؛ فقط دو استراتژی مجاز و بدون اجرای معامله.</p>
<form id="signal-form" class="panel"><div class="controls">
<div><label for="market">نماد بازار</label><select id="market" name="market">{markets}</select></div>
<div><label for="timeframe">تایم‌فریم</label><select id="timeframe" name="timeframe">{timeframes}</select></div>
<div><label for="strategy">استراتژی</label><select id="strategy" name="strategy">{strategies}</select></div>
<div><label for="days">بازه داده (روز؛ فقط طلا)</label><input id="days" name="days" type="number" min="1" max="90" value="30"></div></div>
<div class="actions"><button id="toggle" type="submit">شروع پایش آنلاین</button></div>
<div class="statusline" style="margin-top:13px"><span id="dot" class="dot"></span><span id="run-status">پایش هنوز شروع نشده است.</span><span id="last-update"></span></div></form>
<section class="panel"><div class="current"><div><div class="muted">تصمیم فعلی</div><div id="decision" class="big no-trade">NO TRADE</div></div><div id="current-timeframe" class="current-timeframe">1m</div><div class="meta" id="decision-meta">پس از شروع پایش، وضعیت بازار اینجا نمایش داده می‌شود.</div></div><div class="meta" id="last-signal">آخرین سیگنال معتبر پس از دریافت داده نمایش داده می‌شود.</div><div class="meta" id="data-meta"></div></section>
<h2 class="section-title">تاریخچه سیگنال‌ها <span class="muted" id="event-count"></span></h2><div id="events" class="cards"></div>
<h2 class="section-title">نمودار کندلی و نشانگرهای استراتژی</h2><section class="panel"><iframe id="chart-frame" title="نمودار کندلی"></iframe></section>
<p class="muted">فقط کندل‌های بسته‌شده بررسی می‌شوند. سطوح با همان RiskConfig موتور بک‌تست محاسبه می‌شوند: ATR(14) از میانگین سادهٔ True Range و حد ضرر ۲ برابر ATR. قیمت ورود و خروج با فرض کارمزد ۲ واحد پایه در هر سمت، اسپرد کل ۲ واحد پایه و لغزش ۱ واحد پایه در هر سمت برآورد می‌شوند؛ TPها برای بازده خالص ۱R، ۲R و ۳R تنظیم می‌شوند. قیمت‌های نمایش‌داده‌شده برآوردی‌اند، نه تضمین قیمت اجرای واقعی.</p><div id="error" class="error" role="alert"></div>
</main><script>
(() => {{
 const form=document.getElementById("signal-form"),toggle=document.getElementById("toggle"),dot=document.getElementById("dot"),status=document.getElementById("run-status"),error=document.getElementById("error"),decision=document.getElementById("decision"),currentTimeframe=document.getElementById("current-timeframe"),meta=document.getElementById("decision-meta"),lastSignal=document.getElementById("last-signal"),dataMeta=document.getElementById("data-meta"),events=document.getElementById("events"),frame=document.getElementById("chart-frame"),lastUpdate=document.getElementById("last-update");
 const marketSelect=document.getElementById("market"),daysInput=document.getElementById("days");
 function syncLookbackControl(){{const gold=marketSelect.value==="XAUUSD";daysInput.disabled=!gold;daysInput.title=gold?"بازه زمانی دریافت طلا":"برای کریپتو پنجره ثابت ۳۰۰ کندل بسته استفاده می‌شود.";}}
 syncLookbackControl();marketSelect.addEventListener("change",syncLookbackControl);
 let active=false,timer=null,busy=false,initialLoaded=false,lastConfig="",requestController=null;
 const sideClass=s=>s==="BUY"?"buy":s==="SELL"?"sell":"no-trade";
 function renderCards(items) {{
  events.replaceChildren();document.getElementById("event-count").textContent="("+items.length+")";
  const fmt=value=>Number(value).toLocaleString(undefined,{{maximumFractionDigits:8}});
  const addLevel=(grid,label,value,kind)=>{{
   const cell=document.createElement("div");cell.className="risk-item "+kind;
   const name=document.createElement("span");name.className="risk-label";name.textContent=label;
   const amount=document.createElement("strong");amount.className="risk-value";amount.textContent=fmt(value);
   cell.append(name,amount);grid.append(cell);
  }};
  for(const item of items) {{
   const card=document.createElement("article");card.className="event"+(item.age_seconds>3600?" old":"");
   const heading=document.createElement("div");heading.className="event-heading";
   const side=document.createElement("div");side.className="side "+sideClass(item.side);side.textContent=item.side;
   const timeframe=document.createElement("div");timeframe.className="timeframe-chip";timeframe.textContent=item.timeframe;
   heading.append(side,timeframe);
   const scope=document.createElement("div");scope.className="scope";scope.textContent=item.market+" · "+item.strategy;
   const time=document.createElement("div");time.className="time";time.textContent=item.signal_time_local+" · "+item.age_text;
   const price=document.createElement("div");price.className="price";price.textContent="قیمت بسته‌شدن کندل: "+fmt(item.close_price);
   const risk=document.createElement("div");risk.className="risk-levels";
   if([item.entry_price,item.stop_loss,item.tp1,item.tp2,item.tp3].every(value=>value!==null&&value!==undefined&&Number.isFinite(Number(value)))) {{
    addLevel(risk,"ENTRY",item.entry_price,"risk-entry");
    addLevel(risk,"STOP LOSS",item.stop_loss,"risk-stop");
    addLevel(risk,"TAKE PROFIT 1 · 1R",item.tp1,"risk-target");
    addLevel(risk,"TAKE PROFIT 2 · 2R",item.tp2,"risk-target");
    addLevel(risk,"TAKE PROFIT 3 · 3R",item.tp3,"risk-target");
   }} else {{
    const unavailable=document.createElement("div");unavailable.className="risk-na";
    unavailable.textContent="سطوح ریسک معتبر برای این سیگنال قابل محاسبه نیست؛ داده یا ساختار کافی موجود نیست.";
    risk.append(unavailable);
   }}
   const modelNote=document.createElement("div");modelNote.className="risk-model-note";modelNote.textContent=item.risk_model||"مدل هزینه و ریسک در زمان ثبت این رویداد موجود نبوده است.";risk.append(modelNote);
   card.append(heading,scope,time,price,risk);events.append(card);
  }}
 }}
 async function poll() {{
  if(busy||!active)return;busy=true;error.textContent="";
  const controller=new AbortController();requestController=controller;
  const requestTimeout=setTimeout(()=>controller.abort(),135000);
  try {{
   const q=new URLSearchParams(new FormData(form));
   const key=[q.get("market"),q.get("timeframe"),q.get("strategy"),q.get("days")].join("|");
   if(key!==lastConfig){{initialLoaded=false;lastConfig=key;}}
   if(initialLoaded)q.set("days","1");
   const response=await fetch("/api/poll?"+q,{{cache:"no-store",signal:controller.signal}});
   const data=await response.json();if(!response.ok)throw new Error(data.error||"دریافت داده ناموفق بود.");
   decision.textContent=data.decision_label;decision.className="big "+sideClass(data.decision_label);
   currentTimeframe.textContent=data.timeframe;
   meta.textContent=data.market+" · "+data.timeframe+" · "+data.strategy+(data.decision.reason?" · علت: "+data.decision.reason:"");
   if(data.latest_signal){{const age=Math.max(0,data.latest_signal.age_seconds||0);const ageText=age<3600?Math.floor(age/60)+" دقیقه قبل":Math.floor(age/3600)+" ساعت و "+Math.floor((age%3600)/60)+" دقیقه قبل";lastSignal.textContent="آخرین سیگنال این استراتژی: "+data.latest_signal.side+" · "+new Date(data.latest_signal.signal_time_utc).toLocaleString()+" · "+ageText;}}else{{lastSignal.textContent="در بازهٔ دریافت‌شده سیگنال تاریخی BUY یا SELL برای این استراتژی پیدا نشد.";}}
   dataMeta.textContent=(data.fresh?"داده تازه":"داده کهنه")+" · "+data.provider+" · "+data.source+" · "+data.closed_candles+" کندل بسته · تلاش ارائه‌دهندگان: "+data.provider_attempts+" · دریافت: "+data.retrieval_time_utc+(data.provider_failures.length?" · مسیرهای ناموفق: "+data.provider_failures.join(" / "):"")+(data.pagination_truncated?" · تاریخچه به سقف زمان دریافت رسید":"")+(data.fetch_elapsed_seconds!==null&&data.fetch_elapsed_seconds!==undefined?" · زمان دریافت: "+data.fetch_elapsed_seconds+"s":"");
   initialLoaded=true;renderCards(data.history||[]);if(data.chart)frame.srcdoc=data.chart;
   lastUpdate.textContent="آخرین بررسی: "+new Date(data.polled_at_utc).toLocaleTimeString();
   status.textContent="پایش فعال · بررسی خودکار هر 10 ثانیه";status.className=data.fresh?"fresh":"stale";
  }} catch(e) {{
   if(e.name==="AbortError"){{if(active){{error.textContent="مهلت دریافت داده پس از ۱۳۵ ثانیه تمام شد؛ درخواست بعدی دوباره تلاش می‌کند.";status.textContent="دریافت داده به مهلت زمانی رسید.";}}}}
   else{{error.textContent="خطا در دریافت داده: "+e.message;status.textContent="پایش فعال است؛ تلاش بعدی خودکار انجام می‌شود.";}}
  }}
  finally {{clearTimeout(requestTimeout);if(requestController===controller)requestController=null;busy=false;}}
 }}
 function start() {{if(active)return;active=true;dot.classList.add("on");toggle.textContent="توقف پایش";status.textContent="در حال اتصال به منبع بازار...";poll();timer=setInterval(poll,10000);}}
 function stop() {{active=false;if(timer)clearInterval(timer);timer=null;if(requestController)requestController.abort();dot.classList.remove("on");toggle.textContent="شروع پایش آنلاین";status.textContent="پایش متوقف شد.";}}
 form.addEventListener("submit",e=>{{e.preventDefault();if(active)stop();else start();}});
 for(const el of form.querySelectorAll("select,input"))el.addEventListener("change",()=>{{if(active){{error.textContent="";poll();}}}});
}})();
</script></body></html>"""

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path=="/api/poll": self._api_poll(parse_qs(parsed.query));return
        self._send(_page())
    def _api_poll(self,query):
        try:
            market=validate_market_or_crypto_symbol(query.get("market",["XAUUSD"])[0])
            timeframe=validate_timeframe(query.get("timeframe",["1m"])[0])
            strategy=query.get("strategy",["Bollinger_Reversion"])[0]
            if strategy not in STRATEGY_REGISTRY: raise ValueError("استراتژی انتخاب‌شده مجاز نیست.")
            days=int(query.get("days",["30"])[0])
            if not 1<=days<=90: raise ValueError("بازه داده باید بین ۱ تا ۹۰ روز باشد.")
            self._send_json(_poll(market,timeframe,strategy,days))
        except Exception as exc: self._send_json({"error":str(exc)},status=400)
    def _send(self,body,status=200):
        encoded=body.encode("utf-8");self.send_response(status)
        self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(encoded)));self.send_header("Cache-Control","no-store");self.end_headers()
        try:self.wfile.write(encoded)
        except BrokenPipeError:return
    def _send_json(self,payload,status=200):
        encoded=json.dumps(payload,ensure_ascii=False,default=str).encode("utf-8");self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(encoded)));self.send_header("Cache-Control","no-store");self.end_headers()
        try:self.wfile.write(encoded)
        except BrokenPipeError:return
    def log_message(self,_format,*_args):return

def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Local live signal dashboard")
    parser.add_argument("--host",default="127.0.0.1",help="Local interface; do not expose publicly")
    parser.add_argument("--port",type=int,default=8765);parser.add_argument("--no-browser",action="store_true");args=parser.parse_args(argv)
    if args.host not in {"127.0.0.1","localhost"}:parser.error("The dashboard is local-only; --host must be 127.0.0.1 or localhost.")
    if not 1<=args.port<=65535:parser.error("--port must be between 1 and 65535.")
    server=ThreadingHTTPServer((args.host,args.port),DashboardHandler);url=f"http://127.0.0.1:{args.port}/"
    print(f"Signal dashboard: {url} (Ctrl+C to stop)",flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return 0
if __name__=="__main__":raise SystemExit(main())
