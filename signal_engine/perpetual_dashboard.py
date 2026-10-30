"""Separate Perpetual dashboard page and API.

This page has its own route and request parameters. The existing Spot/XAUUSD
dashboard remains the default route and keeps its current data path.
"""
from __future__ import annotations

import html
import json
from datetime import timedelta

import pandas as pd

from signal_engine.constants import TIMEFRAMES, validate_crypto_symbol, validate_timeframe
from signal_engine.latest_signal import latest_decision_report
from signal_engine.risk import RiskConfig
from signal_engine.strategies import STRATEGY_REGISTRY
from signal_engine.strategies.base import transition_direction
from signal_engine.data.perpetual_source import PROVIDERS, fetch_range, list_symbols

TIMEFRAME_DURATION = {
    "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15), "30m": timedelta(minutes=30),
    "1H": timedelta(hours=1), "4H": timedelta(hours=4), "1D": timedelta(days=1),
}


def _page() -> str:
    provider_options = "".join(
        f'<option value="{html.escape(p)}">{html.escape(p)}</option>' for p in PROVIDERS
    )
    market_options = '<select id="symbol" name="symbol" required disabled><option value="">در حال دریافت نمادهای قرارداد...</option></select>'
    timeframe_options = "".join(
        f'<option value="{tf}"{" selected" if tf == "1m" else ""}>{tf}</option>'
        for tf in TIMEFRAMES
    )
    strategy_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in STRATEGY_REGISTRY
    )
    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perpetual · Independent Signal Engine</title>
<style>
:root{{color-scheme:dark;--bg:#0b1218;--panel:#14232d;--line:#2a414e;--muted:#9aabb7;--text:#edf4f7;--green:#21c58e;--red:#ff6177;--gold:#f3c76a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(160deg,#101e28,var(--bg) 55%);color:var(--text);font:15px system-ui,sans-serif}}
main{{width:min(1400px,100%);margin:auto;padding:20px 15px 55px}}h1{{margin:0;font-size:25px}}h2{{font-size:18px;margin:22px 2px 10px}}
.muted{{color:var(--muted)}}.intro{{margin:5px 0 18px;color:var(--muted)}}
.panel{{background:linear-gradient(145deg,#172833,#121e27);border:1px solid var(--line);border-radius:16px;padding:15px;margin:13px 0}}
.controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px}}
label{{display:block;color:#b9c8d1;font-size:13px;margin-bottom:6px}}
select,input,button{{width:100%;padding:11px;border-radius:10px;border:1px solid #3a505d;background:#0c171e;color:var(--text);font:inherit}}
button{{background:#137b68;border:0;font-weight:750;cursor:pointer;margin-top:12px}}
.sourcebar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}}
.provider-pill{{display:inline-flex;padding:8px 13px;border-radius:9px;border:1px solid #577080;background:#0b1821;font-weight:800;letter-spacing:.2px}}
.market-pill{{color:#b9c9ff}}.status{{color:var(--muted);font-size:13px}}.green{{color:var(--green)}}.red{{color:var(--red)}}.gold{{color:var(--gold)}}
.decision{{font-size:30px;font-weight:900}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}}
.metric{{padding:10px;border:1px solid var(--line);border-radius:10px;background:#0d1921}}.metric b{{display:block;font-size:16px;margin-top:5px}}
#chart{{display:block;width:100%;height:620px;background:#0b1117;border-radius:12px;touch-action:pan-y}}
.chart-help{{font-size:12px;color:var(--muted);line-height:1.8;margin-top:8px}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:850px;font-size:13px}}
th,td{{padding:10px 9px;border-bottom:1px solid #263946;text-align:right;white-space:nowrap}}
th{{color:#b9c8d1;background:#101d26;position:sticky;top:0}}.buy{{color:var(--green);font-weight:900}}.sell{{color:var(--red);font-weight:900}}
.error{{color:#ff8999;white-space:pre-wrap}}.warning{{color:var(--gold)}}
@media(max-width:600px){{main{{padding:14px 10px 35px}}#chart{{height:470px}}.decision{{font-size:26px}}}}
</style></head><body><main>
<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap"><h1>بازار Perpetual</h1><a href="/" style="color:#8fd9c2;font-weight:750;text-decoration:none">بازگشت به داشبورد اصلی ←</a></div>
<p class="intro">صفحه مستقل قراردادهای دائمی USDT؛ انتخاب صرافی اجباری است و هیچ داده‌ای از Spot یا صرافی دیگر جایگزین نمی‌شود.</p>
<form id="controls" class="panel"><div class="controls">
<div><label for="provider">منبع دقیق داده</label><select id="provider" name="provider">{provider_options}</select></div>
<div><label for="symbol">نماد قرارداد Perpetual (USDT)</label>{market_options}</div>
<div><label for="timeframe">تایم‌فریم مستقل</label><select id="timeframe" name="timeframe">{timeframe_options}</select></div>
<div><label for="strategy">استراتژی</label><select id="strategy" name="strategy">{strategy_options}</select></div>
<div><label for="days">تاریخچه کندل‌ها (روز، حداکثر ۳۰)</label><input id="days" name="days" type="number" min="1" max="30" value="30"></div>
</div><button type="submit">دریافت نمودار و سیگنال‌ها</button>
<div class="status" id="status">برای دریافت داده، صرافی و تایم‌فریم را انتخاب کنید.</div></form>
<section class="panel">
<div class="sourcebar"><span class="provider-pill" id="source-label">Perpetual · Provider not loaded</span><span class="provider-pill market-pill" id="contract-label">Contract: —</span><span class="status" id="retrieval-label"></span></div>
<div class="metrics">
<div class="metric"><span class="muted">تصمیم فعلی</span><b class="decision gold" id="decision">NO TRADE</b></div>
<div class="metric"><span class="muted">آخرین قیمت بسته</span><b id="last-price">—</b></div>
<div class="metric"><span class="muted">کندل‌های نمایش‌داده‌شده</span><b id="candle-count">—</b></div>
<div class="metric"><span class="muted">آخرین کندل UTC</span><b id="last-candle">—</b></div>
<div class="metric"><span class="muted">تازگی داده</span><b id="freshness">—</b></div>
</div><p class="status" id="decision-reason"></p><p class="status" id="signal-diagnostics">پس از دریافت داده، تعداد رویدادهای استراتژی و وضعیت محاسبه ریسک نمایش داده می‌شود.</p>
</section>
<h2>نمودار کندلی Perpetual و سطوح سیگنال</h2>
<section class="panel"><canvas id="chart" aria-label="Perpetual candlestick chart"></canvas>
<div class="chart-help">نشانگر BUY/SELL روی کندل تصمیم نمایش داده می‌شود. خطوط افقی هر سیگنال شامل Entry، Stop Loss و TP1/TP2/TP3 است. همه زمان‌ها UTC هستند. نمودار فقط پنج کندل بسته‌شده اخیرِ تایم‌فریم انتخابی و جدول فقط پنج سیگنال اخیر را نمایش می‌دهد. خطوط سطوح فقط برای سیگنال آنلاین جاری نمایش داده می‌شوند.</div></section>
<h2>سیگنال‌های همین نمودار <span class="muted" id="signal-count"></span></h2>
<section class="panel table-wrap"><table><thead><tr><th>جهت</th><th>زمان تصمیم UTC</th><th>استراتژی</th><th>Entry</th><th>Stop Loss</th><th>TP1</th><th>TP2</th><th>TP3</th><th>وضعیت سطوح ریسک</th></tr></thead><tbody id="signals"><tr><td colspan="9" class="muted">هنوز داده دریافت نشده است.</td></tr></tbody></table></section>
<p class="status">فقط کندل بسته‌شده؛ داده Perpetual از همان صرافی انتخاب‌شده؛ بدون سفارش‌گذاری. اگر قرارداد یا نماد در صرافی انتخابی موجود نباشد، درخواست با خطا متوقف می‌شود و منبع دیگری جایگزین نمی‌شود. این صفحه قیمت Trade/Contract را نمایش می‌دهد، نه Mark Price یا Index Price. محاسبه سطوح فعلاً Funding را لحاظ نمی‌کند؛ سطوح، برآورد هندسی ATR و هزینه عمومی موتور هستند.</p>
<div id="error" class="error" role="alert"></div>
</main><script>
(() => {{
const form=document.getElementById("controls"),status=document.getElementById("status"),error=document.getElementById("error");
const canvas=document.getElementById("chart"),ctx=canvas.getContext("2d");
let candleData=[],signalData=[],chartSignalData=[];
const fmt=v=>v===null||v===undefined||!Number.isFinite(Number(v))?"—":Number(v).toLocaleString(undefined,{{maximumFractionDigits:8}});
const el=id=>document.getElementById(id);
async function loadSymbols(){{
 const provider=el("provider"),symbol=el("symbol"),previous=symbol.value;
 symbol.disabled=true;symbol.replaceChildren(new Option("در حال دریافت قراردادهای فعال...",""));
 status.textContent="در حال دریافت فهرست قراردادهای فعال "+provider.value+"...";
 try{{
  const response=await fetch("/api/perpetual/symbols?provider="+encodeURIComponent(provider.value),{{cache:"no-store"}});
  const data=await response.json();if(!response.ok)throw new Error(data.error||"دریافت فهرست نمادها ناموفق بود");
  symbol.replaceChildren();
  for(const value of data.symbols)symbol.add(new Option(value,value));
  const selected=data.symbols.includes(previous)?previous:(data.symbols.includes("BTCUSDT")?"BTCUSDT":data.symbols[0]);
  if(selected)symbol.value=selected;
  symbol.disabled=!data.symbols.length;
  status.textContent=data.symbols.length+" قرارداد فعال USDT از "+provider.value+" دریافت شد.";
 }}catch(err){{
  symbol.replaceChildren(new Option("فهرست قراردادها در دسترس نیست",""));
  symbol.disabled=true;status.textContent="دریافت فهرست نمادها ناموفق";error.textContent=err.message;
 }}
}}
function draw(){{
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=Math.max(1,Math.floor(rect.width*dpr));canvas.height=Math.max(1,Math.floor(rect.height*dpr));
 ctx.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;
 ctx.fillStyle="#0b1117";ctx.fillRect(0,0,w,h);
 if(!candleData.length){{ctx.fillStyle="#9aabb7";ctx.fillText("No perpetual candles loaded",20,30);return;}}
 const left=74,right=18,top=18,bottom=34,pw=w-left-right,ph=h-top-bottom;
 let lo=Infinity,hi=-Infinity;for(const c of candleData){{lo=Math.min(lo,c.low);hi=Math.max(hi,c.high);}}
 for(const s of chartSignalData)for(const v of [s.entry,s.stop,s.tp1,s.tp2,s.tp3])if(Number.isFinite(v)){{lo=Math.min(lo,v);hi=Math.max(hi,v);}}
 const pad=Math.max((hi-lo)*.05,Math.abs(hi)*.0002,1e-9);lo-=pad;hi+=pad;
 const y=p=>top+(hi-p)/(hi-lo)*ph;
 ctx.font="11px system-ui";ctx.textAlign="right";ctx.fillStyle="#aebbc5";
 for(let n=0;n<=5;n++){{const p=lo+(hi-lo)*n/5,yy=y(p);ctx.strokeStyle="#263640";ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.fillText(fmt(p),left-7,yy+4);}}
 const step=pw/candleData.length,bw=Math.max(.7,Math.min(8,step*.62));
 candleData.forEach((c,i)=>{{const x=left+(i+.5)*step,up=c.close>=c.open,col=up?"#21c58e":"#ff6177";ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x,y(c.high));ctx.lineTo(x,y(c.low));ctx.stroke();const yy=Math.min(y(c.open),y(c.close)),bh=Math.max(1,Math.abs(y(c.open)-y(c.close)));ctx.fillRect(x-bw/2,yy,bw,bh);}});
 const first=candleData[0].t,last=candleData[candleData.length-1].t;
 const xAt=t=>left+((t-first)/(last-first||1))*pw;
 chartSignalData.forEach(s=>{{
  if(s.t<first||s.t>last)return;const x=xAt(s.t),buy=s.side==="BUY";
  ctx.strokeStyle=buy?"#21c58e":"#ff6177";ctx.fillStyle=ctx.strokeStyle;
  const yy=y(s.entry);ctx.beginPath();ctx.arc(x,yy,5,0,Math.PI*2);ctx.fill();
  ctx.font="bold 10px system-ui";ctx.textAlign="center";ctx.fillText(buy?"B":"S",x,yy-9);
  const levels=[["ENTRY",s.entry,"#dbe7ee"],["SL",s.stop,"#ff6177"],["TP1",s.tp1,"#21c58e"],["TP2",s.tp2,"#21c58e"],["TP3",s.tp3,"#21c58e"]];
  const endX=Math.min(w-right,x+Math.max(35,pw*.12));
  levels.forEach(([name,val,color])=>{{if(!Number.isFinite(val))return;const ly=y(val);ctx.strokeStyle=color;ctx.globalAlpha=.72;ctx.setLineDash(name==="ENTRY"?[]:[4,3]);ctx.beginPath();ctx.moveTo(x,ly);ctx.lineTo(endX,ly);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;ctx.fillStyle=color;ctx.textAlign="left";ctx.font="9px system-ui";ctx.fillText(name,endX+2,ly-2);}});
 }});
 ctx.fillStyle="#aebbc5";ctx.textAlign="left";ctx.font="10px system-ui";
 const dateFmt=t=>new Date(t).toISOString().replace("T"," ").slice(0,16)+" UTC";
 ctx.fillText(dateFmt(first),left,h-9);ctx.textAlign="right";ctx.fillText(dateFmt(last),w-right,h-9);
}}
function render(data){{
 candleData=data.candles;signalData=data.signals;
 el("source-label").textContent=data.provider+" PERPETUAL · USDT-M";
 el("contract-label").textContent="Contract: "+data.contract_symbol;
 el("retrieval-label").textContent=data.source+" · timestamp unit: "+data.timestamp_unit+" · retrieved "+data.retrieval_time_utc;
 el("decision").textContent=data.decision.label||"NO TRADE";
 el("decision").className="decision "+(data.decision.label==="BUY"?"green":data.decision.label==="SELL"?"red":"gold");
 el("decision-reason").textContent=data.decision.reason||"";
 const currentSide=data.fresh&&["BUY","SELL"].includes(data.decision.label)?data.decision.label:null;
 chartSignalData=currentSide?signalData.filter(s=>s.side===currentSide&&s.t>=candleData[0].t&&s.t<=candleData[candleData.length-1].t).slice(-1):[];
 el("last-price").textContent=fmt(candleData[candleData.length-1].close);
 el("candle-count").textContent=candleData.length.toLocaleString();
 el("last-candle").textContent=new Date(candleData[candleData.length-1].t).toISOString();
 el("freshness").textContent=data.fresh?"FRESH":"STALE";
 el("freshness").className=data.fresh?"green":"gold";
 el("signal-count").textContent="("+Math.min(signalData.length,5)+")";
 const d=data.signal_diagnostics||{};el("signal-diagnostics").textContent="رویدادهای خام استراتژی: "+(d.strategy_events??"—")+" · رویدادهای دارای کندل اجرای بعدی: "+(d.actionable_events??"—")+" · سیگنال‌های منتظر کندل بعدی: "+(d.pending_events??"—")+" · سیگنال با سطوح ریسک ناموجود: "+(d.risk_levels_unavailable??"—");
 const tbody=el("signals");tbody.replaceChildren();
 if(!signalData.length){{const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=9;td.textContent="در این بازه رویداد قابل‌اقدام BUY/SELL ثبت نشده است؛ جزئیات در بخش تشخیص سیگنال آمده است.";tr.append(td);tbody.append(tr);}}
 for(const s of signalData.slice(-5).reverse()){{
  const tr=document.createElement("tr"),values=[s.side,new Date(s.t).toISOString(),s.strategy,fmt(s.entry),fmt(s.stop),fmt(s.tp1),fmt(s.tp2),fmt(s.tp3),s.risk_status==="AVAILABLE"?"سطوح آماده":"سطوح ریسک ناموجود"];
  values.forEach((v,i)=>{{const td=document.createElement("td");td.textContent=v;if(i===0)td.className=s.side==="BUY"?"buy":"sell";tr.append(td);}});tbody.append(tr);
 }}
 draw();
}}
form.addEventListener("submit",async e=>{{
 e.preventDefault();error.textContent="";status.textContent="در حال دریافت کندل‌های بازار انتخاب‌شده...";status.className="status";
 try{{
  const q=new URLSearchParams(new FormData(form));
  const response=await fetch("/api/perpetual/poll?"+q.toString(),{{cache:"no-store"}});
  const data=await response.json();if(!response.ok)throw new Error(data.error||"Perpetual provider request failed");
  render(data);status.textContent="دریافت موفق · "+data.candles.length+" کندل در نمودار · "+data.signals.length+" سیگنال اخیر در Log · بدون fallback";status.className="status green";
 }}catch(err){{status.textContent="دریافت ناموفق";status.className="status red";error.textContent=err.message;}}
}});
el("provider").addEventListener("change",loadSymbols);
loadSymbols();
window.addEventListener("resize",()=>{{if(candleData.length)draw();}});
}})();
</script></body></html>"""


def _poll(provider: str, symbol: str, timeframe: str, strategy_name: str, days: int) -> dict:
    if provider not in PROVIDERS:
        raise ValueError("ارائه‌دهنده باید یکی از Bitget، KuCoin، HTX یا MEXC باشد.")
    market = validate_crypto_symbol(symbol)
    tf = validate_timeframe(timeframe)
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError("استراتژی انتخاب‌شده مجاز نیست.")
    if not 1 <= days <= 30:
        raise ValueError("بازه تاریخچه باید بین ۱ تا ۳۰ روز باشد؛ برای دریافت تاریخچه طولانی‌تر باید سقف صفحه‌بندی هر ارائه‌دهنده جداگانه افزایش یابد.")
    now = pd.Timestamp.now(tz="UTC")
    duration = pd.Timedelta(TIMEFRAME_DURATION[tf])
    start = now - pd.Timedelta(days=days)
    df = fetch_range(provider, market, tf, start, now)
    df = df.loc[(df.index + duration) <= now]
    if df.empty:
        raise ValueError("هیچ کندل بسته‌شده‌ای برای این قرارداد دریافت نشد.")
    strategy = STRATEGY_REGISTRY[strategy_name]()
    decision_report = latest_decision_report(
        df, strategy, RiskConfig(), market, tf,
        provenance=str(df.attrs.get("source", "public perpetual market data")),
    )
    age = max(0.0, (now - (df.index[-1] + duration)).total_seconds())
    fresh = age <= duration.total_seconds() * 3
    decision = decision_report.get("decision", {})
    if not fresh:
        decision = {"label": "NO TRADE", "reason": "stale_data"}
    # Import lazily so the original dashboard remains the stable entry point.
    from signal_engine.dashboard import _risk_levels_for_event, RISK_CONFIG
    transitions = transition_direction(strategy.raw_direction(df))
    atr_values = __import__("signal_engine.indicators", fromlist=["atr"]).atr(
        df["high"], df["low"], df["close"], RISK_CONFIG.atr_window
    )
    signals = []
    signal_diagnostics = {
        "strategy_events": 0,
        "actionable_events": 0,
        "pending_events": 0,
        "risk_levels_unavailable": 0,
    }
    for stamp, side_value in transitions.items():
        side = int(side_value)
        if side not in {-1, 1}:
            continue
        signal_diagnostics["strategy_events"] += 1
        idx = int(df.index.get_indexer([stamp])[0])
        if idx < 0:
            continue
        # A decision is actionable only when the following candle exists.
        # The newest candle's signal remains pending; never invent its entry.
        if idx + 1 >= len(df):
            signal_diagnostics["pending_events"] += 1
            continue
        signal_diagnostics["actionable_events"] += 1
        levels = _risk_levels_for_event(df, idx, side, atr_values, market, tf)
        if not levels:
            signal_diagnostics["risk_levels_unavailable"] += 1
        decision_stamp = pd.Timestamp(df.index[idx + 1])
        signals.append({
            "t": int(decision_stamp.value // 1_000_000),
            "side": "BUY" if side > 0 else "SELL",
            "strategy": strategy_name,
            "entry": levels["entry_price"] if levels else None,
            "stop": levels["stop_loss"] if levels else None,
            "tp1": levels["tp1"] if levels else None,
            "tp2": levels["tp2"] if levels else None,
            "tp3": levels["tp3"] if levels else None,
            "risk_status": "AVAILABLE" if levels else "UNAVAILABLE",
        })
    candles = [{
        "t": int(pd.Timestamp(stamp).value // 1_000_000),
        "open": float(row.open), "high": float(row.high),
        "low": float(row.low), "close": float(row.close),
    } for stamp, row in df.tail(5).iterrows()]
    return {
        "provider": provider, "market_type": "PERPETUAL", "symbol": market,
        "contract_symbol": df.attrs["contract_symbol"], "timeframe": tf,
        "strategy": strategy_name, "source": df.attrs["source"],
        "timestamp_unit": df.attrs["timestamp_unit"],
        "retrieval_time_utc": df.attrs["retrieval_time_utc"],
        "candles": candles, "closed_candle_count": int(len(df)), "chart_candle_limit": 5,
        "signals": signals[-5:], "signal_diagnostics": signal_diagnostics, "decision": decision,
        "fresh": fresh, "data_age_seconds": int(age),
        "latest_closed_candle_utc": df.index[-1].isoformat(),
        "validation_report": df.attrs["validation_report"],
        "pagination_truncated": bool(df.attrs.get("pagination_truncated", False)),
    }


def api_symbols(query: dict[str, list[str]]) -> dict:
    provider = query.get("provider", ["Bitget"])[0]
    return {"provider": provider, "symbols": list_symbols(provider)}

def api_poll(query: dict[str, list[str]]) -> dict:
    provider = query.get("provider", ["Bitget"])[0]
    symbol = query.get("symbol", ["BTCUSDT"])[0]
    timeframe = query.get("timeframe", ["1m"])[0]
    strategy = query.get("strategy", ["Bollinger_Reversion"])[0]
    try:
        days = int(query.get("days", ["30"])[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("تعداد روز باید عدد صحیح باشد.") from exc
    return _poll(provider, symbol, timeframe, strategy, days)
