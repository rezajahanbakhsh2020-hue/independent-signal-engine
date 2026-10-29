"""Dependency-free interactive signal chart exported as a standalone HTML file."""
from __future__ import annotations

import html
import json

import pandas as pd

from signal_engine.strategies.base import Strategy, transition_direction


def build_chart_html(
    df: pd.DataFrame,
    *,
    market: str,
    timeframe: str,
    strategy_name: str,
    strategy_params: dict,
    strategy: Strategy,
    latest_decision: dict,
    max_candles: int = 300,
) -> str:
    """Render OHLC candles and executed BUY/SELL transition markers as standalone SVG/HTML."""
    if df.empty:
        raise ValueError("cannot chart an empty candle frame")
    view = df.tail(max_candles).copy()
    directions = transition_direction(strategy.raw_direction(df))
    signals = []
    index_positions = {stamp: i for i, stamp in enumerate(df.index)}
    for stamp, direction in directions.items():
        if int(direction) == 0:
            continue
        source_pos = index_positions[stamp]
        execution_pos = source_pos + 1
        if execution_pos >= len(df):
            continue
        execution_stamp = df.index[execution_pos]
        if execution_stamp not in view.index:
            continue
        signals.append({
            "timestamp": execution_stamp.isoformat(),
            "direction": "BUY" if int(direction) > 0 else "SELL",
            "price": float(df.iloc[execution_pos]["open"]),
        })

    width, height = 1200, 650
    left, right, top, bottom = 86, 28, 48, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    lows = view["low"].astype(float)
    highs = view["high"].astype(float)
    low = float(lows.min())
    high = float(highs.max())
    pad = max((high - low) * 0.06, abs(high) * 0.0005, 1e-9)
    low -= pad
    high += pad

    def y(price: float) -> float:
        return top + (high - price) / (high - low) * plot_h

    count = len(view)
    step = plot_w / max(count, 1)
    body_w = max(1.2, min(9.0, step * 0.58))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(market)} {html.escape(timeframe)} candlestick chart">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#101820"/>',
    ]
    for n in range(6):
        price = low + (high - low) * n / 5
        yy = y(price)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#293640" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" fill="#b9c6cf" font-size="12">{price:.6g}</text>')
    for i, (stamp, row) in enumerate(view.iterrows()):
        x = left + (i + 0.5) * step
        op, hi, lo, cl = (float(row[k]) for k in ("open", "high", "low", "close"))
        color = "#25c79a" if cl >= op else "#ff657a"
        parts.append(f'<line x1="{x:.2f}" y1="{y(hi):.2f}" x2="{x:.2f}" y2="{y(lo):.2f}" stroke="{color}" stroke-width="1.4"/>')
        by, bh = min(y(op), y(cl)), max(abs(y(op)-y(cl)), 1.2)
        parts.append(f'<rect x="{x-body_w/2:.2f}" y="{by:.2f}" width="{body_w:.2f}" height="{bh:.2f}" fill="{color}" rx="0.7"/>')
        if i in (0, count//2, count-1):
            label = pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M UTC")
            parts.append(f'<text x="{x:.2f}" y="{height-70}" text-anchor="middle" fill="#b9c6cf" font-size="12">{html.escape(label)}</text>')
    first_stamp = view.index[0]
    last_stamp = view.index[-1]
    for sig in signals:
        stamp = pd.Timestamp(sig["timestamp"])
        if stamp < first_stamp or stamp > last_stamp:
            continue
        i = view.index.get_loc(stamp)
        x = left + (i + 0.5) * step
        yy = y(sig["price"])
        is_buy = sig["direction"] == "BUY"
        marker_y = yy - 17 if is_buy else yy + 17
        color = "#25c79a" if is_buy else "#ff657a"
        parts.append(f'<circle cx="{x:.2f}" cy="{marker_y:.2f}" r="9" fill="{color}" stroke="#101820" stroke-width="2"/>')
        parts.append(f'<text x="{x:.2f}" y="{marker_y+4:.2f}" text-anchor="middle" fill="#101820" font-weight="700" font-size="9">{"B" if is_buy else "S"}</text>')
    parts.append("</svg>")

    decision = latest_decision.get("decision", {})
    label = str(decision.get("label", "NO TRADE"))
    reason = str(decision.get("reason") or "")
    badge_color = "#25c79a" if label == "BUY" else "#ff657a" if label == "SELL" else "#f2c14e"
    title = f"{market} · {timeframe} · {strategy_name}"
    status = f"{label}{': ' + reason if reason else ''}"
    meta = {
        "market": market,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "strategy_params": strategy_params,
        "latest_decision": latest_decision,
        "candles_shown": len(view),
        "signals_shown": len(signals),
        "chart_note": "Markers are placed at the next candle open where the decision would be acted on; prior signals are historical and are not live orders.",
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} signal chart</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1117;color:#e7edf2;font:15px system-ui,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:22px}}h1{{font-size:22px;margin:0 0 8px}}.sub{{color:#aebbc5;margin-bottom:18px}}
.panel{{background:#101820;border:1px solid #293640;border-radius:14px;padding:14px 18px;margin:12px 0}}
.signal{{font-size:20px;font-weight:750;color:{badge_color}}}.muted{{color:#aebbc5}}svg{{display:block;width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#b9c6cf;font-size:12px}}
</style></head><body><main><h1>{html.escape(title)}</h1>
<div class="sub">Public market data · {len(view)} closed candles displayed · no order execution</div>
<div class="panel"><div class="muted">Latest decision</div><div class="signal">{html.escape(status)}</div>
<div class="muted">Selected strategy: {html.escape(strategy_name)} · Parameters: {html.escape(json.dumps(strategy_params, sort_keys=True))}</div></div>
<div class="panel">{"".join(parts)}</div>
<div class="panel muted">BUY/SELL markers are historical strategy transitions plotted at the next candle open. They are not orders and do not imply profitability. A NO TRADE result is valid when the strategy has no entry condition.</div>
<details class="panel"><summary>Decision and data details</summary><pre>{html.escape(json.dumps(meta, indent=2, default=str))}</pre></details>
</main></body></html>"""
