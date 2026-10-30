""""Small local dashboard for one-step market signal viewing.

The dashboard is deliberately local-only: it fetches public candles, computes
one of the two registered strategies, and renders the existing standalone
candlestick chart. It never places orders or stores credentials.
"""
from __future__ import annotations

import argparse
import html
import webbrowser
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from signal_engine.chart import build_chart_html
from signal_engine.constants import TIMEFRAMES, validate_market_or_crypto_symbol, validate_timeframe
from signal_engine.latest_signal import latest_decision_report
from signal_engine.risk import RiskConfig
from signal_engine.strategies import STRATEGY_REGISTRY

TIMEFRAME_DURATION = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "1D": timedelta(days=1),
}


def _market_data(market: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp):
    if market == "XAUUSD":
        from signal_engine.data.biquote_source import fetch_range
        return fetch_range(market, timeframe, start, end)
    from signal_engine.data.binance_source import fetch_range
    return fetch_range(market, timeframe, start, end)


def _page(message: str = "", chart: str = "", *, market: str = "BTCUSDT",
          timeframe: str = "1H", strategy: str = "Bollinger_Reversion", days: int = 30) -> str:
    strategy_options = "".join(
        f'<option value="{html.escape(name)}"{" selected" if name == strategy else ""}>{html.escape(name)}</option>'
        for name in STRATEGY_REGISTRY
    )
    market_options = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "XAUUSD")
    market_select_options = "".join(
        f'<option value="{symbol}"{" selected" if symbol == market else ""}>{symbol}</option>'
        for symbol in market_options
    )
    timeframe_options = "".join(
        f'<option value="{tf}"{" selected" if tf == timeframe else ""}>{tf}</option>'
        for tf in TIMEFRAMES
    )
    safe_chart = html.escape(chart, quote=True)
    frame = (
        f'<iframe title="نمودار و سیگنال" srcdoc="{safe_chart}" '
        'style="width:100%;height:760px;border:0;border-radius:12px;background:#0b1117"></iframe>'
        if chart else ""
    )
    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>داشبورد سیگنال</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#101820;color:#edf2f6;font:15px system-ui,sans-serif}}
main{{max-width:1100px;margin:24px auto;padding:0 16px}}h1{{font-size:23px;margin:0 0 8px}}
p,.muted{{color:#aebbc5}}form,.panel{{background:#192630;border:1px solid #34434e;border-radius:12px;padding:16px;margin:14px 0}}
.controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
label{{display:block;color:#bdc9d1;font-size:13px;margin-bottom:6px}}
input,select,button{{width:100%;padding:11px;border-radius:8px;border:1px solid #52616c;background:#101820;color:#edf2f6;font:inherit}}
button{{background:#197c68;border:0;font-weight:700;cursor:pointer;margin-top:14px}}
button:disabled{{opacity:.65;cursor:wait}}
.notice{{color:#f2c66d;white-space:pre-wrap}}
.loading{{display:none;color:#f2c66d;background:#26313a;border-radius:8px;padding:12px;margin:12px 0}}
.loading.visible{{display:block}}
iframe{{display:block;margin-top:14px}}
</style></head><body><main>
<h1>داشبورد سیگنال</h1>
<p>دریافت داده، محاسبه سیگنال و نمودار در یک صفحه؛ فقط دو استراتژی فعال. بدون اجرای معامله.</p>
<form id="signal-form" method="get" action="/">
<div class="controls">
<div><label for="market">نماد بازار</label><select id="market" name="market">{market_select_options}</select></div>
<div><label for="timeframe">تایم‌فریم</label><select id="timeframe" name="timeframe">{timeframe_options}</select></div>
<div><label for="strategy">استراتژی</label><select id="strategy" name="strategy">{strategy_options}</select></div>
<div><label for="days">بازه داده (روز)</label><input id="days" name="days" type="number" min="1" max="90" value="{days}"></div>
</div><button id="submit-button" type="submit">دریافت آخرین سیگنال و نمایش نمودار</button>
<div id="loading-status" class="loading" role="status" aria-live="polite">در حال دریافت داده‌های بازار و محاسبه سیگنال؛ لطفاً منتظر بمانید...</div>
</form>
{f'<div class="panel notice">{html.escape(message)}</div>' if message else ''}
{frame}
<div class="muted">داده کهنه یا کندل ناتمام به‌عنوان سیگنال معتبر نمایش داده نمی‌شود. نشانگرهای گذشته سفارش واقعی نیستند.</div>
</main>
<script>
(() => {{
  const form = document.getElementById("signal-form");
  const button = document.getElementById("submit-button");
  const status = document.getElementById("loading-status");
  if (!form || !button || !status) return;
  form.addEventListener("submit", () => {{
    status.classList.add("visible");
    button.disabled = true;
    button.textContent = "در حال دریافت داده...";
  }});
}})();
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not query:
            self._send(_page())
            return
        market = query.get("market", ["BTCUSDT"])[0].strip().upper()
        timeframe = query.get("timeframe", ["1H"])[0]
        strategy_name = query.get("strategy", ["Bollinger_Reversion"])[0]
        try:
            market = validate_market_or_crypto_symbol(market)
            timeframe = validate_timeframe(timeframe)
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError("استراتژی انتخاب‌شده مجاز نیست.")
            days = int(query.get("days", ["30"])[0])
            if not 1 <= days <= 90:
                raise ValueError("بازه داده باید بین ۱ تا ۹۰ روز باشد.")
            now = pd.Timestamp.now(tz="UTC")
            end = now
            start = end - pd.Timedelta(days=days)
            df = _market_data(market, timeframe, start, end)
            duration = pd.Timedelta(TIMEFRAME_DURATION[timeframe])
            df = df.loc[(df.index + duration) <= now]
            if df.empty:
                raise ValueError("داده کندل بسته‌شده برای این نماد و تایم‌فریم دریافت نشد.")
            age_hours = (now - (df.index[-1] + duration)).total_seconds() / 3600
            max_age_hours = duration.total_seconds() * 3 / 3600
            fresh = age_hours >= 0 and age_hours <= max_age_hours
            strategy = STRATEGY_REGISTRY[strategy_name]()
            decision = latest_decision_report(
                df, strategy, RiskConfig(), market, timeframe,
                provenance=str(df.attrs.get("source", "public market data")),
            )
            if not fresh:
                decision["decision"] = {
                    "label": "NO TRADE",
                    "reason": "stale_data" if age_hours > max_age_hours else "latest_candle_not_closed",
                }

            # Display the latest actual strategy transition, regardless of age.
            # This is presentation only and does not alter the current decision.
            direction = strategy.raw_direction(df)
            signal_positions = [
                position for position, value in enumerate(direction.tolist())
                if int(value) in {-1, 1}
            ]
            recent_signal_text = "آخرین سیگنال در داده‌های دریافت‌شده پیدا نشد."
            if signal_positions:
                signal_position = signal_positions[-1]
                signal_side = int(direction.iloc[signal_position])
                signal_label = "BUY" if signal_side == 1 else "SELL"
                signal_open = pd.Timestamp(df.index[signal_position])
                if signal_open.tzinfo is None:
                    signal_open = signal_open.tz_localize("UTC")
                signal_closed = signal_open + duration
                signal_age = max(0, int((now - signal_closed).total_seconds()))
                signal_local_time = signal_closed.to_pydatetime().astimezone()
                if not fresh:
                    signal_status = "تاریخی · داده فعلی کهنه است"
                elif signal_position == len(df) - 1:
                    signal_status = "NEW · سیگنال جدید، در انتظار کندل بعدی"
                elif (
                    signal_position == len(df) - 2
                    and decision.get("decision", {}).get("label") == signal_label
                ):
                    signal_status = "LIVE · سیگنال در آخرین تصمیم"
                else:
                    signal_status = "سیگنال قبلی"
                if signal_age < 3600:
                    age_text = f"{signal_age // 60} دقیقه قبل"
                else:
                    age_text = f"{signal_age // 3600} ساعت و {(signal_age % 3600) // 60} دقیقه قبل"
                recent_signal_text = (
                    f"آخرین سیگنال استراتژی: {signal_label} | وضعیت: {signal_status}\\n"
                    f"زمان کندل سیگنال (محلی): {signal_local_time.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                    f"زمان UTC: {signal_closed.isoformat()} | {age_text}"
                )

            chart = build_chart_html(
                df, market=market, timeframe=timeframe,
                strategy_name=strategy_name, strategy_params={},
                strategy=strategy, latest_decision=decision,
            )
            label = decision.get("decision", {}).get("label", "NO TRADE")
            source = df.attrs.get("source", "public market data")
            retrieval = df.attrs.get("retrieval_time_utc", "نامشخص")
            detail = decision.get("decision", {})
            reason = detail.get("reason")
            levels = ""
            if label in {"BUY", "SELL"}:
                levels = (
                    f"\nورود: {detail.get('entry')} | حد ضرر: {detail.get('stop')} | "
                    f"هدف: {detail.get('target')}"
                )
            elif reason:
                levels = f"\nعلت: {reason}"
            message = (
                f"تصمیم فعلی: {label} | نماد: {market} | تایم‌فریم: {timeframe} | "
                f"استراتژی: {strategy_name}{levels}\n"
                f"{recent_signal_text}\\n"
                f"وضعیت داده: {'تازه' if fresh else 'کهنه'} | منبع: {source} | زمان دریافت: {retrieval} | کندل‌های بسته: {len(df)}"
            )
            self._send(_page(
                message, chart, market=market, timeframe=timeframe,
                strategy=strategy_name, days=days,
            ))
        except Exception as exc:
            self._send(_page(f"خطا: {exc}", market=market, timeframe=timeframe, strategy=strategy_name), status=400)

    def _send(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            # The browser may cancel a slow request while market data is unavailable.
            return

    def log_message(self, _format: str, *_args) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local unified signal dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Local interface; do not expose publicly")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("The dashboard is local-only; --host must be 127.0.0.1 or localhost.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Signal dashboard: {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
