# Independent Signal Engine

A small, single-purpose market signal application. Install it once, start one local dashboard, choose a market, timeframe and one of the **two fixed strategies**, then see the latest decision and candlestick chart on the same page.

## Active strategies — exactly two

- `Bollinger_Reversion`: original Bollinger re-entry rules, period 20 and 2.0 standard deviations.
- `RSI_Reversal`: original Wilder RSI(14) rules, BUY above 30 after an oversold reading and SELL below 70 after an overbought reading.

No other strategy is registered or selectable. The dashboard uses the original strategy defaults and exposes no parameter controls.

## Supported markets and timeframes

- Gold: `XAUUSD`, using the independent public BiQuote OHLC adapter.
- Crypto: valid USDT spot pairs such as `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `ADAUSDT`. Each configured public provider must confirm and return the requested pair; symbols are never silently substituted.
- Timeframes: `1m`, `5m`, `15m`, `30m`, `1H`, `4H`, `1D`.

## Start the application

Python 3.11 or newer is required.

```bash
python -m pip install -e .
signal-dashboard
```

The dashboard opens at `http://127.0.0.1:8765`. If the browser does not open automatically, enter that address manually. Stop the local server with `Ctrl+C`.

Choose the market, timeframe, strategy and data lookback (1–90 days), then press **شروع پایش آنلاین** once. The page polls automatically every 10 seconds while open. It loads the selected historical range once, then uses a one-day rolling range for updates. Signal events from both approved strategies are deduplicated and stored locally in `~/.signal-engine/signal-history.sqlite3`; each BUY/SELL appears in its own card with market, timeframe, strategy, signal time, age and candle close price, newest first. The current decision remains separate from the historical event feed.

The dashboard binds only to localhost. It does not expose a public server, request credentials, store secrets, or place orders.

## Data and signal safeguards

- Only completed candles are considered.
- Stale data, invalid OHLC data, insufficient history and invalid risk geometry fail closed; they do not create fabricated entries, stops or targets.
- Crypto OHLCV failover priority is KuCoin → Bitget → HTX → MEXC. Binance is not configured as a provider. Each source must independently return the complete requested candle grid with no gaps or duplicate timestamps; candles from different exchanges are never mixed. Every individual HTTP request has a 2.5-second timeout. HTX's limited recent-history endpoint is accepted only when it covers the entire requested range. XAUUSD remains on its separate BiQuote adapter.
- BiQuote `tickVolume` is a tick count, not centralized exchange-traded volume.
- Historical markers are research signals, not executed orders. A valid `NO TRADE` is an acceptable result.
- No profitability is guaranteed.

## Scope

The dashboard is the simple, primary user workflow. Existing research modules and their CLI commands remain separate optional developer functionality; they are not part of the dashboard flow. The automated 16-combination strategy comparison, Ichimoku strategy/backtests, Ichimoku confirmation grids and their dedicated workflows are excluded. No strategy beyond the two listed above is authorized.

## Validation status

The unified dashboard change has been reviewed statically only. Tests, backtests, simulations and live-market verification were intentionally not run under the current repository-owner instruction.
