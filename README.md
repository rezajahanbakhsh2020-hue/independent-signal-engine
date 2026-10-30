# Independent Signal Engine

A standalone Python 3.11+ multi-market signal engine. The active strategy allow-list contains **only two selected strategies**:

1. **`Bollinger_Reversion`** — BUY when price crosses back above the lower Bollinger band; SELL when it crosses back below the upper band. Original defaults: period 20, 2.0 standard deviations.
2. **`RSI_Reversal`** — Wilder RSI(14); BUY when RSI crosses upward through 30; SELL when RSI crosses downward through 70. These original settings are fixed.

No other strategy is registered or selectable through the CLI.

## Markets and timeframes

- Gold: `XAUUSD` through the public BiQuote OHLC adapter.
- Crypto: any syntactically valid Binance USDT spot pair (for example `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `ADAUSDT`). Binance is authoritative for actual symbol availability; invalid or delisted pairs fail clearly and are never silently substituted.
- Timeframes: `1m`, `5m`, `15m`, `30m`, `1H`, `4H`, `1D`.

## Install

```bash
python -m pip install -e .
```

## Fetch crypto candles

```bash
signal-engine fetch-crypto --market SOLUSDT --timeframe 15m \
  --start 2026-01-01T00:00 --end 2026-01-02T00:00 --out data/SOLUSDT_15m.csv
```

The adapter uses official public Binance endpoints, respects pagination limits, validates OHLCV data, records provenance, and refuses to mix providers during a paginated request.

## Produce a current signal from a CSV

```bash
signal-engine signal --market SOLUSDT --timeframe 15m \
  --csv data/SOLUSDT_15m.csv --strategy RSI_Reversal
```

For Bollinger:

```bash
signal-engine signal --market SOLUSDT --timeframe 15m \
  --csv data/SOLUSDT_15m.csv --strategy Bollinger_Reversion \
  --param window=20 --param num_std=2.0
```

## Fetch and chart crypto in one command

```bash
signal-engine chart-crypto --market SOLUSDT --timeframe 15m \
  --strategy Bollinger_Reversion --param window=20 --param num_std=2.0 \
  --out charts/SOLUSDT_15m.html
```

The standalone HTML chart contains candlesticks, historical BUY/SELL markers, the latest decision, and data/decision details. Markers are placed at the next candle open where a decision would be acted on; they are not live orders.

## Data integrity and freshness

- CSV and fetched candles are validated; malformed OHLCV data is rejected with an audit report.
- Current-signal and chart paths exclude unfinished candles and reject stale data.
- Data source, retrieval time, symbol, timeframe, and coverage are recorded where available.
- A valid `NO TRADE` is reported honestly. No entry, stop, or target is fabricated when the evidence or risk geometry is insufficient.

## Scope boundaries

This repository is a signal/research tool, not an order-execution system. It does not guarantee profitability.

The automated 16-combination crypto comparison, Ichimoku strategy/backtests, Ichimoku confirmation grids, and their dedicated GitHub Actions workflows are excluded. No backtest or test suite was run as part of this PR correction.
