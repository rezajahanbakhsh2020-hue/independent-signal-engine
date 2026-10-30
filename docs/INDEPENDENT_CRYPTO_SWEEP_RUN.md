# Independent crypto strategy sweep

- Test window: rolling two calendar months ending at the latest fully closed UTC hour.
- Data: Binance public spot klines; BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT.
- Timeframes: 15m and 1h.
- Strategies: Donchian 20 breakout; EMA20/EMA50 pullback; Bollinger(20,2) mean reversion; RSI(2) < 10 mean reversion; MACD(12,26,9) with EMA200 trend; RSI(14) < 30 reversal.
- Holding caps: 8 and 24 bars, fixed before the test.
- Execution: signal on completed close; entry/exit at next open; one long-only spot position at a time; no leverage.
- Costs: taker fee 10 bps per side, spread 2 bps round trip, slippage 2 bps per side.
- Missing intervals fail closed. No synthetic or interpolated candles.
- Results are a screening sample. A candidate needs at least 100 trades and a positive net expectancy/PF after costs before it can be considered for chronological holdout validation.
- No live trading, exchange credentials, deployment, or merge is authorized by this experiment.
