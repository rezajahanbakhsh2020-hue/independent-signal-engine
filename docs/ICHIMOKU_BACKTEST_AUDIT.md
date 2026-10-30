# Ichimoku Backtest Audit — 2026-10-30

## Scope and evidence

- Audited implementation commit: `2e18c19504241e17e1c6200438256e319cf3fb48`
- Audit branch: `audit/ichimoku-20261030`
- Prior GitHub Actions run: [40930825576](https://github.com/rezajahanbakhsh2020-hue/independent-signal-engine/actions/runs/40930825576)
- Artifact: `ichimoku-crypto-backtest`, artifact id `12922456976`, SHA-256 `7a956b0afdd190e8006617baca49d63031dbb9f4ee41e51599ee5dc1e1a1eef5`
- Run conclusion: workflow/job succeeded and all three backtest scripts executed. This establishes execution success, not correctness or completion of the requested experiment.

## Verified prior results

The four-market RSI-confirmed 1H run used 1,440 closed candles per market from 2026-08-30 18:00 UTC through 2026-10-29 17:00 UTC, retrieved from the official Binance mirror.

| Market | Net return | Trades | Profit factor | Max drawdown |
|---|---:|---:|---:|---:|
| BTCUSDT | +4.333787% | 9 | 2.080098 | -3.041758% |
| ETHUSDT | -1.482326% | 4 | 0.535220 | -2.106230% |
| SOLUSDT | -3.051137% | 6 | 0.386146 | -4.600490% |
| ADAUSDT | +2.782753% | 6 | 1.901000 | -2.649661% |

All four rows are explicitly marked `insufficient_samples=true`. These are descriptive, in-sample observations only.

The BTC 1H structural-stop script produced 16 configurations, not the requested complete 17-cell experiment across both 15m and 1H. It produced only 1H results. The ADX-direction-only cell had 3 trades, +2.258887% net return, PF 3.100755, and -1.751612% max drawdown; its sample is insufficient and it is not evidence of a robust edge.

## Confirmed gaps / defects

1. `scripts/backtest_btc_ichimoku_confirmations.py` defines `IchimokuCombo.stop_for_signal()`, but the shared `signal_engine.backtest.backtest()` path does not call that method. It calls the generic `build_signal()` using `RiskConfig.atr_multiplier=2.0`. Therefore this comparison script's metadata claims the structural stop, but its trades use the generic ATR stop. Do not treat its results as structural-stop results.
2. `scripts/backtest_btc_ichimoku_structural_stop.py` is hard-coded to BTCUSDT / 1H and tests 16 combinations from five filters (0, 1, or 2 filters). It does not test the requested 15m timeframe, HTF EMA filter, separate ADX-DI thresholds 20 and 25, or the full requested combination matrix.
3. The structural-stop script reports aggregate counts/metrics but does not provide the required per-signal filter pass/reject diagnostics, stop-distance distribution, and same-opportunity ADX/+DI/-DI comparison.
4. No dedicated tests for the structural-stop state machine, stop-anchor indexing, HTF candle-close ordering, or same-bar stop/target conflict were found at the audited commit.
5. The current branch implementation's successful CI run is not a substitute for those missing tests. The workflow's green status only says the scripts exited successfully and uploaded artifacts.

## HTF event-order rule for follow-up implementation

For a base-candle signal timestamp equal to an HTF candle close timestamp, that HTF candle is not eligible unless the event ordering explicitly establishes that the HTF close is processed first. Default to the previous fully closed HTF candle; enforce strict earlier-than matching (no exact timestamp match) in the merge/alignment logic.

## Change boundary

This audit was performed on a separate branch from the verified implementation SHA. PR #5 and its branch were not modified. No merge, deployment, credentials, or live order action was performed.

## Completion status

`PARTIAL — NOT READY FOR STRATEGY CONCLUSION`

The existing results are preserved as historical evidence. A corrected 15m/1H structural-stop matrix, explicit HTF no-lookahead alignment, complete per-signal diagnostics, and focused regression tests remain necessary before the requested work can be called complete.
