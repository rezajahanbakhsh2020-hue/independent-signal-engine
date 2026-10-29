# Independent Signal Engine

A standalone, deterministic Python 3.11+ research and evaluation engine for
multi-market trading-signal research on:

- **Markets**: `XAUUSD`, `BTCUSDT`, `ETHUSDT` (exact identifiers)
- **Timeframes**: `1m`, `5m`, `15m`, `30m`, `1H`, `4H`, `1D` (the only accepted spellings)
- **Matrix**: exactly 21 market/timeframe combinations. Unavailable
  combinations are reported explicitly, never silently omitted.

> **This is research software.** Nothing here executes live trades, nothing
> here guarantees profitability, and simulated results are never presented as
> live performance.

---

## 1. Install

```bash
pip install -e ".[dev]"
```

Runtime dependencies are intentionally minimal: `pandas`, `numpy`.
Dev extra adds `pytest`. No credentials, keys, or private services are
required or consulted anywhere in this codebase.

## 2. Run tests

```bash
python -m pytest
```

All tests are deterministic (fixed seeds, synthetic fixtures).

## 3. Validate a CSV dataset

```bash
signal-engine validate-csv --csv data/BTCUSDT_1H.csv --expected-spacing 1H
```

Produces an audit record: rows seen, rows rejected and *why* (NaN, infinity,
negative volume, invalid OHLC relationships, malformed rows/timestamps),
duplicates removed, unsorted data repaired, documented gaps (never filled
or invented), normalized UTC coverage.

## 4. Run a backtest

```bash
signal-engine backtest --csv data/BTCUSDT_1H.csv \
  --market BTCUSDT --timeframe 1H \
  --strategy crossover --param fast=10 --param slow=30 \
  --fee-bps 2 --spread-bps 2 --slippage-bps 1
```

Strategies: `crossover` (SMA/EMA), `pullback`, `breakout`,
`bollinger_mean_reversion`. Parameters are passed via `--param key=value`.

## 5. Run walk-forward evaluation

```bash
signal-engine walkforward --csv data/BTCUSDT_1H.csv \
  --market BTCUSDT --timeframe 1H \
  --strategy crossover --param fast=5 --param slow=20 \
  --folds 3 --train-candles 250 --eval-candles 100 --holdout-fraction 0.2
```

Folds are disjoint and strictly chronological. The final 20% (configurable)
is a single **untouched holdout**: it is never used for training,
validation, or parameter selection, and is evaluated exactly once after all
tuning decisions. Insufficient data **fails closed** with
`WalkForwardError`; windows never silently shrink. Every boundary is
recorded as exact UTC timestamps plus candle counts, and an independent
leakage/overlap check re-verifies non-contamination.

## 6. Run the statistical baseline

```bash
signal-engine baseline --csv data/BTCUSDT_1H.csv \
  --market BTCUSDT --timeframe 1H \
  --strategy crossover --param fast=5 --param slow=20 \
  --m 999 --seed 20260101
```

### Test design

| Property | Value |
|---|---|
| Observing statistic | **net total P&L** (final equity − initial equity) over the identical eval window |
| Null model | identical dataset, timestamps, risk geometry and costs; same *entry-opportunity timing* as the real strategy, but each entry side drawn i.i.d. uniform from {+1, −1} using `numpy.random.default_rng(seed)`; destroys only the directional information |
| Simulations | `M` (caller-chosen, ≥ 1) |
| p-value formula | `p = (1 + count(random_stat >= observed_stat)) / (M + 1)` (empirical, one-sided) |
| Seed | reproducible via `--seed` |

### Reported limitations (also printed in every JSON result)

- The null model is **not** a universal or fully realistic market null: it
  does not model regime persistence, volatility clustering, or realistic
  microstructure.
- If the observed run produces **zero trades**, no p-value is produced
  (fail-closed) — the note `p-value withheld` is emitted instead.
- Per-configuration p-values do **not** correct for multiple testing. When
  `K` configurations are tested, apply a Bonferroni-style correction (e.g. treat
  `p_min × K` as a family-wise-error bound) or interpret raw p-values with
  selection-bias caution.
- Any z-score-like numbers are **descriptive statistics only**, never
  standalone proof of an edge.

## 7. Produce a report for one market/timeframe

```bash
signal-engine report --csv data/BTCUSDT_1H.csv \
  --market BTCUSDT --timeframe 1H \
  --strategy crossover --param fast=40 --param slow=240 \
  --baseline-m 99 --seed 20260101
```

Adds walk-forward (disable with `--no-walkforward`) and the randomized
baseline (disable with `--no-baseline`) to a single JSON cell with full
provenance. With `--synthetic`, a clearly-labeled deterministic synthetic
dataset (`SYNTHETIC`) is used instead of a CSV for demos/tests only.

## 8. Produce a matrix report for all 21 combinations

```bash
# from per-cell CSV files named {MARKET}_{TIMEFRAME}.csv in one directory:
signal-engine matrix --csv-dir data/csv_cells/BTCUSDT \
  --strategy crossover --param fast=5 --param slow=20 \
  --baseline-m 99 --seed 20260101

# or over synthetic data (explicitly labeled SYNTHETIC; demo only):
signal-engine matrix --synthetic --strategy breakout --no-walkforward
```

The output contains exactly `n_cells_expected = 21` entries. Cells whose
data file is missing appear with status `"UNAVAILABLE_DATA"` (e.g.
`missing_data_file:...`), and cells whose pipeline hits an error appear
with status `"ERROR"` plus the error string. Nothing is dropped or hidden.

## 9. Latest-decision signal check (current-data verification)

```bash
signal-engine signal --market XAUUSD --timeframe 1H \
  --csv XAUUSD_1H.csv \
  --strategy crossover --param fast=20 --param slow=60 \
  --initial-equity 10000
```

Computes the most recent decision the engine's uniform execution semantics
would have acted on from genuinely recent closed candles (fetch them with
`fetch-crypto`/`fetch-gold` first). By default, the latest candle must be closed
and no older than three candle durations; stale or still-open candles return
`NO TRADE` with a freshness reason. `--max-age-hours N` explicitly overrides
the freshness limit when intentionally researching historical data. Synthetic
fixtures skip freshness checks and remain clearly labeled. Semantics mirror the backtester exactly:

- the decision acted at candle *t* derives from the strategy direction at
  *t-1* (evidence candle),
- the latest acted decision therefore uses the **penultimate** candle as the
  signal candle and the **last** candle's open as the execution reference
  with validated ATR-based stop/target geometry and position sizing,
- the strategy direction on the final candle is reported separately as
  `pending_raw_direction` — it would only be acted upon at a candle that
  does not exist yet, so it can never carry an entry/stop/target.

Labels: `BUY`, `SELL`, `NO TRADE` (with an explicit reason such as
`no_active_transition_signal`, `risk_geometry_rejected_zero_or_insufficient_atr`,
or `insufficient_candles:N`). If the CSV fails validation the command emits
`ok:false` plus the audit report and exits 1 — fail-closed, never fabricated.
`--synthetic` uses a deterministic fixture labeled
`SYNTHETIC test/demo-only fixture; not market data`. The output is a research
verification aid only; no profitability is implied.

## 10. Fetch public crypto data (BTCUSDT / ETHUSDT only)

```bash
signal-engine fetch-crypto --market BTCUSDT --timeframe 1H \
  --start 2024-01-01T00:00 --end 2024-03-01T00:00 --out BTCUSDT_1H.csv
```

Uses official public Binance v3 klines through stdlib `urllib`, trying
`api.binance.com` first and the official public mirror
`data-api.binance.vision` if the primary host is unreachable or blocked
(no credentials). It respects the 1000-kline paging limit and inserts a
small sleep between pages. Provenance records the actual host used, retrieval
time (UTC), symbol mapping, timezone, and full coverage. A host change during
paging aborts the request rather than mixing sources. If both official hosts
fail, a `BinanceUnavailableError` is raised — **no fake fallback data is ever
substituted**.


For gold (XAUUSD), use the public BiQuote OHLC endpoint (no API key):

```bash
signal-engine fetch-gold --timeframe 1H \
  --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z \
  --out XAUUSD_1H.csv
```

The adapter requests only XAUUSD, excludes the still-open candle, pages
backward within the provider's 1,000-bar request limit, and records requested
range, actual coverage, validation gaps, retrieval time, and source provenance.
BiQuote may report zero traded volume for FX/CFD bars; the adapter uses
`tickVolume` as a tick-count proxy and explicitly labels it as **not
exchange-traded volume**. This is not mapped to BTCUSDT/ETHUSDT: BTCUSD and
BTCUSDT are different instruments and are never silently substituted.

---
 
## 11. CSV schema

| Column | Meaning | Notes |
|---|---|---|
| `timestamp` (or `time`, `date`, `datetime`) | candle **open** time | naive values are treated as UTC (documented assumption); tz-aware values are converted to UTC |
| `open`, `high`, `low`, `close` | prices | must be finite strictly-positive numerics |
| `volume` | traded volume | must be ≥ 0; finite |
| index | optional DatetimeIndex | used when no timestamp column is present |

Validation **rejects and documents** (never silently repairs/drops):
NaN / infinity values, malformed numeric or timestamp values, invalid
OHLC relationships (`low > high`, `high < max(open, close)`,
`low > min(open, close)`), negative volume, duplicate timestamps (first
kept, count recorded), unsorted data (re-sorted, count recorded).
**Gaps** are detected against the dataset's dominant candle spacing and
reported in `validation.gaps` — they are never filled and no candles are
invented.

## 12. Signal, decision and execution semantics

For a signal computed from a candle that opened at time `t`:

1. `signal_ts = t` — the completed candle's open time (evidence candle).
2. `decision_ts = t + <delta of the candle size>` — when the decision is
   acted upon (the next candle's open).
3. `execution` occurs at that next candle's **open**, on the worst side:
   long entry at `open × (1 + (spread/2 + slippage))`, short entry at
   `open × (1 − (spread/2 + slippage))`.
4. Stops and targets are evaluated against each candle's high/low only
   **after** the entry fill; if both are hit in the same candle, the stop
   fills first (conservative). Exit fills include one spread side, plus
   slippage on stop exits.
5. Fees are a percentage of notional charged at **every** fill and
   deducted from equity immediately.
6. At most one open position at a time (no overlapping trades).
7. A pending position at the end of data is force-closed at the final
   candle's close with documented costs (`end_of_data` exit reason).

`NO TRADE` is an explicit, acceptable output. Signals rejected because of
zero ATR, insufficient warm-up, or incoherent geometry are *counted*
(`n_signals_rejected_zero_atr`, `n_signals_rejected_incoherent`) rather
than hidden.

## 13. Risk geometry

`stop_distance = atr(n) × atr_multiplier`, `target = entry ± reward_risk × stop_distance`,
position size = `(equity × risk_fraction) / stop_distance`, capped so
that `qty × entry ≤ max_position_fraction × equity`.

Directional coherence is enforced and fail-closed:

- **BUY**: `stop < entry < target`
- **SELL**: `target < entry < stop`
- All four levels must be finite; violation produces no signal at all.

## 14. Concentration metric (documented definition)

Computed via **HHI (Herfindahl–Hirschman Index)** over the dimension
defaults to **trades**, on **net realized monetary P&L per trade**:

```
positive_pnl = Σ net_pnl[i] where net_pnl[i] > 0
if positive_pnl ≤ 0 or no positive trades  → concentration = None  (no misleading %)
else concentration = Σ (net_pnl[i] / positive_pnl)² for net_pnl[i] > 0
```

- Denominator: the sum of positive net trade P&L.
- Losses do **not** share the denominator.
- Zero/negative denominators produce `None`, not a fabricated percentage.
- The evaluator can change the dimension (e.g. strategies) by applying the
  same function `hhi_concentration` to that dimension's net P&L values;
  the dimension used is stated wherever concentration is reported.

## 15. Completion statuses

| Status | Meaning |
|---|---|
| `COMPLETE` | implemented + tested functionality is complete and required evidence for the claimed scope exists |
| `COMPLETE_OFFLINE` | offline functionality is complete and tested; no live-data verification yet established |
| `COMPLETE_OFFLINE_NO_TRADES` | pipeline ran end-to-end but the strategy produced zero trades on this window (still a complete offline run) |
| `PARTIAL` | some required capability or market/timeframe combination is unfinished |
| `UNAVAILABLE_DATA` | the cell has no data source available (matrix reports include the exact reason) |
| `ERROR` | the cell's pipeline failed; the exact error is carried in the cell |

## 16. Known limitations and honest-scoping notes

- Data sources: user-supplied OHLCV CSVs for any supported market; the public
  Binance API/mirror for BTCUSDT and ETHUSDT; and the public BiQuote OHLC API
  for XAUUSD. Provider coverage differs: BiQuote did not provide 1m XAUUSD bars
  in the verified run and returned too few intraday candles for the configured
  walk-forward minimum. The engine reports those cells as unavailable or
  insufficient instead of inventing candles. Additional XAUUSD 1m history may
  be supplied via CSV or a separately verified provider.
- Simulated `end_of_data` close-outs, `stop-first` filling on same-candle
  stop/target collisions, and bid/ask approximations via half-spread are
  conservative but simplified; they cannot be claimed to exactly reproduce
  live fills.
- Walk-forward here evaluates **fixed parameter sets** across folds; the
  holdout is guaranteed untouched regardless.
- Statistical evidence is conditioned on sample size; `p ≈ 0` with `M` very
  small is weak evidence by construction (see section 6 notes).

## 17. Security

No secrets, keys, or tokens are read, written, or required. The adapter
uses public unauthenticated endpoints only. Issue content and any external
context are treated as untrusted; they cannot override data/boundary
policies of this engine.

## 18. Layout

```
signal_engine/
  constants.py          markets, timeframes, 21-cell matrix
  indicators.py         SMA/EMA/ATR/Bollinger (causal rolling only)
  risk.py               Signal + ATR stop geometry, position sizing
  backtest.py           deterministic backtester with fees/spread/slippage
  metrics.py            performance metrics + HHI concentration
  walkforward.py        walk-forward + holdout + leakage checks
  stats.py              randomized-baseline p-value test
  report.py             cell/matrix reports with provenance + statuses
  cli.py                argparse CLI (all commands above)
  data/validate.py      strict OHLCV validation
  data/csv_source.py    CSV loader, resampling (coarser-only)
  data/binance_source.py public Binance klines adapter (BTC/ETH only)
  data/biquote_source.py public XAUUSD OHLC adapter (tick-volume semantics)
  latest_signal.py      latest decision and freshness gate
  data/synthetic.py     deterministic synthetic OHLCV (tests/demos, labeled)
tests/                  200+ focused tests incl. lookahead/leakage regressions
```
