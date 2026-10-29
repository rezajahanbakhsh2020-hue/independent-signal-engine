"""Deterministic synthetic OHLCV generator for tests and CLI demos ONLY.

This data is explicitly labeled 'synthetic' in every artifact it touches.
It is used to write reproducible unit/integration tests and to exercise
the matrix workflow when no real data file is supplied.  It is NEVER to
be presented as real market data, and `matrix` reports distinguish real
from synthetic provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.constants import TIMEFRAME_TO_PANDAS, validate_timeframe


def synth_ohlcv(
    timeframe: str,
    start_utc: pd.Timestamp | str,
    n_candles: int,
    seed: int = 42,
    start_price: float = 100.0,
    yearly_vol: float = 0.25,
    regime_switch_prob: float = 0.02,
) -> pd.DataFrame:
    """Generate n_candles of deterministic OHLC starting at start_utc."""
    tf = validate_timeframe(timeframe)
    td = pd.Timedelta(TIMEFRAME_TO_PANDAS[tf])
    start = pd.Timestamp(start_utc)
    index = pd.date_range(start, periods=n_candles, freq=td, tz="UTC")
    # per-step variance from yearly vol
    seconds_per_year = 365 * 24 * 3600
    step_var = (yearly_vol**2) * (td.total_seconds() / seconds_per_year)
    step_sigma = max(step_var, 1e-12) ** 0.5
    rng = np.random.default_rng(seed)
    drifts = np.zeros(n_candles)
    cur = 0.0
    for i in range(n_candles):
        if rng.random() < regime_switch_prob:
            cur = float(rng.choice([-1.0, 1.0])) * step_sigma * 0.5
        drifts[i] = cur
    log_ret = rng.normal(0.0, step_sigma, n_candles) + drifts
    close = start_price * np.exp(np.cumsum(log_ret))
    o = np.empty(n_candles)
    o[0] = start_price
    o[1:] = close[:-1]
    hi = np.maximum(o, close) + np.abs(rng.normal(0, step_sigma, n_candles)) * np.maximum(o, close) * 0.5
    lo = np.minimum(o, close) - np.abs(rng.normal(0, step_sigma, n_candles)) * np.maximum(o, close) * 0.5
    v = rng.integers(1_000, 5_000, n_candles).astype(float)
    df = pd.DataFrame(
        {"open": o, "high": hi, "low": lo, "close": close, "volume": v},
        index=pd.DatetimeIndex(index, name="timestamp"),
    )
    df.attrs["source"] = "synthetic"
    df.attrs["provenance"] = "SYNTHETIC-deterministic-test-data"
    return df
