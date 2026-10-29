"""CSV data source (primary offline path) and resampling utilities."""

from __future__ import annotations

import pandas as pd

from signal_engine.constants import TIMEFRAMES, TIMEFRAME_TO_PANDAS, UnsupportedTimeframeError
from signal_engine.data.validate import (
    DataValidationError,
    ValidationReport,
    dominant_spacing,
    validate_ohlcv,
)


def read_csv_ohlcv(
    path: str,
    expected_spacing: str | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load and validate an OHLCV CSV file.

    Accepts a timestamp column named one of (timestamp, time, date,
    datetime) or a DatetimeIndex, plus open/high/low/close/volume columns.
    Timestamps may be naive (assumed UTC, document this in your data notes)
    or tz-aware (normalized to UTC).
    """
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(f"CSV file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"CSV file is empty: {path}") from exc
    if raw.empty:
        raise DataValidationError(f"CSV file is empty: {path}")
    df, rep = validate_ohlcv(raw)
    if expected_spacing is not None:
        spacing = dominant_spacing(df.index)
        want = pd.Timedelta(TIMEFRAME_TO_PANDAS[validate_timeframe(expected_spacing)])
        if spacing != want:
            raise DataValidationError(
                f"Expected {expected_spacing} candle spacing {want} but dominant spacing is {spacing}"
            )
    df.attrs["source"] = f"csv:{path}"
    df.attrs["load_time_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    return df, rep


def validate_timeframe(timeframe: str) -> str:
    tf = str(timeframe).strip()
    if tf not in TIMEFRAMES:
        raise UnsupportedTimeframeError(
            f"Unsupported timeframe {timeframe!r}; supported: {', '.join(TIMEFRAMES)}"
        )
    return tf


def resample_ohlcv(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV to a **coarser** canonical timeframe.

    Deterministic, boundary-safe: candles are grouped by left-closed
    windows labeled with the window *open* time in UTC.  We never invent
    candles: an output candle holds an open time only if at least one
    source candle exists inside it.  Resampling to a finer timeframe than
    the source is not supported (it would require fabricating data).
    """
    src = validate_timeframe(source_tf)
    tgt = validate_timeframe(target_tf)
    src_td = pd.Timedelta(TIMEFRAME_TO_PANDAS[src])
    tgt_td = pd.Timedelta(TIMEFRAME_TO_PANDAS[tgt])
    if tgt_td < src_td:
        raise DataValidationError(
            f"Cannot resample {src} to finer timeframe {tgt}: would fabricate data"
        )
    if tgt_td % src_td != pd.Timedelta(0):
        raise DataValidationError(
            f"Target timeframe {tgt} is not a clean multiple of source {src}"
        )
    agg = df.resample(TIMEFRAME_TO_PANDAS[tgt], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open"])
    agg = agg[agg["volume"] > 0]
    agg.index = agg.index.tz_convert("UTC")
    return agg
