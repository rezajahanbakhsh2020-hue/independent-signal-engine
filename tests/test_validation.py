"""Data validation tests: all rejection paths are explicit and audited."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.data.validate import DataValidationError


def _base(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 105, n),
            "high": np.linspace(101, 106, n),
            "low": np.linspace(99, 104, n),
            "close": np.linspace(100.5, 105.5, n),
            "volume": np.linspace(10, 50, n),
        },
        index=idx,
    )


def test_happy_path_clean_data():
    from signal_engine.data.validate import validate_ohlcv

    df, rep = validate_ohlcv(_base())
    assert rep.n_rows_out == rep.n_rows_in
    assert rep.n_rejected_rows == 0
    assert df.index.tz is not None


def test_nan_in_close_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    out, rep = validate_ohlcv(df)
    assert rep.n_rejected_rows == 1
    assert rep.rejection_reasons.get("non_finite_or_missing_value") == 1
    assert len(out) == 49


def test_infinity_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.iloc[3, df.columns.get_loc("high")] = np.inf
    out, rep = validate_ohlcv(df)
    assert rep.rejection_reasons.get("non_finite_or_missing_value") == 1
    assert len(out) == 49


def test_duplicate_timestamps_removed_and_counted():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    dup = df.iloc[[10]]  # duplicate timestamp row (slightly different values)
    dup.iloc[0, 0] = 999.0
    raw = pd.concat([df, dup]).sort_index()
    out, rep = validate_ohlcv(raw)
    assert rep.n_duplicate_timestamps_removed == 1
    assert out.index.is_unique


def test_invalid_ohlc_relationship_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.iloc[7, df.columns.get_loc("high")] = 90.0  # below open/close
    out, rep = validate_ohlcv(df)
    assert rep.rejection_reasons.get("invalid_ohlc_relationship") == 1
    assert len(out) == 49


def test_negative_volume_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.iloc[2, df.columns.get_loc("volume")] = -5.0
    out, rep = validate_ohlcv(df)
    assert rep.rejection_reasons.get("negative_volume") == 1
    assert (out["volume"] >= 0).all()


def test_malformed_row_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base().reset_index(names="timestamp")
    df["open"] = df["open"].astype(object)
    df.loc[11, "open"] = "not-a-number"
    out, rep = validate_ohlcv(df)
    assert rep.rejection_reasons.get("malformed_open") == 1


def test_unsorted_data_repaired_and_flagged():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    swapped = pd.concat([df.iloc[:3], df.iloc[-4:], df.iloc[3:-4]])
    out, rep = validate_ohlcv(swapped)
    assert rep.n_unsorted_repaired > 0
    assert out.index.is_monotonic_increasing


def test_missing_required_column_raises():
    from signal_engine.data.validate import validate_ohlcv

    df = _base().drop(columns=["volume"])
    with pytest.raises(DataValidationError, match="volume"):
        validate_ohlcv(df)


def test_gaps_documented_not_filled():
    from signal_engine.data.validate import validate_ohlcv

    df = _base(30)
    keep = list(range(0, 10)) + list(range(20, 30))  # remove 10 candles
    out, rep = validate_ohlcv(df.iloc[keep])
    assert len(rep.gaps) == 1
    assert rep.gaps[0]["missing_candles_estimated"] == 10
    # no invented candles: output rows == input rows
    assert len(out) == 20


def test_naive_timestamps_localized_to_utc():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.index = df.index.tz_localize(None)
    out, _ = validate_ohlcv(df)
    assert out.index.tz is not None and str(out.index.tz) == "UTC"


def test_malformed_timestamp_row_rejected():
    from signal_engine.data.validate import validate_ohlcv

    df = _base().reset_index(names="timestamp")
    df["timestamp"] = df["timestamp"].astype(object)
    df.loc[9, "timestamp"] = "garbage"
    out, rep = validate_ohlcv(df)
    assert rep.rejection_reasons.get("malformed_timestamp") == 1


def test_tz_aware_timestamps_converted_to_utc():
    from signal_engine.data.validate import validate_ohlcv

    df = _base()
    df.index = df.index.tz_convert("America/New_York")
    out, _ = validate_ohlcv(df)
    assert str(out.index.tz) == "UTC"
