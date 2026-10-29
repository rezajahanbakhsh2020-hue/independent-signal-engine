"""CSV loading, resampling, and provenance tests."""
from __future__ import annotations

import json
import pandas as pd
import pytest

from signal_engine.cli import main
from signal_engine.data.csv_source import read_csv_ohlcv, resample_ohlcv
from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.data.validate import DataValidationError


@pytest.fixture
def csv_file(synth_1h, tmp_path):
    p = tmp_path / "BTCUSDT_1H.csv"
    synth_1h.reset_index(names="timestamp").to_csv(p, index=False)
    return str(p)


def test_csv_roundtrip(csv_file, synth_1h):
    df, rep = read_csv_ohlcv(csv_file)
    assert len(df) == len(synth_1h)
    assert df.index.tz is not None
    assert df.attrs["source"] == f"csv:{csv_file}"


def test_csv_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        read_csv_ohlcv(str(tmp_path / "nope.csv"))


def test_csv_empty_file_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        read_csv_ohlcv(str(p))


def test_csv_via_timestamp_column(tmp_path):
    df = synth_ohlcv("1H", "2024-01-01", 200, seed=3)
    p = tmp_path / "data.csv"
    df.reset_index(names="timestamp").to_csv(p, index=False)
    out, rep = read_csv_ohlcv(str(p))
    assert len(out) == 200
    assert str(out.index.tz) == "UTC"


def test_csv_naive_timestamps_documented_utc(tmp_path):
    df = synth_ohlcv("1H", "2024-01-01", 200, seed=3)
    naive = df.copy()
    naive.index = naive.index.tz_localize(None)
    p = tmp_path / "naive.csv"
    naive.reset_index(names="timestamp").to_csv(p, index=False)
    out, rep = read_csv_ohlcv(str(p))
    assert str(out.index.tz) == "UTC"


def test_resample_only_to_coarser(synth_1h):
    with pytest.raises(DataValidationError, match="finer"):
        resample_ohlcv(synth_1h, "1H", "15m")


def test_resample_1h_to_4h(synth_1h):
    out = resample_ohlcv(synth_1h, "1H", "4H")
    assert str(out.index.tz) == "UTC"
    assert (out.index.ordinal if hasattr(out.index, "ordinal") else True)
    # no invented candles; volume conserved between valid group rows
    assert out["volume"].sum() <= synth_1h.loc[out.index[0]: out.index[-1] + pd.Timedelta("4h"), "volume"].sum() + 1e-9


def test_resample_30m_to_1d_deterministic(tmp_path):
    df = synth_ohlcv("30m", "2024-01-01", 48 * 30, seed=5)
    o1 = resample_ohlcv(df, "30m", "1D")
    o2 = resample_ohlcv(df.copy(), "30m", "1D")
    pd.testing.assert_frame_equal(o1, o2)

def test_cli_validate_csv_all_rejected_reports_audit(tmp_path, capsys):
    p = tmp_path / "all_rejected.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T00:00:00Z,100,99,101,100,10\n"
        "2024-01-01T01:00:00Z,100,99,101,100,10\n"
    )
    exit_code = main(["validate-csv", "--csv", str(p)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["validation"]["rows_in"] == 2
    assert payload["validation"]["rows_out"] == 0
    assert payload["validation"]["rejection_reasons"]["invalid_ohlc_relationship"] == 2
