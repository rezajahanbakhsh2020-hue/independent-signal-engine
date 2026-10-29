"""CLI and matrix integration tests."""
from __future__ import annotations

import json

import pytest

from signal_engine.cli import main
from signal_engine.data.synthetic import synth_ohlcv


@pytest.fixture
def csv_dir(tmp_path):
    d = tmp_path / "csv"
    d.mkdir()
    for tf in ("1m", "5m", "15m", "30m", "1H", "4H", "1D"):
        n = 600 if tf != "1D" else 400
        df = synth_ohlcv(tf, "2024-01-01", n, seed=42)
        df.reset_index(names="timestamp").to_csv(d / f"BTCUSDT_{tf}.csv", index=False)
    return str(d)


def test_cli_validate_csv(csv_dir, capsys):
    rc = main(["validate-csv", "--csv", f"{csv_dir}/BTCUSDT_1H.csv"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["validation"]["rows_out"] == 600


def test_cli_backtest(csv_dir, capsys):
    rc = main([
        "backtest", "--csv", f"{csv_dir}/BTCUSDT_1H.csv",
        "--market", "BTCUSDT", "--timeframe", "1H",
        "--strategy", "crossover", "--param", "fast=5", "--param", "slow=20",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["market"] == "BTCUSDT" and out["timeframe"] == "1H"
    assert "final_equity" in out


def test_cli_matrix_all_21_cells(csv_dir, capsys):
    rc = main([
        "matrix", "--csv-dir", csv_dir, "--strategy", "crossover",
        "--param", "fast=5", "--param", "slow=20",
        "--no-walkforward", "--no-baseline",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_cells_expected"] == 21
    assert out["n_cells_reported"] == 21
    cells = out["cells"]
    markets = {(c["market"], c["timeframe"]) for c in cells}
    assert len(markets) == 21
    # we only provided BTCUSDT files: other markets must be explicit UNAVAILABLE_DATA
    unavailable = [c for c in cells if c["market"] != "BTCUSDT"]
    assert all(c["status"] == "UNAVAILABLE_DATA" for c in unavailable)
    assert len(unavailable) == 14


def test_cli_matrix_missing_files_reported_explicitly(tmp_path, capsys):
    empty_dir = tmp_path / "none"
    empty_dir.mkdir()
    rc = main([
        "matrix", "--csv-dir", str(empty_dir), "--strategy", "crossover",
        "--no-walkforward", "--no-baseline",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_cells_reported"] == 21
    assert all(c["status"] == "UNAVAILABLE_DATA" for c in out["cells"])


def test_cli_report_tiny_data(csv_dir, capsys):
    rc = main([
        "report", "--csv", f"{csv_dir}/BTCUSDT_5m.csv",
        "--market", "BTCUSDT", "--timeframe", "5m",
        "--strategy", "bollinger_mean_reversion",
        "--no-walkforward", "--no-baseline",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    cell = out["cells"][0]
    assert cell["status"] in {"COMPLETE_OFFLINE", "COMPLETE_OFFLINE_NO_TRADES"}
    assert cell["metrics"]["n_trades"] >= 0


def test_cli_walkforward_and_baseline(csv_dir, capsys):
    rc = main([
        "walkforward", "--csv", f"{csv_dir}/BTCUSDT_1H.csv",
        "--market", "BTCUSDT", "--timeframe", "1H",
        "--strategy", "crossover", "--param", "fast=5", "--param", "slow=20",
        "--folds", "2", "--train-candles", "150", "--eval-candles", "50",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["leakage_check_passed"] is True

    rc = main([
        "baseline", "--csv", f"{csv_dir}/BTCUSDT_1H.csv",
        "--market", "BTCUSDT", "--timeframe", "1H",
        "--strategy", "crossover", "--param", "fast=5", "--param", "slow=20",
        "--m", "5", "--seed", "99",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["p_value"] * (5 + 1) - (1 + round(out["p_value"] * 6 - 1))) < 1e-9


def test_cli_rejects_unknown_market(csv_dir):
    with pytest.raises(SystemExit):
        main([
            "backtest", "--csv", f"{csv_dir}/BTCUSDT_1H.csv",
            "--market", "DOGEUSDT", "--timeframe", "1H", "--strategy", "crossover",
        ])


def test_cli_rejects_unknown_timeframe(csv_dir):
    with pytest.raises(SystemExit):
        main([
            "backtest", "--csv", f"{csv_dir}/BTCUSDT_1H.csv",
            "--market", "BTCUSDT", "--timeframe", "2H", "--strategy", "crossover",
        ])


def test_cli_report_synthetic_is_labeled(tmp_path, capsys):
    rc = main([
        "report", "--synthetic", "--market", "XAUUSD", "--timeframe", "15m",
        "--strategy", "breakout", "--no-walkforward", "--no-baseline",
        "--synthetic-n", "400",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    cell = out["cells"][0]
    assert cell["data_provenance"] == "SYNTHETIC"
