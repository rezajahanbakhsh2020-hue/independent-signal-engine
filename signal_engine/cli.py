"""Command-line interface for the signal engine.

Subcommands:
- validate-csv    : validate an OHLCV CSV and print the audit report
- backtest        : run a deterministic backtest for one strategy/market/timeframe
- walkforward     : run chronological walk-forward evaluation
- baseline        : run the randomized-baseline statistical test
- report          : produce a full JSON report for one market/timeframe
- matrix          : run the engine across the complete 21-cell matrix
- fetch-crypto    : fetch BTCUSDT/ETHUSDT klines from the public Binance API

All costs expressed in basis points. Provenance and honest statuses are
included in every JSON output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from signal_engine.backtest import backtest, CostConfig
from signal_engine.constants import MARKETS, TIMEFRAMES, validate_market, validate_timeframe
from signal_engine.data.csv_source import read_csv_ohlcv
from signal_engine.data.validate import DataValidationError
from signal_engine.data.synthetic import synth_ohlcv
from signal_engine.report import evaluate_cell, matrix_cells, report_to_json
from signal_engine.risk import RiskConfig
from signal_engine.strategies import STRATEGY_REGISTRY


def _common_opts(p: argparse.ArgumentParser, *, strategy: bool = False) -> None:
    if strategy:
        p.add_argument("--strategy", required=True, choices=sorted(STRATEGY_REGISTRY))
        p.add_argument(
            "--param",
            action="append",
            default=[],
            help="strategy parameter as key=value (repeatable)",
        )
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-multiplier", type=float, default=2.0)
    p.add_argument("--reward-risk", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=2.0)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--initial-equity", type=float, default=10_000.0)


def _risk_and_costs(args) -> tuple[RiskConfig, CostConfig]:
    risk = RiskConfig(
        atr_window=args.atr_window,
        atr_multiplier=args.atr_multiplier,
        reward_risk_multiple=args.reward_risk,
        risk_fraction=args.risk_fraction,
    )
    costs = CostConfig(
        fee_bps=args.fee_bps, spread_bps=args.spread_bps, slippage_bps=args.slippage_bps
    )
    return risk, costs


def _kv_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got: {pair!r}")
        k, v = pair.split("=", 1)
        try:
            val = int(v)
        except ValueError:
            try:
                val = float(v)
            except ValueError:
                val = v
        params[k] = val
    return params


# ----------------------------------------------------------------- commands
def cmd_validate_csv(args) -> int:
    try:
        _df, rep = read_csv_ohlcv(args.csv, expected_spacing=args.expected_spacing)
    except DataValidationError as exc:
        out = {"ok": False, "error": str(exc)}
        if exc.report is not None:
            out["validation"] = exc.report.as_dict()
        print(json.dumps(out, indent=2))
        return 1
    out = {"ok": True, "validation": rep.as_dict()}
    print(json.dumps(out, indent=2))
    return 0


def cmd_backtest(args) -> int:
    market = validate_market(args.market)
    tf = validate_timeframe(args.timeframe)
    df, rep = read_csv_ohlcv(args.csv)
    risk, costs = _risk_and_costs(args)
    params = _kv_params(args.param)
    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    strat = strategy_cls(**params)
    res = backtest(
        df,
        strat,
        market,
        tf,
        risk,
        costs,
        initial_equity=args.initial_equity,
        provenance={"source": f"csv:{args.csv}"},
    )
    out = res.as_dict()
    out["validation"] = rep.as_dict()
    out["n_trades"] = len(res.trades)
    print(json.dumps(out, indent=2))
    return 0


def cmd_walkforward(args) -> int:
    market = validate_market(args.market)
    tf = validate_timeframe(args.timeframe)
    df, _ = read_csv_ohlcv(args.csv)
    from signal_engine.walkforward import run_walk_forward

    risk, costs = _risk_and_costs(args)
    params = _kv_params(args.param)
    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    wf = run_walk_forward(
        df,
        lambda: strategy_cls(**params),
        market,
        tf,
        risk,
        costs,
        n_folds=args.folds,
        train_candles=args.train_candles,
        eval_candles=args.eval_candles,
        holdout_fraction=args.holdout_fraction,
        initial_equity=args.initial_equity,
        provenance={"source": f"csv:{args.csv}"},
    )
    print(json.dumps(wf.as_dict(), indent=2))
    return 0


def cmd_baseline(args) -> int:
    market = validate_market(args.market)
    tf = validate_timeframe(args.timeframe)
    df, _ = read_csv_ohlcv(args.csv)
    if args.m < 1:
        raise SystemExit("--m must be >= 1")
    from signal_engine.stats import run_randomized_baseline

    risk, costs = _risk_and_costs(args)
    params = _kv_params(args.param)
    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    base = run_randomized_baseline(
        df,
        lambda: strategy_cls(**params),
        market,
        tf,
        risk,
        costs,
        m_simulations=args.m,
        seed=args.seed,
        initial_equity=args.initial_equity,
        provenance={"source": f"csv:{args.csv}"},
    )
    print(json.dumps(base.as_dict(), indent=2))
    return 0


def cmd_report(args) -> int:
    market = validate_market(args.market)
    tf = validate_timeframe(args.timeframe)
    if args.csv is not None:
        df, rep = read_csv_ohlcv(args.csv)
        provenance = f"csv:{args.csv}"
    elif args.synthetic:
        df = synth_ohlcv(tf, args.synthetic_start, args.synthetic_n, seed=args.synthetic_seed)
        rep = None
        provenance = "SYNTHETIC"
    else:
        raise SystemExit("--csv or --synthetic must be provided")
    from signal_engine.data.validate import validate_ohlcv

    if rep is None:
        df, rep = validate_ohlcv(df)
    risk, costs = _risk_and_costs(args)
    cell = evaluate_cell(
        df,
        rep,
        market,
        tf,
        strategy_name=args.strategy,
        strategy_params=_kv_params(args.param),
        risk=risk,
        costs=costs,
        provenance=provenance,
        run_walkforward_flag=not args.no_walkforward,
        run_baseline_flag=not args.no_baseline,
        baseline_m=args.baseline_m,
        baseline_seed=args.seed,
        initial_equity=args.initial_equity,
    )
    print(report_to_json([cell]))
    return 0


def cmd_matrix(args) -> int:
    risk, costs = _risk_and_costs(args)
    params = _kv_params(args.param)
    cells = []
    from signal_engine.data.validate import validate_ohlcv

    for market, tf in matrix_cells():
        try:
            if args.csv_dir:
                csv_path = Path(args.csv_dir) / f"{market}_{tf}.csv"
                if not csv_path.exists():
                    cells.append(
                        {
                            "market": market,
                            "timeframe": tf,
                            "status": "UNAVAILABLE_DATA",
                            "detail": f"missing_data_file:{csv_path}",
                        }
                    )
                    continue
                df, rep = read_csv_ohlcv(str(csv_path), expected_spacing=tf)
                provenance = f"csv:{csv_path}"
            elif args.synthetic:
                df = synth_ohlcv(tf, args.synthetic_start, args.synthetic_n, seed=args.synthetic_seed)
                df, rep = validate_ohlcv(df)
                provenance = "SYNTHETIC"
            else:
                cells.append(
                    {
                        "market": market,
                        "timeframe": tf,
                        "status": "UNAVAILABLE_DATA",
                        "detail": "no data source provided (--csv-dir or --synthetic required)",
                    }
                )
                continue

            strategy_name = args.strategy
            if strategy_name is None:
                cells.append(
                    {
                        "market": market,
                        "timeframe": tf,
                        "status": "UNAVAILABLE_DATA",
                        "detail": "--strategy must be provided for evaluation",
                    }
                )
                continue

            cell = evaluate_cell(
                df,
                rep,
                market,
                tf,
                strategy_name=strategy_name,
                strategy_params=params,
                risk=risk,
                costs=costs,
                provenance=provenance,
                run_walkforward_flag=not args.no_walkforward,
                run_baseline_flag=not args.no_baseline,
                baseline_m=args.baseline_m,
                baseline_seed=args.seed,
                initial_equity=args.initial_equity,
            )
            cells.append(cell.as_dict())
        except Exception as exc:
            cells.append(
                {"market": market, "timeframe": tf, "status": "ERROR", "error": str(exc)}
            )

    print(json.dumps({"n_cells_expected": 21, "n_cells_reported": len(cells), "cells": cells}, indent=2))
    return 0


def cmd_fetch_crypto(args) -> int:
    from signal_engine.data.binance_source import BinanceUnavailableError, fetch_range

    market = validate_market(args.market)
    tf = validate_timeframe(args.timeframe)
    try:
        df = fetch_range(
            market,
            tf,
            pd.Timestamp(args.start, tz="UTC"),
            pd.Timestamp(args.end, tz="UTC"),
        )
    except BinanceUnavailableError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    out_path = Path(args.out)
    df.reset_index().to_csv(out_path, index=False)
    print(
        json.dumps(
            {
                "ok": True,
                "n_candles": len(df),
                "coverage_start_utc": df.index[0].isoformat(),
                "coverage_end_utc": df.index[-1].isoformat(),
                "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
                "symbol_mapping": df.attrs.get("symbol_mapping"),
                "source": df.attrs.get("source"),
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="signal-engine",
        description="Multi-market signal research and evaluation engine (research only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-csv", help="Validate an OHLCV CSV dataset")
    v.add_argument("--csv", required=True)
    v.add_argument("--expected-spacing", choices=TIMEFRAMES, default=None)
    v.set_defaults(func=cmd_validate_csv)

    b = sub.add_parser("backtest", help="Run a deterministic backtest")
    _common_opts(b, strategy=True)
    b.add_argument("--csv", required=True)
    b.add_argument("--market", required=True, choices=MARKETS)
    b.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="Run chronological walk-forward evaluation")
    _common_opts(w, strategy=True)
    w.add_argument("--csv", required=True)
    w.add_argument("--market", required=True, choices=MARKETS)
    w.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    w.add_argument("--folds", type=int, default=4)
    w.add_argument("--train-candles", type=int, default=400)
    w.add_argument("--eval-candles", type=int, default=100)
    w.add_argument("--holdout-fraction", type=float, default=0.2)
    w.set_defaults(func=cmd_walkforward)

    bl = sub.add_parser("baseline", help="Run randomized-baseline statistical evaluation")
    _common_opts(bl, strategy=True)
    bl.add_argument("--csv", required=True)
    bl.add_argument("--market", required=True, choices=MARKETS)
    bl.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    bl.add_argument("--m", type=int, default=999)
    bl.add_argument("--seed", type=int, default=20260101)
    bl.set_defaults(func=cmd_baseline)

    r = sub.add_parser("report", help="Full report for one market/timeframe")
    _common_opts(r, strategy=True)
    r.add_argument("--csv", default=None)
    r.add_argument(
        "--synthetic", action="store_true", help="Use deterministic synthetic data (labeled)"
    )
    r.add_argument("--synthetic-start", default="2024-01-01")
    r.add_argument("--synthetic-n", type=int, default=2000)
    r.add_argument("--synthetic-seed", type=int, default=42)
    r.add_argument("--market", required=True, choices=MARKETS)
    r.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    r.add_argument("--no-walkforward", action="store_true")
    r.add_argument("--no-baseline", action="store_true")
    r.add_argument("--baseline-m", type=int, default=99)
    r.add_argument("--seed", type=int, default=20260101)
    r.set_defaults(func=cmd_report)

    m = sub.add_parser("matrix", help="Run the full 21-cell matrix") 
    _common_opts(m, strategy=True)
    m.add_argument("--csv-dir", default=None, help="Directory of {MARKET}_{TF}.csv files")
    m.add_argument("--synthetic", action="store_true")
    m.add_argument("--synthetic-start", default="2024-01-01")
    m.add_argument("--synthetic-n", type=int, default=2000)
    m.add_argument("--synthetic-seed", type=int, default=42)
    m.add_argument("--no-walkforward", action="store_true")
    m.add_argument("--no-baseline", action="store_true")
    m.add_argument("--baseline-m", type=int, default=99)
    m.add_argument("--seed", type=int, default=20260101)
    m.set_defaults(func=cmd_matrix)

    f = sub.add_parser("fetch-crypto", help="Fetch public klines for BTCUSDT/ETHUSDT")
    f.add_argument("--market", required=True, choices=("BTCUSDT", "ETHUSDT"))
    f.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    f.add_argument("--start", required=True, help="UTC start, e.g. 2025-01-01T00:00")
    f.add_argument("--end", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fetch_crypto)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
