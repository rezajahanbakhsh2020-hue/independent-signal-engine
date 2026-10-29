"""Command-line interface for the signal engine.

Subcommands:
- validate-csv    : validate an OHLCV CSV and print the audit report
- backtest        : run a deterministic backtest for one strategy/market/timeframe
- walkforward     : run chronological walk-forward evaluation
- baseline        : run the randomized-baseline statistical test
- report          : produce a full JSON report for one market/timeframe
- matrix          : run the engine across the complete 21-cell matrix
- signal          : latest BUY/SELL/NO TRADE decision check on real recent candles
- fetch-crypto    : fetch public klines for any valid Binance USDT crypto pair
- fetch-gold      : fetch closed XAUUSD OHLC candles from the public BiQuote API

All costs expressed in basis points. Provenance and honest statuses are
included in every JSON output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd

from signal_engine.backtest import backtest, CostConfig
from signal_engine.constants import MARKETS, TIMEFRAMES, validate_market, validate_market_or_crypto_symbol, validate_timeframe
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


_TIMEFRAME_DURATION = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "1D": timedelta(days=1),
}


def _freshness_report(df: pd.DataFrame, timeframe: str, max_age_hours: float | None) -> dict:
    """Check that the newest candle is closed and recent; never call stale data current."""
    if max_age_hours is not None and (not math.isfinite(max_age_hours) or max_age_hours < 0):
        raise SystemExit("--max-age-hours must be finite and >= 0")
    now = pd.Timestamp.now(tz="UTC")
    duration = _TIMEFRAME_DURATION[timeframe]
    latest_open = pd.Timestamp(df.index[-1]).tz_convert("UTC")
    latest_close = latest_open + duration
    age_hours = (now - latest_close).total_seconds() / 3600.0
    allowed_hours = max_age_hours if max_age_hours is not None else duration.total_seconds() * 3 / 3600.0
    result = {
        "checked_at_utc": now.isoformat(),
        "latest_candle_open_utc": latest_open.isoformat(),
        "latest_candle_close_utc": latest_close.isoformat(),
        "age_since_close_hours": round(age_hours, 6),
        "max_age_hours": allowed_hours,
        "is_closed": latest_close <= now,
        "is_fresh": latest_close <= now and age_hours <= allowed_hours,
    }
    if latest_close > now:
        result["reason"] = "latest_candle_not_closed"
    elif age_hours > allowed_hours:
        result["reason"] = "stale_data"
    else:
        result["reason"] = None
    return result


def cmd_signal(args) -> int:
    market = validate_market_or_crypto_symbol(args.market)
    tf = validate_timeframe(args.timeframe)
    from signal_engine.data.validate import validate_ohlcv
    from signal_engine.latest_signal import latest_decision_report

    try:
        if args.csv:
            df, rep = read_csv_ohlcv(args.csv, expected_spacing=tf)
            provenance = f"csv:{args.csv}"
        elif args.synthetic:
            df = synth_ohlcv(tf, args.synthetic_start, args.synthetic_n, seed=args.synthetic_seed)
            df, rep = validate_ohlcv(df)
            provenance = "SYNTHETIC test/demo-only fixture; not market data"
        else:
            raise SystemExit("either --csv or --synthetic must be provided")
    except DataValidationError as exc:
        out = {"ok": False, "error": str(exc), "status": "NO TRADE (data rejected; fail-closed)"}
        if exc.report is not None:
            out["validation"] = exc.report.as_dict()
        print(json.dumps(out, indent=2))
        return 1
    freshness = None
    if not args.synthetic:
        freshness = _freshness_report(df, tf, args.max_age_hours)
        if not freshness["is_fresh"]:
            out = {
                "kind": "latest_signal_check",
                "market": market,
                "timeframe": tf,
                "strategy": args.strategy,
                "provenance": provenance,
                "decision": {"label": "NO TRADE", "reason": freshness["reason"]},
                "pending_raw_direction": None,
                "freshness": freshness,
                "validation": rep.as_dict(),
                "research_only": "stale or unfinished data is rejected; no profitability is implied",
            }
            print(json.dumps(out, indent=2))
            return 0

    risk, _costs = _risk_and_costs(args)
    params = _kv_params(args.param)
    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    strat = strategy_cls(**params)
    out = latest_decision_report(
        df,
        strat,
        risk,
        market,
        tf,
        provenance=provenance,
        equity=args.initial_equity,
    )
    out["strategy_params"] = params
    out["validation"] = rep.as_dict()
    out["freshness"] = freshness or {"checked": False, "reason": "synthetic_test_demo_fixture"}
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

    from signal_engine.constants import validate_crypto_symbol
    market = validate_crypto_symbol(args.market)
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


def cmd_chart_crypto(args) -> int:
    """Fetch a chosen USDT crypto pair, compute its latest decision, and export a chart."""
    from signal_engine.constants import validate_crypto_symbol
    from signal_engine.data.binance_source import BinanceUnavailableError, fetch_range
    from signal_engine.chart import build_chart_html
    from signal_engine.latest_signal import latest_decision_report

    try:
        market = validate_crypto_symbol(args.market)
        tf = validate_timeframe(args.timeframe)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    if args.max_candles < 1:
        print(json.dumps({"ok": False, "error": "--max-candles must be >= 1"}), file=sys.stderr)
        return 2
    now = pd.Timestamp.now(tz="UTC")

    def utc_timestamp(value: str) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            return stamp.tz_localize("UTC")
        return stamp.tz_convert("UTC")

    try:
        end = utc_timestamp(args.end) if args.end else now
        start = utc_timestamp(args.start) if args.start else end - pd.Timedelta(days=30)
    except (TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"invalid --start/--end timestamp: {exc}"}), file=sys.stderr)
        return 2
    if start >= end:
        print(json.dumps({"ok": False, "error": "--start must be earlier than --end"}), file=sys.stderr)
        return 2
    try:
        df = fetch_range(market, tf, start, end)
    except (BinanceUnavailableError, DataValidationError) as exc:
        print(json.dumps({"ok": False, "market": market, "timeframe": tf, "error": str(exc)}), file=sys.stderr)
        return 1
    duration = _TIMEFRAME_DURATION[tf]
    df = df.loc[(df.index + duration) <= now]
    if df.empty:
        raise SystemExit("No closed candles were returned for the selected pair/timeframe")
    params = _kv_params(args.param)
    strategy = STRATEGY_REGISTRY[args.strategy](**params)
    risk, _costs = _risk_and_costs(args)
    freshness = _freshness_report(df, tf, args.max_age_hours)
    decision = latest_decision_report(
        df, strategy, risk, market, tf,
        provenance=str(df.attrs.get("source", "public:binance")),
        equity=args.initial_equity,
    )
    decision["freshness"] = freshness
    if not freshness["is_fresh"]:
        decision["decision"] = {"label": "NO TRADE", "reason": freshness["reason"]}
    page = build_chart_html(
        df, market=market, timeframe=tf, strategy_name=args.strategy,
        strategy_params=params, strategy=strategy, latest_decision=decision,
        max_candles=args.max_candles,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(json.dumps({
        "ok": True, "market": market, "timeframe": tf, "strategy": args.strategy,
        "n_closed_candles": len(df), "latest_decision": decision["decision"],
        "freshness": freshness, "source": df.attrs.get("source"),
        "chart_html": str(out_path),
        "note": "Historical markers are research signals, not live orders or a profitability guarantee.",
    }, indent=2))
    return 0


def cmd_fetch_gold(args) -> int:
    from signal_engine.data.biquote_source import BiQuoteUnavailableError, fetch_range

    try:
        df = fetch_range(
            "XAUUSD",
            args.timeframe,
            pd.Timestamp(args.start),
            pd.Timestamp(args.end),
        )
    except (BiQuoteUnavailableError, DataValidationError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    out_path = Path(args.out)
    df.reset_index().to_csv(out_path, index=False)
    print(
        json.dumps(
            {
                "ok": True,
                "market": "XAUUSD",
                "timeframe": args.timeframe,
                "n_candles": len(df),
                "coverage_start_utc": df.index[0].isoformat(),
                "coverage_end_utc": df.index[-1].isoformat(),
                "retrieval_time_utc": df.attrs.get("retrieval_time_utc"),
                "source": df.attrs.get("source"),
                "volume_semantics": df.attrs.get("volume_semantics"),
                "validation": df.attrs.get("validation_report"),
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

    s = sub.add_parser(
        "signal",
        help="Latest BUY/SELL/NO TRADE decision check on real recent candles (e.g. fetched with fetch-crypto/fetch-gold)",
    )
    _common_opts(s, strategy=True)
    s.add_argument("--csv", default=None)
    s.add_argument(
        "--synthetic", action="store_true", help="Use deterministic synthetic data (labeled test/demo-only)"
    )
    s.add_argument("--synthetic-start", default="2024-01-01")
    s.add_argument("--synthetic-n", type=int, default=2000)
    s.add_argument("--synthetic-seed", type=int, default=42)
    s.add_argument("--market", required=True, help="Market or USDT crypto pair such as SOLUSDT")
    s.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    s.add_argument("--max-age-hours", type=float, default=None, help="Override freshness limit; default is three candle durations")
    s.set_defaults(func=cmd_signal)

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

    f = sub.add_parser("fetch-crypto", help="Fetch public klines for any Binance USDT crypto pair")
    f.add_argument("--market", required=True, help="Any valid Binance USDT pair, e.g. SOLUSDT or ADAUSDT")
    f.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    f.add_argument("--start", required=True, help="UTC start, e.g. 2025-01-01T00:00")
    f.add_argument("--end", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fetch_crypto)

    ch = sub.add_parser("chart-crypto", help="Fetch any USDT crypto pair, calculate a signal, and export an HTML chart")
    _common_opts(ch, strategy=True)
    ch.add_argument("--market", required=True, help="Any valid Binance USDT pair, e.g. SOLUSDT")
    ch.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    ch.add_argument("--start", default=None, help="UTC start; defaults to 30 days ago")
    ch.add_argument("--end", default=None, help="UTC end; defaults to now")
    ch.add_argument("--max-age-hours", type=float, default=None, help="Freshness limit; default is three candle durations")
    ch.add_argument("--max-candles", type=int, default=300)
    ch.add_argument("--out", required=True, help="Output standalone HTML chart path")
    ch.set_defaults(func=cmd_chart_crypto)

    g = sub.add_parser("fetch-gold", help="Fetch public XAUUSD OHLC candles from BiQuote")
    g.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    g.add_argument("--start", required=True, help="UTC start, e.g. 2025-01-01T00:00:00Z")
    g.add_argument("--end", required=True, help="UTC end, e.g. 2025-02-01T00:00:00Z")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_fetch_gold)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
