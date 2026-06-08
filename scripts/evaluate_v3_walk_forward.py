#!/usr/bin/env python3
"""Evaluate a v3 candidate on fixed-parameter walk-forward slices."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import DataLoader
from scripts.explore_v3_edge import BASE_FIELDS, PARAM_FIELDS, make_strategy, run_metrics
from scripts.run_backtest import vectorized_preprocess


logger = logging.getLogger("v3_walk_forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v3 candidate on rolling OOS slices")
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--exit-on-flat-signal", action="store_true")
    parser.add_argument("--no-opposite-signal-exit", dest="exit_on_opposite_signal", action="store_false", default=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def load_candidate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(path.read_text())
    best = summary["best"]
    params = {field: best[field] for field in PARAM_FIELDS}
    return summary, params


def walk_forward_windows(start: str, end: str, train_months: int, test_months: int):
    train_start = pd.Timestamp(start, tz="UTC")
    final_end = pd.Timestamp(end, tz="UTC")
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > final_end:
            break
        yield train_start, train_end, test_start, test_end
        train_start = train_start + pd.DateOffset(months=test_months)


def load_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    loader = DataLoader(use_cache=True)
    df = loader.load_symbol(symbol, "1m")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    return vectorized_preprocess(df, timeframe)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    profitable = [row for row in rows if row["total_return_pct"] > 0]
    positive_exp = [row for row in rows if row["expectancy_r"] > 0]
    return {
        "windows": len(rows),
        "profitable_windows": len(profitable),
        "positive_expectancy_windows": len(positive_exp),
        "mean_profit_factor": sum(row["profit_factor"] for row in rows) / len(rows),
        "min_profit_factor": min(row["profit_factor"] for row in rows),
        "mean_expectancy_r": sum(row["expectancy_r"] for row in rows) / len(rows),
        "min_expectancy_r": min(row["expectancy_r"] for row in rows),
        "mean_return_pct": sum(row["total_return_pct"] for row in rows) / len(rows),
        "min_return_pct": min(row["total_return_pct"] for row in rows),
        "max_drawdown_r": max(row["max_drawdown_r"] for row in rows),
        "max_drawdown_pct": max(row["max_drawdown_pct"] for row in rows),
        "max_consecutive_losses": max(row["max_consecutive_losses"] for row in rows),
        "total_trades": sum(row["total_trades"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    source_summary, params = load_candidate(args.params_file)
    symbol = args.symbol or source_summary.get("symbol", "BTCUSDT")
    timeframe = args.timeframe or source_summary.get("timeframe", "4h")

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"wf_{symbol}_{timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s %s data", symbol, timeframe)
    base_df = load_data(symbol, timeframe, args.start, args.end)
    strategy = make_strategy(symbol, timeframe, args.risk_per_trade, params)
    featured = strategy.initialize(base_df)

    rows: list[dict[str, Any]] = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.end, args.train_months, args.test_months),
        start=1,
    ):
        test_df = featured.loc[(featured.index >= test_start) & (featured.index <= test_end)]
        metrics = run_metrics(
            test_df,
            symbol,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
            args.exit_on_flat_signal,
            args.exit_on_opposite_signal,
            strategy.config.min_holding_bars,
            strategy.config.max_holding_bars,
        )
        row: dict[str, Any] = {
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
        }
        row.update({field: metrics.get(field, 0) for field in BASE_FIELDS})
        rows.append(row)

    aggregate = summarize(rows)
    write_csv(run_dir / "walk_forward.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps({
        "symbol": symbol,
        "timeframe": timeframe,
        "params_file": str(args.params_file),
        "params": params,
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "sizing_mode": args.sizing_mode,
            "max_position_pct": args.max_position_pct,
            "exit_on_flat_signal": args.exit_on_flat_signal,
            "exit_on_opposite_signal": args.exit_on_opposite_signal,
        },
        "aggregate": aggregate,
    }, indent=2, sort_keys=True, default=str))
    logger.info("Walk-forward saved to %s", run_dir)
    logger.info(
        "WF %s/%s profitable, mean expR %.3f, min expR %.3f, max ddR %.2f",
        aggregate.get("profitable_windows", 0),
        aggregate.get("windows", 0),
        aggregate.get("mean_expectancy_r", 0),
        aggregate.get("min_expectancy_r", 0),
        aggregate.get("max_drawdown_r", 0),
    )


if __name__ == "__main__":
    main()
