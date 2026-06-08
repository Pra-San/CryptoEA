#!/usr/bin/env python3
"""Search v3 parameters using rolling in-sample walk-forward robustness."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import DataLoader
from scripts.explore_v3_edge import BASE_FIELDS, PARAM_FIELDS, make_strategy, run_metrics, sample_params
from scripts.run_backtest import vectorized_preprocess


logger = logging.getLogger("v3_wf_edge_explorer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward robust v3 search")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


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


def window_score(metrics: dict[str, Any]) -> float:
    trades = metrics.get("total_trades", 0)
    if trades < 5:
        return -25
    return (
        float(metrics.get("expectancy_r", 0)) * 80
        + min(float(metrics.get("profit_factor", 0)), 3.0) * 12
        + float(metrics.get("total_return_pct", 0)) * 0.4
        - float(metrics.get("max_drawdown_r", 0)) * 1.5
        - float(metrics.get("max_consecutive_losses", 0)) * 1.0
    )


def aggregate_score(window_metrics: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    if not window_metrics:
        return -1_000_000, {}

    exp_values = [float(item.get("expectancy_r", 0)) for item in window_metrics]
    pf_values = [float(item.get("profit_factor", 0)) for item in window_metrics]
    dd_values = [float(item.get("max_drawdown_r", 0)) for item in window_metrics]
    ret_values = [float(item.get("total_return_pct", 0)) for item in window_metrics]
    trade_values = [int(item.get("total_trades", 0)) for item in window_metrics]
    positive_exp = sum(1 for value in exp_values if value > 0)
    profitable = sum(1 for value in ret_values if value > 0)
    total_trades = sum(trade_values)
    if total_trades < 80:
        return -1_000_000 + total_trades, {}

    per_window = [window_score(item) for item in window_metrics]
    aggregate = {
        "wf_windows": len(window_metrics),
        "wf_positive_exp": positive_exp,
        "wf_profitable": profitable,
        "wf_mean_exp_r": sum(exp_values) / len(exp_values),
        "wf_min_exp_r": min(exp_values),
        "wf_mean_pf": sum(pf_values) / len(pf_values),
        "wf_min_pf": min(pf_values),
        "wf_mean_return_pct": sum(ret_values) / len(ret_values),
        "wf_min_return_pct": min(ret_values),
        "wf_max_dd_r": max(dd_values),
        "wf_total_trades": total_trades,
    }
    score = (
        sum(per_window) / len(per_window)
        + positive_exp / len(window_metrics) * 80
        + profitable / len(window_metrics) * 40
        + aggregate["wf_min_exp_r"] * 60
        - aggregate["wf_max_dd_r"] * 1.2
    )
    return score, aggregate


def evaluate_slice(
    featured: pd.DataFrame,
    symbol: str,
    args: argparse.Namespace,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    strategy = make_strategy(symbol, args.timeframe, args.risk_per_trade, params)
    test_df = featured.loc[(featured.index >= start) & (featured.index <= end)]
    return run_metrics(
        test_df,
        symbol,
        args.balance,
        args.risk_per_trade,
        args.fee_rate,
        args.slippage_rate,
        args.max_position_pct,
        args.sizing_mode,
        False,
        True,
        strategy.config.min_holding_bars,
        strategy.config.max_holding_bars,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"v3_wf_edge_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s %s data", args.symbol, args.timeframe)
    base_df = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    logger.info("Walk-forward training search: %d candidates across %d windows", args.trials, len(train_windows))
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        strategy = make_strategy(args.symbol, args.timeframe, args.risk_per_trade, params)
        featured = strategy.initialize(base_df)
        window_metrics = [
            evaluate_slice(featured, args.symbol, args, params, test_start, test_end)
            for _, _, test_start, test_end in train_windows
        ]
        score, aggregate = aggregate_score(window_metrics)
        row = {
            "trial": trial,
            "wf_score": score,
            **{key: getattr(strategy.config, key) for key in PARAM_FIELDS},
            **aggregate,
        }
        rows.append(row)
        if (trial + 1) % 50 == 0:
            best = max(rows, key=lambda item: item["wf_score"])
            logger.info(
                "  %d/%d candidates, best score %.2f pos %s/%s mean expR %.3f max ddR %.2f",
                trial + 1,
                args.trials,
                best["wf_score"],
                best.get("wf_positive_exp", 0),
                best.get("wf_windows", 0),
                best.get("wf_mean_exp_r", 0),
                best.get("wf_max_dd_r", 0),
            )

    rows.sort(key=lambda item: item["wf_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["wf_rank"] = rank

    top = rows[: args.top]
    logger.info("Validating top %d robust candidates on holdout", len(top))
    for row in top:
        params = {field: row[field] for field in PARAM_FIELDS}
        strategy = make_strategy(args.symbol, args.timeframe, args.risk_per_trade, params)
        featured = strategy.initialize(base_df)
        validation_metrics = evaluate_slice(featured, args.symbol, args, params, validation_start, validation_end)
        for field in BASE_FIELDS:
            row[f"validation_{field}"] = validation_metrics.get(field, 0)

    top.sort(
        key=lambda item: (
            float(item.get("validation_expectancy_r", 0)) * 100
            + min(float(item.get("validation_profit_factor", 0)), 3.0) * 20
            - float(item.get("validation_max_drawdown_r", 0)) * 2
            + float(item.get("wf_score", 0)) * 0.25
        ),
        reverse=True,
    )

    write_csv(run_dir / "walkforward_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top)
    best = top[0] if top else rows[0]
    (run_dir / "best_summary.json").write_text(json.dumps({
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "period": {
            "start": args.start,
            "train_end": args.train_end,
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
        },
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "sizing_mode": args.sizing_mode,
            "max_position_pct": args.max_position_pct,
            "exit_on_flat_signal": False,
            "exit_on_opposite_signal": True,
        },
        "seed": args.seed,
        "trials": args.trials,
        "best": best,
    }, indent=2, sort_keys=True, default=str))
    logger.info("Best robust candidate saved in %s", run_dir)
    logger.info(
        "Best validation PF %.2f expR %.3f ddR %.2f | train WF pos %s/%s mean expR %.3f",
        best.get("validation_profit_factor", 0),
        best.get("validation_expectancy_r", 0),
        best.get("validation_max_drawdown_r", 0),
        best.get("wf_positive_exp", 0),
        best.get("wf_windows", 0),
        best.get("wf_mean_exp_r", 0),
    )


if __name__ == "__main__":
    main()
