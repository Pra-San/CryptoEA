#!/usr/bin/env python3
"""Search strategy-family candidates using rolling OOS training windows."""

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
from scripts.explore_strategy_families import (
    BASE_FIELDS,
    PARAM_FIELDS,
    apply_strategy,
    run_metrics,
    sample_params,
)
from scripts.run_backtest import vectorized_preprocess


logger = logging.getLogger("strategy_family_wf_explorer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward robust strategy-family search")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--min-total-trades", type=int, default=80)
    parser.add_argument("--min-validation-trades", type=int, default=25)
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
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def window_score(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 4:
        return -35.0
    return (
        float(metrics.get("expectancy_r", 0)) * 90
        + min(float(metrics.get("profit_factor", 0)), 3.0) * 10
        + float(metrics.get("total_return_pct", 0)) * 0.25
        - float(metrics.get("max_drawdown_r", 0)) * 1.8
        - float(metrics.get("max_consecutive_losses", 0)) * 1.2
    )


def aggregate_score(window_metrics: list[dict[str, Any]], min_total_trades: int) -> tuple[float, dict[str, Any]]:
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
    if total_trades < min_total_trades:
        return -1_000_000 + total_trades, {}

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
    per_window = [window_score(item) for item in window_metrics]
    score = (
        sum(per_window) / len(per_window)
        + positive_exp / len(window_metrics) * 90
        + profitable / len(window_metrics) * 45
        + aggregate["wf_min_exp_r"] * 70
        - aggregate["wf_max_dd_r"] * 1.5
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
    return run_metrics(
        featured.loc[(featured.index >= start) & (featured.index <= end)],
        symbol,
        args.balance,
        args.risk_per_trade,
        args.fee_rate,
        args.slippage_rate,
        args.max_position_pct,
        args.sizing_mode,
        int(params["min_holding_bars"]),
        int(params["max_holding_bars"]),
    )


def validation_rank(row: dict[str, Any], min_validation_trades: int) -> float:
    if int(row.get("validation_total_trades", 0)) < min_validation_trades:
        return -1_000_000 + int(row.get("validation_total_trades", 0))
    return (
        float(row.get("validation_expectancy_r", 0)) * 120
        + min(float(row.get("validation_profit_factor", 0)), 3.0) * 20
        - float(row.get("validation_max_drawdown_r", 0)) * 2.5
        + float(row.get("wf_score", 0)) * 0.35
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
    run_dir = output_root / f"family_wf_edge_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    logger.info("WF training search: %d candidates across %d windows", args.trials, len(train_windows))
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        featured = apply_strategy(base_df, params)
        window_metrics = [
            evaluate_slice(featured, args.symbol, args, params, test_start, test_end)
            for _, _, test_start, test_end in train_windows
        ]
        score, aggregate = aggregate_score(window_metrics, args.min_total_trades)
        row = {
            "trial": trial,
            "wf_score": score,
            **{field: params.get(field) for field in PARAM_FIELDS},
            **aggregate,
        }
        rows.append(row)
        if (trial + 1) % 50 == 0:
            best = max(rows, key=lambda item: item["wf_score"])
            logger.info(
                "  %d/%d best %s score %.2f pos %s/%s mean expR %.3f max ddR %.2f trades %s",
                trial + 1,
                args.trials,
                best.get("family"),
                best.get("wf_score", 0),
                best.get("wf_positive_exp", 0),
                best.get("wf_windows", 0),
                best.get("wf_mean_exp_r", 0),
                best.get("wf_max_dd_r", 0),
                best.get("wf_total_trades", 0),
            )

    rows.sort(key=lambda item: item["wf_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["wf_rank"] = rank

    top = rows[: args.top]
    logger.info("Validating top %d WF-ranked candidates on holdout", len(top))
    for row in top:
        params = {field: row[field] for field in PARAM_FIELDS}
        featured = apply_strategy(base_df, params)
        validation = evaluate_slice(featured, args.symbol, args, params, validation_start, validation_end)
        for field in BASE_FIELDS:
            row[f"validation_{field}"] = validation.get(field, 0)
        row["validation_rank_score"] = validation_rank(row, args.min_validation_trades)

    top.sort(key=lambda item: item.get("validation_rank_score", -1_000_000), reverse=True)
    best = top[0] if top else rows[0]

    write_csv(run_dir / "walkforward_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top)
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
        },
        "seed": args.seed,
        "trials": args.trials,
        "top": args.top,
        "best": best,
    }, indent=2, sort_keys=True, default=str))
    logger.info("WF family search saved in %s", run_dir)
    logger.info(
        "Best %s | validation PF %.2f expR %.3f ddR %.2f trades %s | WF pos %s/%s mean expR %.3f",
        best.get("family"),
        best.get("validation_profit_factor", 0),
        best.get("validation_expectancy_r", 0),
        best.get("validation_max_drawdown_r", 0),
        best.get("validation_total_trades", 0),
        best.get("wf_positive_exp", 0),
        best.get("wf_windows", 0),
        best.get("wf_mean_exp_r", 0),
    )


if __name__ == "__main__":
    main()
