#!/usr/bin/env python3
"""Adaptive walk-forward selector across saved strategy candidates.

For each rolling window, candidates are ranked using only the prior training
period. The selected top K candidates are then evaluated as an equal-weight
portfolio on the next out-of-sample period.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_strategy_portfolio import (
    Candidate,
    apply_candidate_strategy,
    load_base_data,
    load_candidate,
    summarize_windows,
    write_csv,
)


logger = logging.getLogger("adaptive_candidate_wf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive walk-forward candidate selector")
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--candidate-glob", action="append", default=[])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--unique-symbols", action="store_true")
    parser.add_argument("--min-train-trades", type=int, default=12)
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


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.candidate)
    for pattern in args.candidate_glob:
        paths.extend(PROJECT_ROOT.glob(pattern))
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        unique[str(resolved)] = resolved
    return sorted(unique.values())


def train_score(metrics: dict[str, Any], min_trades: int) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < min_trades:
        return -1_000_000 + trades
    exp_r = float(metrics.get("expectancy_r", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 4.0)
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    return exp_r * 130 + (pf - 1.0) * 24 - dd_r * 2.0 - loss_streak * 1.5


def precompute_features(
    candidates: list[Candidate],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, pd.DataFrame]]:
    data_cache: dict[tuple[str, str], pd.DataFrame] = {}
    feature_cache: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        base = load_base_data(candidate, args.start, args.end, data_cache)
        feature_cache[candidate.label] = apply_candidate_strategy(base, candidate, args.risk_per_trade)
    return data_cache, feature_cache


def evaluate_selected_portfolio(
    selected: list[Candidate],
    feature_cache: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Use a lightweight local copy of evaluate_period that consumes precomputed
    # feature frames to avoid recomputing all strategy columns per window.
    from scripts.evaluate_strategy_portfolio import run_candidate, scale_trade, metrics_dict

    all_trades = []
    timeline = pd.DatetimeIndex([])
    weight = 1.0 / len(selected)
    for candidate in selected:
        featured = feature_cache[candidate.label].loc[
            (feature_cache[candidate.label].index >= start)
            & (feature_cache[candidate.label].index <= end)
        ]
        timeline = timeline.union(featured.index)
        result = run_candidate(
            candidate,
            featured,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
        )
        all_trades.extend(scale_trade(trade, candidate, weight) for trade in result.get("trades", []))
    return metrics_dict(all_trades, timeline, args.balance)


def write_selection_rankings(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_top(scored: list[tuple[float, Candidate, dict[str, Any]]], top_k: int, unique_symbols: bool) -> list[Candidate]:
    if not unique_symbols:
        return [candidate for _, candidate, _ in scored[:top_k]]

    selected: list[Candidate] = []
    seen_symbols: set[str] = set()
    for _, candidate, _ in scored:
        if candidate.symbol in seen_symbols:
            continue
        selected.append(candidate)
        seen_symbols.add(candidate.symbol)
        if len(selected) >= top_k:
            return selected
    for _, candidate, _ in scored:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    paths = candidate_paths(args)
    if not paths:
        raise SystemExit("No candidates supplied.")
    candidates = [load_candidate(path) for path in paths]
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"adaptive_wf_top{args.top_k}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Precomputing features for %d candidates", len(candidates))
    _, feature_cache = precompute_features(candidates, args)

    window_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()

    windows = list(walk_forward_windows(args.start, args.end, args.train_months, args.test_months))
    logger.info("Running adaptive WF across %d windows", len(windows))
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        scored = []
        for candidate in candidates:
            train_metrics = evaluate_selected_portfolio(
                [candidate],
                feature_cache,
                train_start,
                train_end,
                args,
            )
            score = train_score(train_metrics, args.min_train_trades)
            scored.append((score, candidate, train_metrics))
            ranking_rows.append({
                "window": i,
                "candidate": candidate.label,
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "kind": candidate.kind,
                "train_score": score,
                "train_total_trades": train_metrics.get("total_trades", 0),
                "train_profit_factor": train_metrics.get("profit_factor", 0),
                "train_expectancy_r": train_metrics.get("expectancy_r", 0),
                "train_max_drawdown_r": train_metrics.get("max_drawdown_r", 0),
            })

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = select_top(scored, args.top_k, args.unique_symbols)
        for candidate in selected:
            selected_counts[candidate.label] += 1

        test_metrics = evaluate_selected_portfolio(selected, feature_cache, test_start, test_end, args)
        window_rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            "selected": "|".join(candidate.label for candidate in selected),
            **test_metrics,
        })
        logger.info(
            "  WF %02d selected %s | trades %s PF %.2f expR %.3f ddR %.2f",
            i,
            ", ".join(candidate.symbol + ":" + candidate.kind for candidate in selected),
            test_metrics.get("total_trades", 0),
            test_metrics.get("profit_factor", 0),
            test_metrics.get("expectancy_r", 0),
            test_metrics.get("max_drawdown_r", 0),
        )

    aggregate = summarize_windows(window_rows)
    write_csv(run_dir / "walk_forward.csv", window_rows)
    write_selection_rankings(run_dir / "candidate_rankings.csv", ranking_rows)
    summary = {
        "symbol": "ADAPTIVE_PORTFOLIO",
        "timeframe": "mixed",
        "strategy": "adaptive_candidate_selector",
        "aggregate": aggregate,
        "selection_counts": dict(selected_counts.most_common()),
        "candidate_count": len(candidates),
        "top_k": args.top_k,
        "unique_symbols": args.unique_symbols,
        "candidates": [
            {
                "label": candidate.label,
                "path": str(candidate.path),
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "kind": candidate.kind,
            }
            for candidate in candidates
        ],
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
            "sizing_mode": args.sizing_mode,
        },
        "period": {
            "start": args.start,
            "end": args.end,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "min_train_trades": args.min_train_trades,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Adaptive WF saved to %s | %s/%s positive exp, mean expR %.3f, ddR %.2f",
        run_dir,
        aggregate.get("positive_expectancy_windows", 0),
        aggregate.get("windows", 0),
        aggregate.get("mean_expectancy_r", 0),
        aggregate.get("max_drawdown_r", 0),
    )


if __name__ == "__main__":
    main()
