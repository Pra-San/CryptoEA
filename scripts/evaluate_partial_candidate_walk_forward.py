#!/usr/bin/env python3
"""Adaptive walk-forward selector for partial/trailing high-winrate candidates."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explore_high_winrate_edge import deployment_gate, trades_per_week
from scripts.explore_trailing_highwin_edge import (
    PARAM_FIELDS,
    apply_strategy,
    load_data,
    metrics_from_trades,
    run_metrics,
    simulate_trades,
    summarize_windows,
    walk_forward_windows,
    write_csv,
)


logger = logging.getLogger("adaptive_partial_wf")


@dataclass(frozen=True)
class PartialCandidate:
    label: str
    source_run: str
    source_rank: int
    symbol: str
    timeframe: str
    params: dict[str, Any]


BOOL_FIELDS = {
    "allow_long",
    "allow_short",
    "exit_on_flat_signal",
    "move_stop_after_partial",
    "use_break_even",
    "use_mid_target",
    "use_partial_exit",
    "use_trailing_stop",
}

INT_FIELDS = {
    "bb_lookback",
    "donchian_lookback",
    "ema_fast",
    "ema_slow",
    "max_holding_bars",
    "min_holding_bars",
    "slope_bars",
    "volume_lookback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive WF selector for partial high-winrate candidates")
    parser.add_argument("--candidate-dir-glob", action="append", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--unique-symbols", action="store_true")
    parser.add_argument("--selector-mode", choices=["balanced", "gate"], default="balanced")
    parser.add_argument("--portfolio-risk-mode", choices=["equal_sleeve", "full_strategy"], default="equal_sleeve")
    parser.add_argument("--max-candidates-per-run", type=int, default=80)
    parser.add_argument("--min-train-trades", type=int, default=20)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--min-validation-trades", type=int, default=20)
    parser.add_argument("--min-trades-per-week", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def coerce_value(field: str, value: Any) -> Any:
    if field in BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)
    if field in INT_FIELDS:
        return int(value)
    if field == "family":
        return str(value)
    return float(value)


def candidate_dirs(patterns: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                paths[str(path.resolve())] = path
    return sorted(paths.values())


def load_candidates(args: argparse.Namespace) -> list[PartialCandidate]:
    candidates: list[PartialCandidate] = []
    for run_dir in candidate_dirs(args.candidate_dir_glob):
        summary_path = run_dir / "best_summary.json"
        csv_path = run_dir / "validated_top_candidates.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("strategy") != "partial_trailing_highwin_edge_search":
            continue
        rows = pd.read_csv(csv_path).head(args.max_candidates_per_run)
        for idx, row in rows.iterrows():
            params = {field: coerce_value(field, row[field]) for field in PARAM_FIELDS if field in row}
            rank = int(row.get("wf_rank", idx + 1))
            candidates.append(
                PartialCandidate(
                    label=f"{run_dir.name}#r{rank}",
                    source_run=run_dir.name,
                    source_rank=rank,
                    symbol=str(summary["symbol"]),
                    timeframe=str(summary["timeframe"]),
                    params=params,
                )
            )
    return candidates


def selector_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < args.min_train_trades:
        return -1_000_000 + trades
    win_rate = float(metrics.get("win_rate", 0))
    profit_factor = min(float(metrics.get("profit_factor", 0)), 6.0)
    expectancy_r = float(metrics.get("expectancy_r", 0))
    sharpe = min(float(metrics.get("sharpe_ratio", 0)), 6.0)
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    frequency = min(trades_per_week(metrics), args.min_trades_per_week * 4)
    if args.selector_mode == "gate":
        score = (
            win_rate * 80
            + expectancy_r * 240
            + (profit_factor - 1) * 42
            + sharpe * 8
            + frequency * 4
            - dd_r * 4
            - loss_streak * 5
        )
        score += 140 if deployment_gate(metrics, make_runtime_args(args)) else 0
        score -= max(0.0, 0.60 - win_rate) * 90
        score -= max(0.0, 2.50 - profit_factor) * 45
        score -= max(0.0, 0.15 - expectancy_r) * 260
        return score
    if expectancy_r <= 0 or profit_factor < 1.2:
        return win_rate * 35 + expectancy_r * 180 + (profit_factor - 1) * 20 - dd_r * 2 - loss_streak * 3
    return (
        win_rate * 85
        + expectancy_r * 180
        + (profit_factor - 1) * 35
        + sharpe * 8
        + frequency * 5
        - dd_r * 3
        - loss_streak * 4
    )


def make_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        min_validation_trades=args.min_validation_trades,
        min_trades_per_week=args.min_trades_per_week,
    )


def precompute_features(
    candidates: list[PartialCandidate],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    base_cache: dict[tuple[str, str], pd.DataFrame] = {}
    feature_cache: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        key = (candidate.symbol, candidate.timeframe)
        if key not in base_cache:
            base_cache[key] = load_data(candidate.symbol, candidate.timeframe, args.start, args.end)
        feature_cache[candidate.label] = apply_strategy(base_cache[key], candidate.params)
    return feature_cache


def aggregate_selected(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics_rows) == 1:
        return metrics_rows[0]
    total_trades = sum(int(row.get("total_trades", 0)) for row in metrics_rows)
    if total_trades == 0:
        return metrics_rows[0]
    weighted: dict[str, Any] = {}
    numeric_fields = {
        key
        for row in metrics_rows
        for key, value in row.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    for field in numeric_fields:
        weighted[field] = sum(float(row.get(field, 0)) * int(row.get("total_trades", 0)) for row in metrics_rows) / total_trades
    weighted["total_trades"] = total_trades
    weighted["winning_trades"] = sum(int(row.get("winning_trades", 0)) for row in metrics_rows)
    weighted["losing_trades"] = sum(int(row.get("losing_trades", 0)) for row in metrics_rows)
    weighted["win_rate"] = weighted["winning_trades"] / total_trades if total_trades else 0.0
    return weighted


def scale_trade(trade: Any, weight: float) -> SimpleNamespace:
    scaled = dict(vars(trade))
    for field in ["pnl", "notional", "risk_amount", "fees", "slippage", "quantity"]:
        scaled[field] = float(scaled.get(field, 0.0)) * weight
    scaled["r_multiple"] = float(scaled.get("r_multiple", 0.0)) * weight
    return SimpleNamespace(**scaled)


def select_candidates(
    scored: list[tuple[float, PartialCandidate, dict[str, Any]]],
    top_k: int,
    unique_symbols: bool,
) -> list[PartialCandidate]:
    if not unique_symbols:
        return [candidate for _, candidate, _ in scored[:top_k]]
    selected: list[PartialCandidate] = []
    symbols: set[str] = set()
    for _, candidate, _ in scored:
        if candidate.symbol in symbols:
            continue
        selected.append(candidate)
        symbols.add(candidate.symbol)
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
    logging.getLogger("data.loader").setLevel(logging.WARNING)
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)

    candidates = load_candidates(args)
    if not candidates:
        raise SystemExit("No partial candidates found.")
    runtime_args = make_runtime_args(args)
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"adaptive_partial_wf_top{args.top_k}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Precomputing %d candidate feature sets", len(candidates))
    feature_cache = precompute_features(candidates, args)
    windows = list(walk_forward_windows(args.start, args.end, args.train_months, args.test_months))
    window_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    logger.info("Running adaptive partial WF across %d windows", len(windows))
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        scored: list[tuple[float, PartialCandidate, dict[str, Any]]] = []
        for candidate in candidates:
            featured = feature_cache[candidate.label]
            train_slice = featured.loc[(featured.index >= train_start) & (featured.index <= train_end)]
            train_metrics = run_metrics(train_slice, candidate.symbol, runtime_args, candidate.params)
            score = selector_score(train_metrics, args)
            scored.append((score, candidate, train_metrics))
            ranking_rows.append({
                "window": i,
                "candidate": candidate.label,
                "source_run": candidate.source_run,
                "source_rank": candidate.source_rank,
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "family": candidate.params["family"],
                "train_score": score,
                "train_total_trades": train_metrics.get("total_trades", 0),
                "train_win_rate": train_metrics.get("win_rate", 0),
                "train_profit_factor": train_metrics.get("profit_factor", 0),
                "train_expectancy_r": train_metrics.get("expectancy_r", 0),
                "train_max_drawdown_r": train_metrics.get("max_drawdown_r", 0),
            })

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = select_candidates(scored, args.top_k, args.unique_symbols)
        all_trades = []
        timeline = pd.DatetimeIndex([])
        weight = 1.0 if args.portfolio_risk_mode == "full_strategy" else 1.0 / len(selected)
        for candidate in selected:
            featured = feature_cache[candidate.label]
            test_slice = featured.loc[(featured.index >= test_start) & (featured.index <= test_end)]
            timeline = pd.DatetimeIndex(timeline.union(test_slice.index))
            trades = simulate_trades(test_slice, candidate.symbol, runtime_args, candidate.params)
            all_trades.extend(scale_trade(trade, weight) for trade in trades)
        test_metrics = metrics_from_trades(all_trades, timeline, args.balance)
        window_rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            "selected": "|".join(candidate.label for candidate in selected),
            "gate_pass": deployment_gate(test_metrics, runtime_args),
            "trades_per_week": trades_per_week(test_metrics),
            **test_metrics,
        })
        logger.info(
            "  WF %02d selected %s | trades %s win %.1f%% PF %.2f expR %.3f ddR %.2f gate %s",
            i,
            ", ".join(f"{candidate.symbol}:{candidate.params['family']}#{candidate.source_rank}" for candidate in selected),
            test_metrics.get("total_trades", 0),
            float(test_metrics.get("win_rate", 0)) * 100,
            test_metrics.get("profit_factor", 0),
            test_metrics.get("expectancy_r", 0),
            test_metrics.get("max_drawdown_r", 0),
            window_rows[-1]["gate_pass"],
        )

    summary = {
        "symbol": "ADAPTIVE_PARTIAL",
        "timeframe": "mixed",
        "strategy": "adaptive_partial_trailing_selector",
        "candidate_count": len(candidates),
        "top_k": args.top_k,
        "selector_mode": args.selector_mode,
        "unique_symbols": args.unique_symbols,
        "portfolio_risk_mode": args.portfolio_risk_mode,
        "aggregate": summarize_windows(window_rows),
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
        },
    }
    write_csv(run_dir / "walk_forward.csv", window_rows)
    write_csv(run_dir / "candidate_rankings.csv", ranking_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Saved to %s | windows %s gate %s positive %s expR %.3f ddR %.2f trades %s",
        run_dir,
        summary["aggregate"].get("windows", 0),
        summary["aggregate"].get("gate_windows", 0),
        summary["aggregate"].get("positive_expectancy_windows", 0),
        summary["aggregate"].get("mean_expectancy_r", 0),
        summary["aggregate"].get("max_drawdown_r", 0),
        summary["aggregate"].get("total_trades", 0),
    )


if __name__ == "__main__":
    main()
