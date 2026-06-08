#!/usr/bin/env python3
"""Evaluate a fixed portfolio of saved strategy candidates.

Each candidate is run independently under the corrected vectorized engine, then
scaled into an equal-weight portfolio. R multiples are scaled by sleeve weight
so portfolio R drawdown reflects account-level risk contribution rather than
per-strategy local R.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from backtest.metrics import BacktestMetrics
from data.loader import DataLoader
from scripts.explore_strategy_families import PARAM_FIELDS as FAMILY_PARAM_FIELDS
from scripts.explore_strategy_families import apply_strategy
from scripts.explore_state_trend_following import PARAM_FIELDS as STATE_PARAM_FIELDS
from scripts.explore_state_trend_following import apply_strategy as apply_state_strategy
from scripts.explore_v3_edge import PARAM_FIELDS as V3_PARAM_FIELDS
from scripts.explore_v3_edge import make_strategy as make_v3_strategy
from scripts.run_backtest import (
    MomentumBreakoutConfig,
    VectorizedBacktestConfig,
    VectorizedBacktestEngine,
    vectorized_preprocess,
)


logger = logging.getLogger("strategy_portfolio")


@dataclass
class Candidate:
    label: str
    path: Path
    symbol: str
    timeframe: str
    kind: str
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a portfolio of saved candidates")
    parser.add_argument("--candidate", action="append", type=Path, required=True, help="Path to best_summary.json")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def load_candidate(path: Path) -> Candidate:
    data = json.loads(path.read_text())
    best = data["best"]
    symbol = data["symbol"]
    timeframe = data["timeframe"]
    strategy = data.get("strategy")
    family = best.get("family")
    if strategy == "state_trend_following":
        params = {field: best[field] for field in STATE_PARAM_FIELDS}
        params["exit_on_flat_signal"] = True
        kind = f"state_trend:{family}"
    elif family:
        params = {field: best[field] for field in FAMILY_PARAM_FIELDS}
        kind = f"family:{family}"
    else:
        defaults = MomentumBreakoutConfig()
        params = {field: best.get(field, getattr(defaults, field)) for field in V3_PARAM_FIELDS}
        kind = "v3"
    return Candidate(
        label=path.parent.name,
        path=path,
        symbol=symbol,
        timeframe=timeframe,
        kind=kind,
        params=params,
    )


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


def load_base_data(
    candidate: Candidate,
    start: str,
    end: str,
    cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    key = (candidate.symbol, candidate.timeframe)
    if key in cache:
        return cache[key]
    loader = DataLoader(use_cache=True)
    df = loader.load_symbol(candidate.symbol, "1m")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    base = vectorized_preprocess(df, candidate.timeframe)
    cache[key] = base
    return base


def apply_candidate_strategy(base: pd.DataFrame, candidate: Candidate, risk_per_trade: float) -> pd.DataFrame:
    if candidate.kind.startswith("state_trend:"):
        return apply_state_strategy(base, candidate.params)
    if candidate.kind.startswith("family:"):
        return apply_strategy(base, candidate.params)
    strategy = make_v3_strategy(candidate.symbol, candidate.timeframe, risk_per_trade, candidate.params)
    return strategy.initialize(base)


def run_candidate(
    candidate: Candidate,
    featured: pd.DataFrame,
    balance: float,
    risk_per_trade: float,
    fee_rate: float,
    slippage_rate: float,
    max_position_pct: float,
    sizing_mode: str,
) -> dict[str, Any]:
    if featured.empty:
        return {"trades": [], "equity_curve": pd.Series(dtype=float), "benchmark_curve": pd.Series(dtype=float)}
    min_hold = int(candidate.params["min_holding_bars"])
    max_hold = int(candidate.params["max_holding_bars"])
    config = VectorizedBacktestConfig(
        initial_balance=balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=risk_per_trade,
        max_position_pct=max_position_pct,
        sizing_mode=sizing_mode,
        min_holding_bars=min_hold,
        max_holding_bars=max_hold,
        exit_on_flat_signal=bool(candidate.params.get("exit_on_flat_signal", False)),
        exit_on_opposite_signal=True,
    )
    return VectorizedBacktestEngine(featured, config, candidate.symbol).run()


def scale_trade(trade: dict[str, Any], candidate: Candidate, weight: float) -> SimpleNamespace:
    scaled = dict(trade)
    scaled["candidate"] = candidate.label
    scaled["symbol"] = candidate.symbol
    scaled["timeframe"] = candidate.timeframe
    scaled["strategy_kind"] = candidate.kind
    scaled["weight"] = weight
    for field in ["pnl", "notional", "risk_amount", "fees", "slippage", "quantity"]:
        scaled[field] = float(scaled.get(field, 0.0)) * weight
    scaled["r_multiple"] = float(scaled.get("r_multiple", 0.0)) * weight
    return SimpleNamespace(**scaled)


def build_equity_curve(
    trades: list[SimpleNamespace],
    timeline: pd.DatetimeIndex,
    initial_balance: float,
) -> pd.Series:
    if len(timeline) == 0:
        exits = [pd.Timestamp(t.exit_time) for t in trades if getattr(t, "exit_time", None) is not None]
        timeline = pd.DatetimeIndex(sorted(set(exits)))
    if len(timeline) == 0:
        return pd.Series([initial_balance], index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]))
    pnl_by_time: dict[pd.Timestamp, float] = {}
    for trade in trades:
        exit_time = getattr(trade, "exit_time", None)
        if exit_time is None:
            continue
        ts = pd.Timestamp(exit_time)
        pnl_by_time[ts] = pnl_by_time.get(ts, 0.0) + float(trade.pnl)
    ordered_timeline = pd.DatetimeIndex(timeline).sort_values().unique()
    ordered_timeline = pd.DatetimeIndex(ordered_timeline)
    values = []
    equity = initial_balance
    for ts in ordered_timeline:
        equity += pnl_by_time.get(pd.Timestamp(ts), 0.0)
        values.append(equity)
    return pd.Series(values, index=ordered_timeline, name="equity")


def portfolio_result(
    trades: list[SimpleNamespace],
    timeline: pd.DatetimeIndex,
    balance: float,
) -> SimpleNamespace:
    equity = build_equity_curve(trades, timeline, balance)
    config = SimpleNamespace(initial_balance=balance)
    return SimpleNamespace(
        trades=trades,
        equity_curve=equity,
        benchmark_curve=None,
        config=config,
        metadata={
            "start_date": str(equity.index[0]) if len(equity) else "",
            "end_date": str(equity.index[-1]) if len(equity) else "",
        },
    )


def metrics_dict(trades: list[SimpleNamespace], timeline: pd.DatetimeIndex, balance: float) -> dict[str, Any]:
    calc = BacktestMetrics()
    metrics = calc.calculate(portfolio_result(trades, timeline, balance))
    return calc.to_dict(metrics)


def evaluate_period(
    candidates: list[Candidate],
    data_cache: dict[tuple[str, str], pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[SimpleNamespace], pd.Series]:
    weight = 1.0 / len(candidates)
    all_trades: list[SimpleNamespace] = []
    timelines: list[pd.DatetimeIndex] = []
    for candidate in candidates:
        base = load_base_data(candidate, args.start, args.end, data_cache)
        featured = apply_candidate_strategy(base, candidate, args.risk_per_trade)
        window = featured.loc[(featured.index >= start) & (featured.index <= end)]
        timelines.append(window.index)
        result = run_candidate(
            candidate,
            window,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
        )
        all_trades.extend(scale_trade(trade, candidate, weight) for trade in result.get("trades", []))
    timeline = pd.DatetimeIndex([])
    for index in timelines:
        timeline = timeline.union(index)
    metrics = metrics_dict(all_trades, timeline, args.balance)
    equity = build_equity_curve(all_trades, timeline, args.balance)
    return metrics, all_trades, equity


def summarize_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    total_trades = sum(row["total_trades"] for row in rows)
    weighted = lambda field: (
        sum(row[field] * row["total_trades"] for row in rows) / total_trades
        if total_trades
        else 0.0
    )
    return {
        "windows": len(rows),
        "profitable_windows": sum(1 for row in rows if row["total_return_pct"] > 0),
        "positive_expectancy_windows": sum(1 for row in rows if row["expectancy_r"] > 0),
        "mean_expectancy_r": sum(row["expectancy_r"] for row in rows) / len(rows),
        "trade_weighted_expectancy_r": weighted("expectancy_r"),
        "min_expectancy_r": min(row["expectancy_r"] for row in rows),
        "mean_profit_factor": sum(min(row["profit_factor"], 10.0) for row in rows) / len(rows),
        "min_profit_factor": min(row["profit_factor"] for row in rows),
        "mean_return_pct": sum(row["total_return_pct"] for row in rows) / len(rows),
        "min_return_pct": min(row["total_return_pct"] for row in rows),
        "max_drawdown_pct": max(row["max_drawdown_pct"] for row in rows),
        "max_drawdown_r": max(row["max_drawdown_r"] for row in rows),
        "max_consecutive_losses": max(row["max_consecutive_losses"] for row in rows),
        "total_trades": total_trades,
        "weighted_avg_win_r": weighted("avg_win_r"),
        "weighted_avg_loss_r": weighted("avg_loss_r"),
        "weighted_payoff_ratio": weighted("payoff_ratio"),
        "weighted_avg_holding_bars": weighted("avg_holding_bars"),
        "weighted_trades_per_day": weighted("trades_per_day"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    candidates = [load_candidate(path) for path in args.candidate]
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    label = "_".join(f"{candidate.symbol}{candidate.timeframe}" for candidate in candidates)
    digest = hashlib.sha1("|".join(str(candidate.path) for candidate in candidates).encode()).hexdigest()[:8]
    run_dir = output_root / f"portfolio_{label}_{digest}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cache: dict[tuple[str, str], pd.DataFrame] = {}
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    validation_metrics, validation_trades, validation_equity = evaluate_period(
        candidates, data_cache, validation_start, validation_end, args
    )

    rows: list[dict[str, Any]] = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.end, args.train_months, args.test_months),
        start=1,
    ):
        window_metrics, _, _ = evaluate_period(candidates, data_cache, test_start, test_end, args)
        rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            **window_metrics,
        })

    trades_rows = [vars(trade) for trade in validation_trades]
    write_csv(run_dir / "validation_trades.csv", trades_rows)
    validation_equity.rename("equity").reset_index().rename(columns={"index": "timestamp"}).to_csv(
        run_dir / "validation_equity.csv", index=False
    )
    write_csv(run_dir / "walk_forward.csv", rows)
    summary = {
        "symbol": "PORTFOLIO",
        "timeframe": "mixed",
        "strategy": "equal_weight_candidate_portfolio",
        "portfolio_metrics": validation_metrics,
        "aggregate": summarize_windows(rows),
        "candidates": [
            {
                "label": candidate.label,
                "path": str(candidate.path),
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "kind": candidate.kind,
                "weight": 1.0 / len(candidates),
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
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
            "train_months": args.train_months,
            "test_months": args.test_months,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info("Portfolio evaluation saved to %s", run_dir)
    logger.info(
        "Validation PF %.2f expR %.3f ddR %.2f trades %s | WF %s/%s positive exp, mean expR %.3f",
        validation_metrics.get("profit_factor", 0),
        validation_metrics.get("expectancy_r", 0),
        validation_metrics.get("max_drawdown_r", 0),
        validation_metrics.get("total_trades", 0),
        summary["aggregate"].get("positive_expectancy_windows", 0),
        summary["aggregate"].get("windows", 0),
        summary["aggregate"].get("mean_expectancy_r", 0),
    )


if __name__ == "__main__":
    main()
