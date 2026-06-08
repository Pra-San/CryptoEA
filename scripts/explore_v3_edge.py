#!/usr/bin/env python3
"""Explore v3 momentum-breakout parameters with OOS validation.

This script keeps the search objective on the training window, then validates
only the top training candidates on later data and harsher execution costs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.metrics import BacktestMetrics
from data.loader import DataLoader
from scripts.run_backtest import (
    MomentumBreakoutConfig,
    MomentumBreakoutStrategy,
    VectorizedBacktestConfig,
    VectorizedBacktestEngine,
    vectorized_preprocess,
)


logger = logging.getLogger("v3_edge_explorer")


BASE_FIELDS = [
    "total_trades",
    "win_rate",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "total_return_pct",
    "max_drawdown_pct",
    "max_drawdown_r",
    "avg_drawdown_r",
    "expectancy",
    "expectancy_r",
    "avg_win",
    "avg_loss",
    "avg_win_r",
    "avg_loss_r",
    "avg_r_multiple",
    "payoff_ratio",
    "trades_per_day",
    "avg_holding_bars",
    "max_consecutive_losses",
    "exposure_pct",
    "total_fees",
    "total_slippage",
]


PARAM_FIELDS = [
    "donchian_lookback",
    "volume_lookback",
    "volume_mult",
    "atr_stop_mult",
    "atr_tp_mult",
    "min_holding_bars",
    "max_holding_bars",
    "trend_sma_period",
    "volatility_min",
    "volatility_max",
    "use_trend_filter",
    "allow_long",
    "allow_short",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore corrected v3 edge")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--top-train", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--exit-on-flat-signal", action="store_true")
    parser.add_argument("--no-opposite-signal-exit", dest="exit_on_opposite_signal", action="store_false", default=True)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0010)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0020)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    loader = DataLoader(use_cache=True)
    df = loader.load_symbol(args.symbol, "1m")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    df = df[(df.index >= start) & (df.index <= end)]
    return vectorized_preprocess(df, args.timeframe)


def sample_params(rng: np.random.Generator, trial: int) -> dict[str, Any]:
    if trial == 0:
        return asdict(MomentumBreakoutConfig())

    min_hold = int(rng.integers(1, 7))
    max_hold = int(rng.choice([24, 36, 48, 72, 96, 120, 168, 240, 360]))
    if max_hold <= min_hold:
        max_hold = min_hold + 6

    side_mode = rng.choice(["both", "long_only", "short_only"], p=[0.35, 0.55, 0.10])

    return {
        "donchian_lookback": int(rng.choice([8, 10, 12, 16, 20, 24, 30, 40, 55, 80, 120, 160])),
        "volume_lookback": int(rng.choice([5, 10, 14, 20, 30, 50, 75, 100, 150, 200])),
        "volume_mult": float(rng.uniform(0.5, 3.2)),
        "atr_stop_mult": float(rng.uniform(0.5, 3.5)),
        "atr_tp_mult": float(rng.uniform(1.5, 14.0)),
        "min_holding_bars": min_hold,
        "max_holding_bars": max_hold,
        "trend_sma_period": int(rng.choice([50, 100, 150, 200, 300, 400, 600])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002, 0.003])),
        "volatility_max": float(rng.choice([0.03, 0.04, 0.05, 0.08, 0.12, 0.2])),
        "use_trend_filter": bool(rng.choice([True, True, True, False])),
        "allow_long": side_mode in {"both", "long_only"},
        "allow_short": side_mode in {"both", "short_only"},
    }


def make_strategy(symbol: str, timeframe: str, risk_per_trade: float, params: dict[str, Any]) -> MomentumBreakoutStrategy:
    config = MomentumBreakoutConfig(symbol=symbol, timeframe=timeframe, risk_per_trade=risk_per_trade)
    for key in PARAM_FIELDS:
        if key in params and hasattr(config, key):
            setattr(config, key, params[key])
    return MomentumBreakoutStrategy(config)


def result_object(result: dict[str, Any], bt_config: VectorizedBacktestConfig) -> SimpleNamespace:
    trades = [SimpleNamespace(**trade) for trade in result["trades"]]
    return SimpleNamespace(
        trades=trades,
        equity_curve=result["equity_curve"],
        benchmark_curve=result["benchmark_curve"],
        config=bt_config,
        metadata=result["metadata"],
    )


def run_metrics(
    featured: pd.DataFrame,
    symbol: str,
    balance: float,
    risk_per_trade: float,
    fee_rate: float,
    slippage_rate: float,
    max_position_pct: float,
    sizing_mode: str,
    exit_on_flat_signal: bool,
    exit_on_opposite_signal: bool,
    min_holding_bars: int,
    max_holding_bars: int,
) -> dict[str, float]:
    if len(featured) < max_holding_bars + 5:
        return BacktestMetrics().to_dict(BacktestMetrics().calculate(SimpleNamespace(trades=[])))

    bt_config = VectorizedBacktestConfig(
        initial_balance=balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=risk_per_trade,
        max_position_pct=max_position_pct,
        sizing_mode=sizing_mode,
        exit_on_flat_signal=exit_on_flat_signal,
        exit_on_opposite_signal=exit_on_opposite_signal,
        min_holding_bars=min_holding_bars,
        max_holding_bars=max_holding_bars,
    )
    engine = VectorizedBacktestEngine(featured, bt_config, symbol)
    result = engine.run()
    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate(result_object(result, bt_config), featured)
    return metrics_calc.to_dict(metrics)


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": metrics.get(key, 0) for key in BASE_FIELDS}


def train_score(metrics: dict[str, Any]) -> float:
    trades = metrics.get("total_trades", 0)
    if trades < 80:
        return -1_000_000 + trades
    pf = min(float(metrics.get("profit_factor", 0)), 5.0)
    exp_r = float(metrics.get("expectancy_r", 0))
    sharpe = float(metrics.get("sharpe_ratio", 0))
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    return exp_r * 140 + (pf - 1.0) * 28 + sharpe * 4 - dd_r * 0.15 - loss_streak * 0.8


def deploy_score(metrics: dict[str, Any], stress: dict[str, Any]) -> float:
    if metrics.get("total_trades", 0) < 30 or stress.get("total_trades", 0) < 30:
        return -1_000_000
    val = train_score(metrics)
    stressed = train_score(stress)
    penalty = max(0.0, -float(stress.get("expectancy_r", 0))) * 200
    return val * 0.65 + stressed * 0.35 - penalty


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"v3_edge_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s %s data", args.symbol, args.timeframe)
    base_df = load_data(args)
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    logger.info("Training search: %d candidates", args.trials)
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        strategy = make_strategy(args.symbol, args.timeframe, args.risk_per_trade, params)
        featured = strategy.initialize(base_df)
        train_metrics = run_metrics(
            featured.loc[featured.index <= train_end],
            args.symbol,
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
            "trial": trial,
            "train_score": train_score(train_metrics),
            **{key: getattr(strategy.config, key) for key in PARAM_FIELDS},
            **prefixed("train", train_metrics),
        }
        rows.append(row)
        if (trial + 1) % 50 == 0:
            best = max(rows, key=lambda item: item["train_score"])
            logger.info(
                "  %d/%d candidates, best train score %.2f PF %.2f expR %.3f ddR %.2f",
                trial + 1,
                args.trials,
                best["train_score"],
                best["train_profit_factor"],
                best["train_expectancy_r"],
                best["train_max_drawdown_r"],
            )

    rows.sort(key=lambda item: item["train_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["train_rank"] = rank

    logger.info("Validating top %d training candidates", min(args.top_train, len(rows)))
    top_rows = rows[: args.top_train]
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        strategy = make_strategy(args.symbol, args.timeframe, args.risk_per_trade, params)
        featured = strategy.initialize(base_df)
        validation_metrics = run_metrics(
            featured.loc[(featured.index >= val_start) & (featured.index <= val_end)],
            args.symbol,
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
        stress_metrics = run_metrics(
            featured.loc[(featured.index >= val_start) & (featured.index <= val_end)],
            args.symbol,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.stress_slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
            args.exit_on_flat_signal,
            args.exit_on_opposite_signal,
            strategy.config.min_holding_bars,
            strategy.config.max_holding_bars,
        )
        row.update(prefixed("validation", validation_metrics))
        row.update(prefixed("stress", stress_metrics))
        row["deploy_score"] = deploy_score(validation_metrics, stress_metrics)

    top_rows.sort(key=lambda item: item.get("deploy_score", -1_000_000), reverse=True)

    write_csv(run_dir / "training_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top_rows)

    best = top_rows[0] if top_rows else rows[0]
    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "period": {
            "start": args.start,
            "end": args.end,
            "train_end": args.train_end,
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
        },
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "stress_slippage_rate": args.stress_slippage_rate,
            "max_position_pct": args.max_position_pct,
            "sizing_mode": args.sizing_mode,
            "exit_on_flat_signal": args.exit_on_flat_signal,
            "exit_on_opposite_signal": args.exit_on_opposite_signal,
        },
        "seed": args.seed,
        "trials": args.trials,
        "top_train": args.top_train,
        "best": best,
    }
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info("Best validated candidate saved in %s", run_dir)
    logger.info(
        "Best deploy score %.2f | val PF %.2f expR %.3f ddR %.2f | stress PF %.2f expR %.3f ddR %.2f",
        best.get("deploy_score", 0),
        best.get("validation_profit_factor", 0),
        best.get("validation_expectancy_r", 0),
        best.get("validation_max_drawdown_r", 0),
        best.get("stress_profit_factor", 0),
        best.get("stress_expectancy_r", 0),
        best.get("stress_max_drawdown_r", 0),
    )


if __name__ == "__main__":
    main()
