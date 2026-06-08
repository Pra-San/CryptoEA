#!/usr/bin/env python3
"""Explore state-based trend following with flat-signal exits.

Unlike the entry-event strategy families, these candidates stay in a position
while a trend state remains active and exit when the state turns flat/opposite.
The search uses walk-forward training windows before holdout validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.metrics import BacktestMetrics
from data.loader import DataLoader
from scripts.explore_strategy_families import BASE_FIELDS, prefixed
from scripts.run_backtest import VectorizedBacktestConfig, VectorizedBacktestEngine, vectorized_preprocess


logger = logging.getLogger("state_trend_explorer")


PARAM_FIELDS = [
    "family",
    "allow_long",
    "allow_short",
    "atr_stop_mult",
    "atr_tp_mult",
    "ema_fast",
    "ema_slow",
    "max_holding_bars",
    "min_holding_bars",
    "slope_bars",
    "slope_min",
    "sma_period",
    "use_take_profit",
    "volatility_max",
    "volatility_min",
]

FAMILIES = ["sma_state", "ema_state", "sma_slope_state"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore state trend-following candidates")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--selection-mode", choices=["deploy", "wf"], default="deploy")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--min-total-trades", type=int, default=20)
    parser.add_argument("--min-validation-trades", type=int, default=8)
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
    raw = loader.load_symbol(symbol, "1m")
    if raw.index.tz is None:
        raw = raw.tz_localize("UTC")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    raw = raw[(raw.index >= start_ts) & (raw.index <= end_ts)]
    return vectorized_preprocess(raw, timeframe)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(int(period), 1), adjust=False, min_periods=1).mean()


def sample_params(rng: np.random.Generator, trial: int) -> dict[str, Any]:
    family = FAMILIES[trial % len(FAMILIES)] if trial < len(FAMILIES) else str(rng.choice(FAMILIES))
    side_mode = str(rng.choice(["long_only", "both", "short_only"], p=[0.70, 0.25, 0.05]))
    min_hold = int(rng.choice([1, 2, 3, 4, 6]))
    max_hold = int(rng.choice([48, 72, 120, 168, 240, 360, 540, 720]))
    if max_hold <= min_hold:
        max_hold = min_hold + 12
    ema_fast = int(rng.choice([10, 20, 30, 50, 80, 100]))
    ema_slow = int(rng.choice([80, 100, 150, 200, 300, 400]))
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 50
    return {
        "family": family,
        "allow_long": side_mode in {"long_only", "both"},
        "allow_short": side_mode in {"short_only", "both"},
        "atr_stop_mult": float(rng.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])),
        "atr_tp_mult": float(rng.choice([4.0, 6.0, 8.0, 12.0, 16.0, 24.0])),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "max_holding_bars": max_hold,
        "min_holding_bars": min_hold,
        "slope_bars": int(rng.choice([6, 12, 18, 24, 36, 48])),
        "slope_min": float(rng.choice([0.0, 0.001, 0.0025, 0.005, 0.01, 0.02])),
        "sma_period": int(rng.choice([50, 80, 100, 150, 200, 300, 400])),
        "use_take_profit": bool(rng.choice([False, True], p=[0.65, 0.35])),
        "volatility_max": float(rng.choice([0.04, 0.05, 0.08, 0.12, 0.20, 1.00])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002, 0.003])),
    }


def apply_strategy(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    featured = df.copy()
    close = featured["close"]
    atr = featured["atr_14"].replace(0, np.nan)
    volatility = featured["daily_vol"].fillna(0)
    vol_ok = (volatility >= float(params["volatility_min"])) & (volatility <= float(params["volatility_max"]))
    signal = pd.Series(0, index=featured.index, dtype=int)

    family = str(params["family"])
    if family == "sma_state":
        sma = close.rolling(int(params["sma_period"]), min_periods=max(10, int(params["sma_period"]) // 4)).mean()
        long_state = close > sma
        short_state = close < sma
    elif family == "ema_state":
        fast = ema(close, int(params["ema_fast"]))
        slow = ema(close, int(params["ema_slow"]))
        long_state = fast > slow
        short_state = fast < slow
    elif family == "sma_slope_state":
        sma = close.rolling(int(params["sma_period"]), min_periods=max(10, int(params["sma_period"]) // 4)).mean()
        slope = sma.pct_change(int(params["slope_bars"]))
        long_state = (close > sma) & (slope > float(params["slope_min"]))
        short_state = (close < sma) & (slope < -float(params["slope_min"]))
    else:
        raise ValueError(f"Unknown family: {family}")

    signal.loc[long_state & vol_ok & bool(params["allow_long"])] = 1
    signal.loc[short_state & vol_ok & bool(params["allow_short"])] = -1

    stop_loss = pd.Series(np.nan, index=featured.index, dtype=float)
    take_profit = pd.Series(np.nan, index=featured.index, dtype=float)
    long_signal = signal == 1
    short_signal = signal == -1
    stop_loss.loc[long_signal] = close - float(params["atr_stop_mult"]) * atr
    stop_loss.loc[short_signal] = close + float(params["atr_stop_mult"]) * atr
    if bool(params["use_take_profit"]):
        take_profit.loc[long_signal] = close + float(params["atr_tp_mult"]) * atr
        take_profit.loc[short_signal] = close - float(params["atr_tp_mult"]) * atr

    featured["signal"] = signal
    featured["stop_loss"] = stop_loss
    featured["take_profit"] = take_profit
    return featured.dropna(subset=["atr_14", "daily_vol"])


def result_object(result: dict[str, Any], config: VectorizedBacktestConfig) -> SimpleNamespace:
    return SimpleNamespace(
        trades=[SimpleNamespace(**trade) for trade in result["trades"]],
        equity_curve=result["equity_curve"],
        benchmark_curve=result["benchmark_curve"],
        config=config,
        metadata=result["metadata"],
    )


def run_metrics(
    featured: pd.DataFrame,
    symbol: str,
    args: argparse.Namespace,
    params: dict[str, Any],
) -> dict[str, Any]:
    if len(featured) < int(params["max_holding_bars"]) + 5:
        return BacktestMetrics().to_dict(BacktestMetrics().calculate(SimpleNamespace(trades=[])))
    config = VectorizedBacktestConfig(
        initial_balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        sizing_mode=args.sizing_mode,
        min_holding_bars=int(params["min_holding_bars"]),
        max_holding_bars=int(params["max_holding_bars"]),
        exit_on_flat_signal=True,
        exit_on_opposite_signal=True,
    )
    result = VectorizedBacktestEngine(featured, config, symbol).run()
    calc = BacktestMetrics()
    metrics = calc.calculate(result_object(result, config), featured)
    return calc.to_dict(metrics)


def window_score(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 2:
        return -40.0
    return (
        float(metrics.get("expectancy_r", 0)) * 95
        + min(float(metrics.get("profit_factor", 0)), 4.0) * 10
        - float(metrics.get("max_drawdown_r", 0)) * 1.7
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
    total_trades = sum(trade_values)
    if total_trades < min_total_trades:
        return -1_000_000 + total_trades, {}
    aggregate = {
        "wf_windows": len(window_metrics),
        "wf_positive_exp": sum(1 for value in exp_values if value > 0),
        "wf_profitable": sum(1 for value in ret_values if value > 0),
        "wf_mean_exp_r": sum(exp_values) / len(exp_values),
        "wf_min_exp_r": min(exp_values),
        "wf_mean_pf": sum(min(value, 10.0) for value in pf_values) / len(pf_values),
        "wf_min_pf": min(pf_values),
        "wf_mean_return_pct": sum(ret_values) / len(ret_values),
        "wf_min_return_pct": min(ret_values),
        "wf_max_dd_r": max(dd_values),
        "wf_total_trades": total_trades,
    }
    per_window = [window_score(item) for item in window_metrics]
    score = (
        sum(per_window) / len(per_window)
        + aggregate["wf_positive_exp"] / len(window_metrics) * 110
        + aggregate["wf_profitable"] / len(window_metrics) * 45
        + aggregate["wf_min_exp_r"] * 70
        - aggregate["wf_max_dd_r"] * 2.0
    )
    return score, aggregate


def validation_rank(row: dict[str, Any], min_validation_trades: int) -> float:
    if int(row.get("validation_total_trades", 0)) < min_validation_trades:
        return -1_000_000 + int(row.get("validation_total_trades", 0))
    return (
        float(row.get("validation_expectancy_r", 0)) * 130
        + min(float(row.get("validation_profit_factor", 0)), 4.0) * 22
        - float(row.get("validation_max_drawdown_r", 0)) * 2.8
        + float(row.get("wf_score", 0)) * 0.45
    )


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

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"state_trend_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    logger.info("Searching %d state-trend candidates across %d train windows", args.trials, len(train_windows))
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        featured = apply_strategy(base, params)
        window_metrics = [
            run_metrics(
                featured.loc[(featured.index >= test_start) & (featured.index <= test_end)],
                args.symbol,
                args,
                params,
            )
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
                "  %d/%d best %s score %.2f pos %s/%s mean expR %.3f ddR %.2f trades %s",
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

    top_rows = rows[: args.top]
    logger.info("Validating top %d candidates on holdout", len(top_rows))
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        featured = apply_strategy(base, params)
        validation = run_metrics(
            featured.loc[(featured.index >= validation_start) & (featured.index <= validation_end)],
            args.symbol,
            args,
            params,
        )
        row.update(prefixed("validation", validation))
        row["deploy_score"] = validation_rank(row, args.min_validation_trades)

    if args.selection_mode == "deploy":
        top_rows.sort(key=lambda item: item.get("deploy_score", -1_000_000), reverse=True)
        best = top_rows[0] if top_rows else rows[0]
    else:
        best = rows[0]

    best_params = {field: best[field] for field in PARAM_FIELDS}
    best_featured = apply_strategy(base, best_params)
    wf_rows: list[dict[str, Any]] = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.validation_end, args.train_months, args.test_months),
        start=1,
    ):
        metrics = run_metrics(
            best_featured.loc[(best_featured.index >= test_start) & (best_featured.index <= test_end)],
            args.symbol,
            args,
            best_params,
        )
        wf_rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            **metrics,
        })

    write_csv(run_dir / "training_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top_rows)
    write_csv(run_dir / "walk_forward.csv", wf_rows)
    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strategy": "state_trend_following",
        "period": {
            "start": args.start,
            "train_end": args.train_end,
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
            "train_months": args.train_months,
            "test_months": args.test_months,
        },
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
            "sizing_mode": args.sizing_mode,
        },
        "seed": args.seed,
        "selection_mode": args.selection_mode,
        "trials": args.trials,
        "top": args.top,
        "best": best,
        "aggregate": {
            "windows": len(wf_rows),
            "positive_expectancy_windows": sum(1 for row in wf_rows if row["expectancy_r"] > 0),
            "profitable_windows": sum(1 for row in wf_rows if row["total_return_pct"] > 0),
            "mean_expectancy_r": sum(row["expectancy_r"] for row in wf_rows) / len(wf_rows) if wf_rows else 0,
            "trade_weighted_expectancy_r": (
                sum(row["expectancy_r"] * row["total_trades"] for row in wf_rows)
                / sum(row["total_trades"] for row in wf_rows)
                if sum(row["total_trades"] for row in wf_rows)
                else 0
            ),
            "max_drawdown_r": max((row["max_drawdown_r"] for row in wf_rows), default=0),
            "max_drawdown_pct": max((row["max_drawdown_pct"] for row in wf_rows), default=0),
            "total_trades": sum(row["total_trades"] for row in wf_rows),
        },
    }
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info("State trend search saved in %s", run_dir)
    logger.info(
        "Best %s | validation PF %.2f expR %.3f ddR %.2f trades %s | fixed WF %s/%s mean expR %.3f ddR %.2f",
        best.get("family"),
        best.get("validation_profit_factor", 0),
        best.get("validation_expectancy_r", 0),
        best.get("validation_max_drawdown_r", 0),
        best.get("validation_total_trades", 0),
        summary["aggregate"]["positive_expectancy_windows"],
        summary["aggregate"]["windows"],
        summary["aggregate"]["mean_expectancy_r"],
        summary["aggregate"]["max_drawdown_r"],
    )


if __name__ == "__main__":
    main()
