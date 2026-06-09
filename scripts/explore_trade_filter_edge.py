#!/usr/bin/env python3
"""Optimize causal entry filters for saved strategy candidates."""

from __future__ import annotations

import argparse
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

from scripts.evaluate_prop_firm_constraints import load_candidate
from scripts.explore_high_winrate_edge import deployment_gate, trades_per_week, walk_forward_windows, write_csv
from scripts.explore_trailing_highwin_edge import load_data, metrics_from_trades, simulate_trades, summarize_windows
from scripts.explore_vwap_volume_profile_edge import apply_strategy as apply_vwap_strategy
from scripts.explore_vwap_volume_profile_edge import prepare_features as prepare_vwap_features
from scripts.explore_trailing_highwin_edge import apply_strategy as apply_partial_strategy


logger = logging.getLogger("trade_filter_explorer")


FILTER_FIELDS = [
    "side_mode",
    "hour_start",
    "hour_len",
    "dow_mode",
    "min_volume_ratio",
    "max_volume_ratio",
    "min_daily_vol",
    "max_daily_vol",
    "min_rsi",
    "max_rsi",
    "min_session_vwap_dist",
    "max_session_vwap_dist",
    "min_vwap_dist",
    "max_vwap_dist",
    "min_poc_dist",
    "max_poc_dist",
    "session_side",
    "poc_side",
]


DOW_MODES = {
    "all": None,
    "weekday": {0, 1, 2, 3, 4},
    "weekend": {5, 6},
    "mon_wed_fri": {0, 2, 4},
    "mon_fri_sun": {0, 4, 6},
    "not_tue_sat": {0, 2, 3, 4, 6},
    "not_weekend": {0, 1, 2, 3, 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize entry filters for a saved candidate")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-section", default="best_validation_ranked", choices=["best", "best_validation_ranked"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--top", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0010)
    parser.add_argument("--train-slippage-rate", type=float, default=0.0050)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0050)
    parser.add_argument("--min-train-trades", type=int, default=80)
    parser.add_argument("--min-validation-trades", type=int, default=60)
    parser.add_argument("--min-trades-per-week", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def runtime_args(args: argparse.Namespace, slippage_rate: float) -> SimpleNamespace:
    return SimpleNamespace(
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        min_validation_trades=args.min_validation_trades,
        min_trades_per_week=args.min_trades_per_week,
    )


def load_featured(args: argparse.Namespace) -> tuple[str, str, str, dict[str, Any], pd.DataFrame]:
    symbol, timeframe, strategy, candidate_name, params = load_candidate(args.candidate_summary, args.candidate_section)
    base = load_data(symbol, timeframe, args.start, args.validation_end)
    if strategy == "vwap_volume_profile_momentum_search":
        featured = apply_vwap_strategy(prepare_vwap_features(base, params, {}), params)
    else:
        featured = apply_partial_strategy(base, params)
    return symbol, timeframe, candidate_name, params, featured


def sample_filter(rng: np.random.Generator) -> dict[str, Any]:
    hour_len = int(rng.choice([24, 24, 6, 8, 10, 12, 16]))
    return {
        "side_mode": str(rng.choice(["both", "long", "short"], p=[0.45, 0.25, 0.30])),
        "hour_start": int(rng.integers(0, 24)),
        "hour_len": hour_len,
        "dow_mode": str(rng.choice(list(DOW_MODES), p=[0.42, 0.15, 0.04, 0.13, 0.14, 0.06, 0.06])),
        "min_volume_ratio": float(rng.choice([0.0, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0])),
        "max_volume_ratio": float(rng.choice([0.0, 1.5, 2.0, 3.0, 5.0])),
        "min_daily_vol": float(rng.choice([0.0, 0.015, 0.025, 0.035, 0.05, 0.065])),
        "max_daily_vol": float(rng.choice([0.0, 0.04, 0.055, 0.08, 0.12, 1.0])),
        "min_rsi": float(rng.choice([0.0, 25.0, 35.0, 45.0, 55.0])),
        "max_rsi": float(rng.choice([0.0, 45.0, 55.0, 65.0, 75.0, 100.0])),
        "min_session_vwap_dist": float(rng.choice([0.0, 0.0015, 0.003, 0.005, 0.0075, 0.012])),
        "max_session_vwap_dist": float(rng.choice([0.0, 0.006, 0.012, 0.025, 0.05])),
        "min_vwap_dist": float(rng.choice([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])),
        "max_vwap_dist": float(rng.choice([0.0, 0.06, 0.10, 0.16, 0.30])),
        "min_poc_dist": float(rng.choice([0.0, 0.005, 0.015, 0.025, 0.04, 0.07])),
        "max_poc_dist": float(rng.choice([0.0, 0.02, 0.04, 0.08, 0.16, 0.30])),
        "session_side": str(rng.choice(["any", "above", "below"], p=[0.55, 0.22, 0.23])),
        "poc_side": str(rng.choice(["any", "above", "below"], p=[0.55, 0.22, 0.23])),
    }


def hours_mask(index: pd.DatetimeIndex, start: int, length: int) -> np.ndarray:
    if length >= 24:
        return np.ones(len(index), dtype=bool)
    hours = index.hour.to_numpy()
    end = (start + length) % 24
    if start < end:
        return (hours >= start) & (hours < end)
    return (hours >= start) | (hours < end)


def optional_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return df[name]
    return pd.Series(default, index=df.index, dtype=float)


def apply_filter(featured: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    df = featured.copy()
    signal = df["signal"].astype(int)
    keep = signal != 0

    if params["side_mode"] == "long":
        keep &= signal > 0
    elif params["side_mode"] == "short":
        keep &= signal < 0

    keep &= hours_mask(df.index, int(params["hour_start"]), int(params["hour_len"]))
    allowed_days = DOW_MODES[str(params["dow_mode"])]
    if allowed_days is not None:
        keep &= pd.Series(df.index.dayofweek, index=df.index).isin(allowed_days)

    volume_ratio = optional_column(df, "volume_ratio_vp", 1.0)
    if "volume_ratio" in df:
        volume_ratio = volume_ratio.where(df.get("volume_ratio_vp", pd.Series(np.nan, index=df.index)).notna(), df["volume_ratio"])
    if params["min_volume_ratio"] > 0:
        keep &= volume_ratio >= float(params["min_volume_ratio"])
    if params["max_volume_ratio"] > 0:
        keep &= volume_ratio <= float(params["max_volume_ratio"])

    if params["min_daily_vol"] > 0:
        keep &= df["daily_vol"] >= float(params["min_daily_vol"])
    if params["max_daily_vol"] > 0:
        keep &= df["daily_vol"] <= float(params["max_daily_vol"])
    if params["min_rsi"] > 0:
        keep &= df["rsi_14"] >= float(params["min_rsi"])
    if params["max_rsi"] > 0:
        keep &= df["rsi_14"] <= float(params["max_rsi"])

    close = df["close"]
    if "session_vwap" in df:
        session_dist = ((close - df["session_vwap"]).abs() / close).fillna(1.0)
        if params["min_session_vwap_dist"] > 0:
            keep &= session_dist >= float(params["min_session_vwap_dist"])
        if params["max_session_vwap_dist"] > 0:
            keep &= session_dist <= float(params["max_session_vwap_dist"])
        if params["session_side"] == "above":
            keep &= close > df["session_vwap"]
        elif params["session_side"] == "below":
            keep &= close < df["session_vwap"]

    if "vwap" in df:
        vwap_dist = ((close - df["vwap"]).abs() / close).fillna(1.0)
        if params["min_vwap_dist"] > 0:
            keep &= vwap_dist >= float(params["min_vwap_dist"])
        if params["max_vwap_dist"] > 0:
            keep &= vwap_dist <= float(params["max_vwap_dist"])

    if "profile_poc" in df:
        poc_dist = ((close - df["profile_poc"]).abs() / close).fillna(1.0)
        if params["min_poc_dist"] > 0:
            keep &= poc_dist >= float(params["min_poc_dist"])
        if params["max_poc_dist"] > 0:
            keep &= poc_dist <= float(params["max_poc_dist"])
        if params["poc_side"] == "above":
            keep &= close > df["profile_poc"]
        elif params["poc_side"] == "below":
            keep &= close < df["profile_poc"]

    df.loc[~keep, "signal"] = 0
    return df


def metrics_for(
    filtered: pd.DataFrame,
    symbol: str,
    source_params: dict[str, Any],
    args: argparse.Namespace,
    slippage_rate: float,
) -> dict[str, Any]:
    rt_args = runtime_args(args, slippage_rate)
    trades = simulate_trades(filtered, symbol, rt_args, source_params)
    return metrics_from_trades(trades, filtered.index, args.balance)


def score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 4:
        return -1000 + trades
    win = float(metrics.get("win_rate", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 8.0)
    exp = float(metrics.get("expectancy_r", 0))
    sharpe = min(float(metrics.get("sharpe_ratio", 0)), 6.0)
    dd = float(metrics.get("max_drawdown_r", 0))
    streak = float(metrics.get("max_consecutive_losses", 0))
    freq = min(trades_per_week(metrics), args.min_trades_per_week * 4)
    return win * 90 + exp * 260 + (pf - 1.0) * 42 + sharpe * 8 + freq * 5 - dd * 4 - streak * 5


def aggregate_score(window_metrics: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    total_trades = sum(int(item.get("total_trades", 0)) for item in window_metrics)
    if total_trades < args.min_train_trades:
        return -1_000_000 + total_trades, {}
    aggregate = {
        "wf_windows": len(window_metrics),
        "wf_gate_windows": sum(1 for item in window_metrics if deployment_gate(item, args)),
        "wf_positive_exp": sum(1 for item in window_metrics if float(item.get("expectancy_r", 0)) > 0),
        "wf_mean_win_rate": sum(float(item.get("win_rate", 0)) for item in window_metrics) / len(window_metrics),
        "wf_mean_exp_r": sum(float(item.get("expectancy_r", 0)) for item in window_metrics) / len(window_metrics),
        "wf_mean_pf": sum(min(float(item.get("profit_factor", 0)), 10.0) for item in window_metrics) / len(window_metrics),
        "wf_max_dd_r": max(float(item.get("max_drawdown_r", 0)) for item in window_metrics),
        "wf_total_trades": total_trades,
    }
    total_score = (
        sum(score(item, args) for item in window_metrics) / len(window_metrics)
        + aggregate["wf_gate_windows"] / len(window_metrics) * 220
        + aggregate["wf_positive_exp"] / len(window_metrics) * 100
        + aggregate["wf_mean_exp_r"] * 160
    )
    return total_score, aggregate


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    symbol, timeframe, candidate_name, candidate_params, featured = load_featured(args)
    rng = np.random.default_rng(args.seed)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"trade_filter_{symbol}_{timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    logger.info("Searching %d filters for %s", args.trials, candidate_name)
    for trial in range(args.trials):
        params = sample_filter(rng)
        filtered = apply_filter(featured, params)
        metrics = [
            metrics_for(
                filtered.loc[(filtered.index >= test_start) & (filtered.index <= test_end)],
                symbol,
                candidate_params,
                args,
                args.train_slippage_rate,
            )
            for _, _, test_start, test_end in train_windows
        ]
        wf_score, aggregate = aggregate_score(metrics, args)
        rows.append({"trial": trial, "wf_score": wf_score, **params, **aggregate})
        if (trial + 1) % 100 == 0:
            best = max(rows, key=lambda item: item["wf_score"])
            logger.info(
                "  %d/%d best score %.2f gates %s/%s win %.1f%% expR %.3f PF %.2f ddR %.2f trades %s",
                trial + 1,
                args.trials,
                best.get("wf_score", 0),
                best.get("wf_gate_windows", 0),
                best.get("wf_windows", 0),
                best.get("wf_mean_win_rate", 0) * 100,
                best.get("wf_mean_exp_r", 0),
                best.get("wf_mean_pf", 0),
                best.get("wf_max_dd_r", 0),
                best.get("wf_total_trades", 0),
            )

    rows.sort(key=lambda item: item["wf_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["wf_rank"] = rank

    top_rows = rows[: args.top]
    for row in top_rows:
        params = {field: row[field] for field in FILTER_FIELDS}
        filtered = apply_filter(featured, params)
        validation_slice = filtered.loc[(filtered.index >= val_start) & (filtered.index <= val_end)]
        validation = metrics_for(validation_slice, symbol, candidate_params, args, args.slippage_rate)
        stress = metrics_for(validation_slice, symbol, candidate_params, args, args.stress_slippage_rate)
        for prefix, metrics in [("validation", validation), ("stress", stress)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_trades_per_week"] = trades_per_week(metrics)
        row["validation_pass"] = deployment_gate(validation, args)
        row["stress_pass"] = deployment_gate(stress, args)
        row["deploy_score"] = (
            (650 if row["stress_pass"] else 0)
            + (250 if row["validation_pass"] else 0)
            + row["stress_win_rate"] * 120
            + min(row["stress_profit_factor"], 10.0) * 42
            + row["stress_expectancy_r"] * 260
            + min(row["stress_sharpe_ratio"], 6.0) * 10
            - row["stress_max_drawdown_r"] * 5
            - row["stress_max_consecutive_losses"] * 6
            + row["wf_score"] * 0.35
        )

    top_rows.sort(key=lambda item: item["deploy_score"], reverse=True)
    best = top_rows[0] if top_rows else rows[0]
    best_params = {field: best[field] for field in FILTER_FIELDS}
    best_filtered = apply_filter(featured, best_params)
    wf_rows = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.validation_end, args.train_months, args.test_months),
        start=1,
    ):
        metrics = metrics_for(
            best_filtered.loc[(best_filtered.index >= test_start) & (best_filtered.index <= test_end)],
            symbol,
            candidate_params,
            args,
            args.train_slippage_rate,
        )
        wf_rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            "gate_pass": deployment_gate(metrics, args),
            "trades_per_week": trades_per_week(metrics),
            **metrics,
        })

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": "trade_filter_edge_search",
        "source_candidate_summary": str(args.candidate_summary),
        "source_candidate_section": args.candidate_section,
        "source_candidate_name": candidate_name,
        "seed": args.seed,
        "trials": args.trials,
        "best": best,
        "best_validation_ranked": top_rows[0] if top_rows else {},
        "aggregate": summarize_windows(wf_rows),
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "stress_slippage_rate": args.stress_slippage_rate,
            "train_slippage_rate": args.train_slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
        },
    }
    write_csv(run_dir / "training_filters.csv", rows)
    write_csv(run_dir / "validated_top_filters.csv", top_rows)
    write_csv(run_dir / "walk_forward.csv", wf_rows)
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Saved to %s | stress pass %s win %.1f%% PF %.2f expR %.3f ddR %.2f trades/week %.2f WF gates %s/%s",
        run_dir,
        best.get("stress_pass", False),
        best.get("stress_win_rate", 0) * 100,
        best.get("stress_profit_factor", 0),
        best.get("stress_expectancy_r", 0),
        best.get("stress_max_drawdown_r", 0),
        best.get("stress_trades_per_week", 0),
        summary["aggregate"].get("gate_windows", 0),
        summary["aggregate"].get("windows", 0),
    )


if __name__ == "__main__":
    main()
