#!/usr/bin/env python3
"""Search high-winrate OHLCV strategies under corrected execution.

The objective is intentionally strict: high win rate, strong profit factor,
positive expectancy in R, low R drawdown, short loss streaks, and enough trade
frequency after fees and adverse slippage. Candidates are ranked on rolling
training windows; validation and stress results are reported separately.
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
from scripts.explore_strategy_families import BASE_FIELDS, prefixed, rolling_bb
from scripts.run_backtest import VectorizedBacktestConfig, VectorizedBacktestEngine, vectorized_preprocess


logger = logging.getLogger("high_winrate_explorer")


PARAM_FIELDS = [
    "family",
    "allow_long",
    "allow_short",
    "atr_stop_mult",
    "atr_tp_mult",
    "bb_lookback",
    "bb_std",
    "donchian_lookback",
    "ema_fast",
    "ema_slow",
    "exit_on_flat_signal",
    "max_holding_bars",
    "min_holding_bars",
    "pullback_depth",
    "rsi_high",
    "rsi_low",
    "slope_bars",
    "slope_min",
    "trend_strength_max",
    "use_mid_target",
    "volatility_max",
    "volatility_min",
    "volume_lookback",
    "volume_min",
]

FAMILIES = [
    "trend_reclaim",
    "trend_bb_reversion",
    "range_bb_reversion",
    "trend_state_tp",
    "channel_reclaim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore high-winrate strategy variants")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0050)
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-validation-trades", type=int, default=80)
    parser.add_argument("--min-trades-per-week", type=float, default=2.0)
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
    side_mode = str(rng.choice(["long_only", "both", "short_only"], p=[0.72, 0.23, 0.05]))
    ema_fast = int(rng.choice([8, 10, 14, 20, 30, 50, 80]))
    ema_slow = int(rng.choice([50, 80, 100, 150, 200, 300, 400]))
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 50
    min_hold = int(rng.choice([1, 2, 3, 4, 6, 8]))
    max_hold = int(rng.choice([8, 12, 18, 24, 36, 48, 72, 96, 120]))
    if max_hold <= min_hold:
        max_hold = min_hold + 6
    return {
        "family": family,
        "allow_long": side_mode in {"long_only", "both"},
        "allow_short": side_mode in {"short_only", "both"},
        "atr_stop_mult": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])),
        "atr_tp_mult": float(rng.choice([1.3, 1.7, 2.2, 3.0, 4.0, 6.0, 8.0, 12.0])),
        "bb_lookback": int(rng.choice([14, 20, 30, 40, 55, 80])),
        "bb_std": float(rng.choice([1.5, 1.8, 2.0, 2.2, 2.5])),
        "donchian_lookback": int(rng.choice([8, 10, 12, 16, 20, 30, 40, 55])),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "exit_on_flat_signal": family == "trend_state_tp",
        "max_holding_bars": max_hold,
        "min_holding_bars": min_hold,
        "pullback_depth": float(rng.choice([0.0, 0.0025, 0.005, 0.008, 0.012, 0.018, 0.025])),
        "rsi_high": float(rng.uniform(58.0, 78.0)),
        "rsi_low": float(rng.uniform(22.0, 45.0)),
        "slope_bars": int(rng.choice([3, 6, 12, 18, 24, 36, 48])),
        "slope_min": float(rng.choice([0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01])),
        "trend_strength_max": float(rng.choice([0.006, 0.01, 0.015, 0.025, 0.04, 0.08])),
        "use_mid_target": bool(rng.choice([False, True], p=[0.85, 0.15])),
        "volatility_max": float(rng.choice([0.03, 0.04, 0.05, 0.08, 0.12, 0.20, 1.00])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002, 0.003])),
        "volume_lookback": int(rng.choice([10, 20, 30, 50, 75, 100, 150])),
        "volume_min": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])),
    }


def common_masks(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, pd.Series]:
    close = df["close"]
    volume = df["volume"]
    fast = ema(close, int(params["ema_fast"]))
    slow = ema(close, int(params["ema_slow"]))
    slope = slow.pct_change(int(params["slope_bars"]))
    trend_strength = ((fast - slow).abs() / slow.replace(0, np.nan)).fillna(0)
    vol_sma = volume.rolling(int(params["volume_lookback"]), min_periods=1).mean()
    volume_ok = (volume / vol_sma.replace(0, np.nan)).fillna(0) >= float(params["volume_min"])
    volatility = df["daily_vol"].fillna(0)
    volatility_ok = (volatility >= float(params["volatility_min"])) & (volatility <= float(params["volatility_max"]))
    return {
        "fast": fast,
        "slow": slow,
        "uptrend": (fast > slow) & (slope > float(params["slope_min"])),
        "downtrend": (fast < slow) & (slope < -float(params["slope_min"])),
        "range_ok": trend_strength <= float(params["trend_strength_max"]),
        "volume_ok": volume_ok,
        "volatility_ok": volatility_ok,
    }


def apply_strategy(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    featured = df.copy()
    close = featured["close"]
    high = featured["high"]
    low = featured["low"]
    atr = featured["atr_14"].replace(0, np.nan)
    masks = common_masks(featured, params)
    mid, upper, lower, _ = rolling_bb(close, int(params["bb_lookback"]), float(params["bb_std"]))
    signal = pd.Series(0, index=featured.index, dtype=int)
    stop_loss = pd.Series(np.nan, index=featured.index, dtype=float)
    take_profit = pd.Series(np.nan, index=featured.index, dtype=float)

    family = str(params["family"])
    if family == "trend_reclaim":
        long_event = (
            masks["uptrend"]
            & (close.shift(1) < masks["fast"].shift(1) * (1 - float(params["pullback_depth"])))
            & (close > masks["fast"])
            & (featured["rsi_14"] >= 42)
        )
        short_event = (
            masks["downtrend"]
            & (close.shift(1) > masks["fast"].shift(1) * (1 + float(params["pullback_depth"])))
            & (close < masks["fast"])
            & (featured["rsi_14"] <= 58)
        )
    elif family == "trend_bb_reversion":
        long_event = (
            masks["uptrend"]
            & (low <= lower)
            & (close > lower)
            & (featured["rsi_14"] <= float(params["rsi_low"]) + 10)
        )
        short_event = (
            masks["downtrend"]
            & (high >= upper)
            & (close < upper)
            & (featured["rsi_14"] >= float(params["rsi_high"]) - 10)
        )
    elif family == "range_bb_reversion":
        long_event = masks["range_ok"] & (close < lower) & (featured["rsi_14"] <= float(params["rsi_low"]))
        short_event = masks["range_ok"] & (close > upper) & (featured["rsi_14"] >= float(params["rsi_high"]))
    elif family == "trend_state_tp":
        long_event = masks["uptrend"] & (close > masks["fast"]) & (featured["rsi_14"] >= 45)
        short_event = masks["downtrend"] & (close < masks["fast"]) & (featured["rsi_14"] <= 55)
    elif family == "channel_reclaim":
        channel_low = low.rolling(int(params["donchian_lookback"]), min_periods=1).min().shift(1)
        channel_high = high.rolling(int(params["donchian_lookback"]), min_periods=1).max().shift(1)
        long_event = masks["uptrend"] & (low.shift(1) <= channel_low.shift(1)) & (close > channel_low)
        short_event = masks["downtrend"] & (high.shift(1) >= channel_high.shift(1)) & (close < channel_high)
    else:
        raise ValueError(f"Unknown family: {family}")

    long_event = long_event & masks["volume_ok"] & masks["volatility_ok"] & bool(params["allow_long"])
    short_event = short_event & masks["volume_ok"] & masks["volatility_ok"] & bool(params["allow_short"])
    signal.loc[long_event] = 1
    signal.loc[short_event] = -1

    stop_loss.loc[long_event] = close - float(params["atr_stop_mult"]) * atr
    stop_loss.loc[short_event] = close + float(params["atr_stop_mult"]) * atr
    if bool(params["use_mid_target"]):
        take_profit.loc[long_event] = mid.loc[long_event].where(mid.loc[long_event] > close.loc[long_event])
        take_profit.loc[short_event] = mid.loc[short_event].where(mid.loc[short_event] < close.loc[short_event])
    missing_tp = take_profit.isna()
    take_profit.loc[long_event & missing_tp] = close + float(params["atr_tp_mult"]) * atr
    take_profit.loc[short_event & missing_tp] = close - float(params["atr_tp_mult"]) * atr

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
    slippage_rate: float | None = None,
) -> dict[str, Any]:
    if len(featured) < int(params["max_holding_bars"]) + 5:
        return BacktestMetrics().to_dict(BacktestMetrics().calculate(SimpleNamespace(trades=[])))
    config = VectorizedBacktestConfig(
        initial_balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate if slippage_rate is None else slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        sizing_mode=args.sizing_mode,
        min_holding_bars=int(params["min_holding_bars"]),
        max_holding_bars=int(params["max_holding_bars"]),
        exit_on_flat_signal=bool(params["exit_on_flat_signal"]),
        exit_on_opposite_signal=True,
    )
    result = VectorizedBacktestEngine(featured, config, symbol).run()
    calc = BacktestMetrics()
    metrics = calc.calculate(result_object(result, config), featured)
    return calc.to_dict(metrics)


def trades_per_week(metrics: dict[str, Any]) -> float:
    return float(metrics.get("trades_per_day", 0)) * 7.0


def deployment_gate(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        int(metrics.get("total_trades", 0)) >= args.min_validation_trades
        and float(metrics.get("win_rate", 0)) >= 0.60
        and float(metrics.get("profit_factor", 0)) >= 2.5
        and float(metrics.get("expectancy_r", 0)) >= 0.15
        and float(metrics.get("sharpe_ratio", 0)) >= 2.0
        and float(metrics.get("max_drawdown_r", 0)) <= 8.0
        and float(metrics.get("max_drawdown_pct", 0)) <= 12.0
        and int(metrics.get("max_consecutive_losses", 0)) <= 7
        and trades_per_week(metrics) >= args.min_trades_per_week
    )


def window_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 4:
        return -100.0 + trades
    win_rate = float(metrics.get("win_rate", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 8.0)
    exp_r = float(metrics.get("expectancy_r", 0))
    sharpe = float(metrics.get("sharpe_ratio", 0))
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    freq = min(trades_per_week(metrics), args.min_trades_per_week * 2)
    if exp_r <= 0 or pf < 1.1:
        return win_rate * 35 + exp_r * 240 + (pf - 1.0) * 35 - dd_r * 3 - loss_streak * 4 + freq * 5
    return win_rate * 90 + (pf - 1.0) * 36 + exp_r * 180 + sharpe * 5 + freq * 8 - dd_r * 2.8 - loss_streak * 4


def aggregate_score(window_metrics: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if not window_metrics:
        return -1_000_000, {}
    total_trades = sum(int(item.get("total_trades", 0)) for item in window_metrics)
    if total_trades < args.min_train_trades:
        return -1_000_000 + total_trades, {}
    exp_values = [float(item.get("expectancy_r", 0)) for item in window_metrics]
    win_values = [float(item.get("win_rate", 0)) for item in window_metrics]
    pf_values = [float(item.get("profit_factor", 0)) for item in window_metrics]
    dd_values = [float(item.get("max_drawdown_r", 0)) for item in window_metrics]
    ret_values = [float(item.get("total_return_pct", 0)) for item in window_metrics]
    gate_windows = sum(1 for item in window_metrics if deployment_gate(item, args))
    positive_exp = sum(1 for value in exp_values if value > 0)
    aggregate = {
        "wf_windows": len(window_metrics),
        "wf_gate_windows": gate_windows,
        "wf_positive_exp": positive_exp,
        "wf_profitable": sum(1 for value in ret_values if value > 0),
        "wf_mean_win_rate": sum(win_values) / len(win_values),
        "wf_min_win_rate": min(win_values),
        "wf_mean_exp_r": sum(exp_values) / len(exp_values),
        "wf_min_exp_r": min(exp_values),
        "wf_mean_pf": sum(min(value, 10.0) for value in pf_values) / len(pf_values),
        "wf_min_pf": min(pf_values),
        "wf_max_dd_r": max(dd_values),
        "wf_total_trades": total_trades,
    }
    score = (
        sum(window_score(item, args) for item in window_metrics) / len(window_metrics)
        + gate_windows / len(window_metrics) * 180
        + positive_exp / len(window_metrics) * 100
        + aggregate["wf_mean_win_rate"] * 90
        + aggregate["wf_mean_exp_r"] * 120
        - max(0.0, args.min_trades_per_week - total_trades / len(window_metrics) / 13.0) * 40
        - aggregate["wf_max_dd_r"] * 2.0
    )
    return score, aggregate


def validation_rank(row: dict[str, Any], args: argparse.Namespace) -> float:
    stress_pass = bool(row.get("stress_pass"))
    validation_pass = bool(row.get("validation_pass"))
    return (
        (500 if stress_pass else 0)
        + (300 if validation_pass else 0)
        + float(row.get("stress_win_rate", row.get("validation_win_rate", 0))) * 120
        + min(float(row.get("stress_profit_factor", row.get("validation_profit_factor", 0))), 8.0) * 24
        + float(row.get("stress_expectancy_r", row.get("validation_expectancy_r", 0))) * 130
        + float(row.get("stress_sharpe_ratio", row.get("validation_sharpe_ratio", 0))) * 4
        - float(row.get("stress_max_drawdown_r", row.get("validation_max_drawdown_r", 0))) * 2.4
        - float(row.get("stress_max_consecutive_losses", row.get("validation_max_consecutive_losses", 0))) * 4
        + float(row.get("wf_score", 0)) * 0.4
    )


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
        "gate_windows": sum(1 for row in rows if row.get("gate_pass")),
        "positive_expectancy_windows": sum(1 for row in rows if row["expectancy_r"] > 0),
        "profitable_windows": sum(1 for row in rows if row["total_return_pct"] > 0),
        "mean_win_rate": sum(row["win_rate"] for row in rows) / len(rows),
        "mean_expectancy_r": sum(row["expectancy_r"] for row in rows) / len(rows),
        "trade_weighted_expectancy_r": weighted("expectancy_r"),
        "max_drawdown_r": max(row["max_drawdown_r"] for row in rows),
        "max_drawdown_pct": max(row["max_drawdown_pct"] for row in rows),
        "max_consecutive_losses": max(row["max_consecutive_losses"] for row in rows),
        "total_trades": total_trades,
        "weighted_avg_win_r": weighted("avg_win_r"),
        "weighted_avg_loss_r": weighted("avg_loss_r"),
        "weighted_payoff_ratio": weighted("payoff_ratio"),
        "weighted_trades_per_week": weighted("trades_per_day") * 7,
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

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"high_winrate_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    logger.info("Searching %d high-winrate candidates across %d train windows", args.trials, len(train_windows))
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
        score, aggregate = aggregate_score(window_metrics, args)
        rows.append({
            "trial": trial,
            "wf_score": score,
            **{field: params.get(field) for field in PARAM_FIELDS},
            **aggregate,
        })
        if (trial + 1) % 100 == 0:
            best = max(rows, key=lambda item: item["wf_score"])
            logger.info(
                "  %d/%d best %s score %.2f gates %s/%s win %.1f%% expR %.3f PF %.2f ddR %.2f trades %s",
                trial + 1,
                args.trials,
                best.get("family"),
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
    logger.info("Validating top %d train-ranked candidates", len(top_rows))
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        featured = apply_strategy(base, params)
        validation_slice = featured.loc[(featured.index >= validation_start) & (featured.index <= validation_end)]
        validation = run_metrics(validation_slice, args.symbol, args, params)
        stress = run_metrics(validation_slice, args.symbol, args, params, args.stress_slippage_rate)
        row.update(prefixed("validation", validation))
        row.update(prefixed("stress", stress))
        row["validation_trades_per_week"] = trades_per_week(validation)
        row["stress_trades_per_week"] = trades_per_week(stress)
        row["validation_pass"] = deployment_gate(validation, args)
        row["stress_pass"] = deployment_gate(stress, args)
        row["deploy_score"] = validation_rank(row, args)

    top_rows.sort(key=lambda item: item.get("deploy_score", -1_000_000), reverse=True)
    best_validation = top_rows[0] if top_rows else rows[0]
    best_train = rows[0]
    best = best_validation if best_validation.get("stress_pass") else best_train

    params = {field: best[field] for field in PARAM_FIELDS}
    featured = apply_strategy(base, params)
    wf_rows = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.validation_end, args.train_months, args.test_months),
        start=1,
    ):
        metrics = run_metrics(
            featured.loc[(featured.index >= test_start) & (featured.index <= test_end)],
            args.symbol,
            args,
            params,
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
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strategy": "high_winrate_edge_search",
        "selection_note": "best is train-rank-1 unless a top train-ranked candidate passes severe stress gate",
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
            "stress_slippage_rate": args.stress_slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
            "sizing_mode": args.sizing_mode,
        },
        "deployment_gate": {
            "min_win_rate": 0.60,
            "min_profit_factor": 2.5,
            "min_expectancy_r": 0.15,
            "min_sharpe_ratio": 2.0,
            "max_drawdown_r": 8.0,
            "max_drawdown_pct": 12.0,
            "max_consecutive_losses": 7,
            "min_trades_per_week": args.min_trades_per_week,
        },
        "seed": args.seed,
        "trials": args.trials,
        "top": args.top,
        "best": best,
        "best_train_ranked": best_train,
        "best_validation_ranked": best_validation,
        "aggregate": summarize_windows(wf_rows),
    }
    write_csv(run_dir / "training_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top_rows)
    write_csv(run_dir / "walk_forward.csv", wf_rows)
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Saved to %s | best %s train rank %s stress pass %s | stress win %.1f%% PF %.2f expR %.3f ddR %.2f trades/week %.2f | WF gates %s/%s",
        run_dir,
        best.get("family"),
        best.get("wf_rank"),
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
