#!/usr/bin/env python3
"""Explore multiple explainable strategy families under corrected execution.

The search intentionally stays simple and auditable: each candidate is a small
set of rule-based OHLCV signals evaluated with realistic fees/slippage,
risk-based sizing, R-multiple metrics, and out-of-sample stress validation.
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
from scripts.run_backtest import (
    VectorizedBacktestConfig,
    VectorizedBacktestEngine,
    vectorized_preprocess,
)


logger = logging.getLogger("strategy_family_explorer")

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
    "family",
    "allow_long",
    "allow_short",
    "atr_stop_mult",
    "atr_tp_mult",
    "bb_lookback",
    "bb_std",
    "bb_width_lookback",
    "bb_width_quantile",
    "donchian_lookback",
    "ema_fast",
    "ema_slow",
    "max_holding_bars",
    "max_trend_strength",
    "min_holding_bars",
    "pullback_depth",
    "pullback_rsi",
    "range_lookback",
    "rsi_high",
    "rsi_low",
    "trend_slope_bars",
    "trend_slope_min",
    "volatility_max",
    "volatility_min",
    "volume_lookback",
    "volume_min",
]

FAMILIES = [
    "trend_pullback",
    "channel_breakout",
    "squeeze_breakout",
    "range_reversion",
    "ema_momentum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore multiple corrected strategy families")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--top-train", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0010)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0020)
    parser.add_argument("--min-train-trades", type=int, default=50)
    parser.add_argument("--min-validation-trades", type=int, default=18)
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
    family = FAMILIES[trial % len(FAMILIES)] if trial < len(FAMILIES) else str(rng.choice(FAMILIES))
    min_hold = int(rng.integers(1, 7))
    max_hold = int(rng.choice([12, 18, 24, 36, 48, 72, 96, 120, 168, 240, 360]))
    if max_hold <= min_hold:
        max_hold = min_hold + 6
    side_mode = str(rng.choice(["both", "long_only", "short_only"], p=[0.30, 0.60, 0.10]))

    return {
        "family": family,
        "allow_long": side_mode in {"both", "long_only"},
        "allow_short": side_mode in {"both", "short_only"},
        "atr_stop_mult": float(rng.uniform(0.7, 4.5)),
        "atr_tp_mult": float(rng.uniform(1.2, 12.0)),
        "bb_lookback": int(rng.choice([14, 20, 30, 40, 55, 80])),
        "bb_std": float(rng.choice([1.5, 1.8, 2.0, 2.2, 2.5])),
        "bb_width_lookback": int(rng.choice([50, 100, 150, 200, 300])),
        "bb_width_quantile": float(rng.choice([0.10, 0.15, 0.20, 0.25, 0.30])),
        "donchian_lookback": int(rng.choice([8, 10, 12, 16, 20, 24, 30, 40, 55, 80, 120])),
        "ema_fast": int(rng.choice([10, 14, 20, 30, 50, 80])),
        "ema_slow": int(rng.choice([50, 80, 100, 150, 200, 300, 400])),
        "max_holding_bars": max_hold,
        "max_trend_strength": float(rng.uniform(0.004, 0.08)),
        "min_holding_bars": min_hold,
        "pullback_depth": float(rng.uniform(0.0, 0.035)),
        "pullback_rsi": float(rng.uniform(34.0, 58.0)),
        "range_lookback": int(rng.choice([20, 30, 40, 55, 80, 120])),
        "rsi_high": float(rng.uniform(58.0, 82.0)),
        "rsi_low": float(rng.uniform(18.0, 42.0)),
        "trend_slope_bars": int(rng.choice([3, 6, 12, 18, 24, 36, 48])),
        "trend_slope_min": float(rng.uniform(0.0, 0.025)),
        "volatility_max": float(rng.choice([0.03, 0.04, 0.05, 0.08, 0.12, 0.20])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002, 0.003])),
        "volume_lookback": int(rng.choice([10, 20, 30, 50, 75, 100, 150, 200])),
        "volume_min": float(rng.uniform(0.0, 2.2)),
    }


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(int(period), 1), adjust=False, min_periods=1).mean()


def rolling_bb(close: pd.Series, lookback: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    std = close.rolling(lookback, min_periods=max(5, lookback // 3)).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def common_filters(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    volume = df["volume"]
    vol_sma = volume.rolling(int(params["volume_lookback"]), min_periods=1).mean()
    volume_ok = (volume / vol_sma.replace(0, np.nan)).fillna(0) >= float(params["volume_min"])
    volatility = df["daily_vol"].fillna(0)
    volatility_ok = (volatility >= float(params["volatility_min"])) & (volatility <= float(params["volatility_max"]))
    return volume_ok, volatility_ok


def apply_strategy(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    featured = df.copy()
    close = featured["close"]
    high = featured["high"]
    low = featured["low"]
    atr = featured["atr_14"].replace(0, np.nan)
    signal = pd.Series(0, index=featured.index, dtype=int)
    stop_loss = pd.Series(np.nan, index=featured.index, dtype=float)
    take_profit = pd.Series(np.nan, index=featured.index, dtype=float)

    volume_ok, volatility_ok = common_filters(featured, params)
    fast = ema(close, int(params["ema_fast"]))
    slow = ema(close, int(params["ema_slow"]))
    slope = slow.pct_change(int(params["trend_slope_bars"]))
    trend_strength = ((fast - slow).abs() / slow.replace(0, np.nan)).fillna(0)
    uptrend = (fast > slow) & (slope > float(params["trend_slope_min"]))
    downtrend = (fast < slow) & (slope < -float(params["trend_slope_min"]))

    family = str(params["family"])
    if family == "trend_pullback":
        long_event = (
            uptrend
            & (close.shift(1) <= fast.shift(1) * (1 - float(params["pullback_depth"])))
            & (close > fast)
            & (featured["rsi_14"] <= float(params["pullback_rsi"]) + 12)
        )
        short_event = (
            downtrend
            & (close.shift(1) >= fast.shift(1) * (1 + float(params["pullback_depth"])))
            & (close < fast)
            & (featured["rsi_14"] >= 100 - float(params["pullback_rsi"]) - 12)
        )
    elif family == "channel_breakout":
        channel_high = high.rolling(int(params["donchian_lookback"]), min_periods=1).max().shift(1)
        channel_low = low.rolling(int(params["donchian_lookback"]), min_periods=1).min().shift(1)
        long_event = (close > channel_high) & uptrend
        short_event = (close < channel_low) & downtrend
    elif family == "squeeze_breakout":
        _, _, _, width = rolling_bb(close, int(params["bb_lookback"]), float(params["bb_std"]))
        threshold = width.shift(1).rolling(int(params["bb_width_lookback"]), min_periods=20).quantile(
            float(params["bb_width_quantile"])
        )
        squeeze = width.shift(1) < threshold
        channel_high = high.rolling(int(params["donchian_lookback"]), min_periods=1).max().shift(1)
        channel_low = low.rolling(int(params["donchian_lookback"]), min_periods=1).min().shift(1)
        long_event = squeeze & (close > channel_high) & (fast >= slow)
        short_event = squeeze & (close < channel_low) & (fast <= slow)
    elif family == "range_reversion":
        mid, upper, lower, _ = rolling_bb(close, int(params["bb_lookback"]), float(params["bb_std"]))
        range_ok = trend_strength <= float(params["max_trend_strength"])
        long_event = range_ok & (close < lower) & (featured["rsi_14"] <= float(params["rsi_low"]))
        short_event = range_ok & (close > upper) & (featured["rsi_14"] >= float(params["rsi_high"]))
    elif family == "ema_momentum":
        long_event = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (close > slow)
        short_event = (fast < slow) & (fast.shift(1) >= slow.shift(1)) & (close < slow)
    else:
        raise ValueError(f"Unknown family: {family}")

    long_event = long_event & volume_ok & volatility_ok & bool(params["allow_long"])
    short_event = short_event & volume_ok & volatility_ok & bool(params["allow_short"])
    signal.loc[long_event] = 1
    signal.loc[short_event] = -1

    if family == "range_reversion":
        mid, _, _, _ = rolling_bb(close, int(params["bb_lookback"]), float(params["bb_std"]))
        stop_loss.loc[long_event] = close - float(params["atr_stop_mult"]) * atr
        take_profit.loc[long_event] = mid
        stop_loss.loc[short_event] = close + float(params["atr_stop_mult"]) * atr
        take_profit.loc[short_event] = mid
    else:
        stop_loss.loc[long_event] = close - float(params["atr_stop_mult"]) * atr
        take_profit.loc[long_event] = close + float(params["atr_tp_mult"]) * atr
        stop_loss.loc[short_event] = close + float(params["atr_stop_mult"]) * atr
        take_profit.loc[short_event] = close - float(params["atr_tp_mult"]) * atr

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
    balance: float,
    risk_per_trade: float,
    fee_rate: float,
    slippage_rate: float,
    max_position_pct: float,
    sizing_mode: str,
    min_holding_bars: int,
    max_holding_bars: int,
) -> dict[str, float]:
    if len(featured) < max_holding_bars + 5:
        return BacktestMetrics().to_dict(BacktestMetrics().calculate(SimpleNamespace(trades=[])))

    config = VectorizedBacktestConfig(
        initial_balance=balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=risk_per_trade,
        max_position_pct=max_position_pct,
        sizing_mode=sizing_mode,
        min_holding_bars=min_holding_bars,
        max_holding_bars=max_holding_bars,
        exit_on_flat_signal=False,
        exit_on_opposite_signal=True,
    )
    result = VectorizedBacktestEngine(featured, config, symbol).run()
    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate(result_object(result, config), featured)
    return metrics_calc.to_dict(metrics)


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": metrics.get(key, 0) for key in BASE_FIELDS}


def score_metrics(metrics: dict[str, Any], min_trades: int) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < min_trades:
        return -1_000_000 + trades
    pf = min(float(metrics.get("profit_factor", 0)), 5.0)
    exp_r = float(metrics.get("expectancy_r", 0))
    sharpe = float(metrics.get("sharpe_ratio", 0))
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    return exp_r * 180 + (pf - 1.0) * 35 + sharpe * 4 - dd_r * 1.4 - loss_streak * 1.2


def deploy_score(validation: dict[str, Any], stress: dict[str, Any], min_trades: int) -> float:
    if validation.get("total_trades", 0) < min_trades or stress.get("total_trades", 0) < min_trades:
        return -1_000_000
    val_score = score_metrics(validation, min_trades)
    stress_score = score_metrics(stress, min_trades)
    stress_exp = float(stress.get("expectancy_r", 0))
    stress_dd = float(stress.get("max_drawdown_r", 0))
    fragility_penalty = max(0.0, -stress_exp) * 250 + max(0.0, stress_dd - 12) * 4
    return val_score * 0.55 + stress_score * 0.45 - fragility_penalty


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
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
    run_dir = output_root / f"family_edge_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_data(args)
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    logger.info("Training %d family candidates on %s %s", args.trials, args.symbol, args.timeframe)
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        featured = apply_strategy(base_df, params)
        train_metrics = run_metrics(
            featured.loc[featured.index <= train_end],
            args.symbol,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
            int(params["min_holding_bars"]),
            int(params["max_holding_bars"]),
        )
        row = {
            "trial": trial,
            "train_score": score_metrics(train_metrics, args.min_train_trades),
            **{key: params.get(key) for key in PARAM_FIELDS},
            **prefixed("train", train_metrics),
        }
        rows.append(row)
        if (trial + 1) % 100 == 0:
            best = max(rows, key=lambda item: item["train_score"])
            logger.info(
                "  %d/%d best %s score %.2f PF %.2f expR %.3f ddR %.2f trades %s",
                trial + 1,
                args.trials,
                best["family"],
                best["train_score"],
                best["train_profit_factor"],
                best["train_expectancy_r"],
                best["train_max_drawdown_r"],
                best["train_total_trades"],
            )

    rows.sort(key=lambda item: item["train_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["train_rank"] = rank

    top_rows = rows[: args.top_train]
    logger.info("Validating top %d candidates with base and stress execution", len(top_rows))
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        featured = apply_strategy(base_df, params)
        validation_slice = featured.loc[(featured.index >= val_start) & (featured.index <= val_end)]
        validation_metrics = run_metrics(
            validation_slice,
            args.symbol,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
            int(params["min_holding_bars"]),
            int(params["max_holding_bars"]),
        )
        stress_metrics = run_metrics(
            validation_slice,
            args.symbol,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.stress_slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
            int(params["min_holding_bars"]),
            int(params["max_holding_bars"]),
        )
        row.update(prefixed("validation", validation_metrics))
        row.update(prefixed("stress", stress_metrics))
        row["deploy_score"] = deploy_score(validation_metrics, stress_metrics, args.min_validation_trades)

    top_rows.sort(key=lambda item: item.get("deploy_score", -1_000_000), reverse=True)
    best = top_rows[0] if top_rows else rows[0]

    write_csv(run_dir / "training_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top_rows)
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
        },
        "seed": args.seed,
        "trials": args.trials,
        "top_train": args.top_train,
        "best": best,
    }
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info("Family search saved in %s", run_dir)
    logger.info(
        "Best %s score %.2f | stress PF %.2f expR %.3f ddR %.2f trades %s",
        best.get("family"),
        best.get("deploy_score", 0),
        best.get("stress_profit_factor", 0),
        best.get("stress_expectancy_r", 0),
        best.get("stress_max_drawdown_r", 0),
        best.get("stress_total_trades", 0),
    )


if __name__ == "__main__":
    main()
