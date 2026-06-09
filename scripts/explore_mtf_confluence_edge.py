#!/usr/bin/env python3
"""Explore multi-timeframe crypto strategies with shifted higher-timeframe features."""

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
from scripts.explore_high_winrate_edge import deployment_gate, ema, trades_per_week, walk_forward_windows
from scripts.explore_trailing_highwin_edge import metrics_from_trades, simulate_trades, summarize_windows
from scripts.run_backtest import vectorized_preprocess


logger = logging.getLogger("mtf_confluence_explorer")


FAMILIES = [
    "mtf_momentum_breakout",
    "mtf_trend_pullback",
    "mtf_bb_expansion",
    "mtf_rsi_reclaim",
    "mtf_state_follow",
]

PARAM_FIELDS = [
    "family",
    "allow_long",
    "allow_short",
    "atr_stop_mult",
    "atr_tp_mult",
    "break_even_atr",
    "channel_lookback",
    "daily_ema",
    "ema_fast",
    "ema_slow",
    "exit_on_flat_signal",
    "htf_ema_fast",
    "htf_ema_slow",
    "htf_macd_min",
    "max_holding_bars",
    "min_holding_bars",
    "move_stop_after_partial",
    "partial_exit_atr",
    "partial_exit_fraction",
    "pullback_atr",
    "rsi_high",
    "rsi_low",
    "slope_bars",
    "slope_min",
    "trail_atr_mult",
    "use_break_even",
    "use_partial_exit",
    "use_trailing_stop",
    "volatility_max",
    "volatility_min",
    "volume_lookback",
    "volume_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore shifted multi-timeframe confluence strategies")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--htf", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1800)
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
    parser.add_argument("--min-trades-per-week", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_raw(symbol: str, start: str, end: str) -> pd.DataFrame:
    loader = DataLoader(use_cache=True)
    raw = loader.load_symbol(symbol, "1m")
    if raw.index.tz is None:
        raw = raw.tz_localize("UTC")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return raw[(raw.index >= start_ts) & (raw.index <= end_ts)]


def add_htf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_close"] = close
    out[f"{prefix}_ema_fast_20"] = ema(close, 20)
    out[f"{prefix}_ema_slow_80"] = ema(close, 80)
    out[f"{prefix}_ema_slow_200"] = ema(close, 200)
    out[f"{prefix}_slope_12"] = out[f"{prefix}_ema_slow_80"].pct_change(12)
    macd_fast = ema(close, 12)
    macd_slow = ema(close, 26)
    macd = macd_fast - macd_slow
    out[f"{prefix}_macd_hist"] = macd - ema(macd, 9)
    mid = close.rolling(20, min_periods=10).mean()
    std = close.rolling(20, min_periods=10).std()
    width = (4.0 * std / mid.replace(0, np.nan)).fillna(0)
    out[f"{prefix}_bb_pos"] = ((close - (mid - 2 * std)) / (4 * std).replace(0, np.nan)).clip(-2, 3)
    out[f"{prefix}_bb_width"] = width
    out[f"{prefix}_bb_width_rank"] = width.rolling(100, min_periods=20).rank(pct=True)
    return out.shift(1)


def prepare_data(symbol: str, timeframe: str, htf: str, start: str, end: str) -> pd.DataFrame:
    raw = load_raw(symbol, start, end)
    base = vectorized_preprocess(raw, timeframe)
    high_tf = vectorized_preprocess(raw, htf)
    daily = vectorized_preprocess(raw, "1d")
    htf_features = add_htf_features(high_tf, "htf")
    daily_features = add_htf_features(daily, "daily")
    merged = pd.merge_asof(
        base.sort_index(),
        htf_features.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_index(),
        daily_features.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


def sample_params(rng: np.random.Generator, trial: int) -> dict[str, Any]:
    family = FAMILIES[trial % len(FAMILIES)] if trial < len(FAMILIES) else str(rng.choice(FAMILIES))
    side_mode = str(rng.choice(["long_only", "both", "short_only"], p=[0.68, 0.24, 0.08]))
    min_hold = int(rng.choice([2, 3, 4, 6, 8, 12]))
    max_hold = int(rng.choice([24, 36, 48, 72, 96, 120, 168, 240]))
    if max_hold <= min_hold:
        max_hold = min_hold + 12
    ema_fast = int(rng.choice([8, 10, 14, 20, 30, 50]))
    ema_slow = int(rng.choice([80, 100, 150, 200, 300]))
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 80
    return {
        "family": family,
        "allow_long": side_mode in {"long_only", "both"},
        "allow_short": side_mode in {"short_only", "both"},
        "atr_stop_mult": float(rng.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])),
        "atr_tp_mult": float(rng.choice([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])),
        "break_even_atr": float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])),
        "channel_lookback": int(rng.choice([12, 18, 24, 36, 48, 72])),
        "daily_ema": int(rng.choice([80, 200])),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "exit_on_flat_signal": bool(rng.choice([True, False], p=[0.70, 0.30])),
        "htf_ema_fast": int(rng.choice([20])),
        "htf_ema_slow": int(rng.choice([80, 200])),
        "htf_macd_min": float(rng.choice([0.0, 0.0005, 0.001, 0.0025])),
        "max_holding_bars": max_hold,
        "min_holding_bars": min_hold,
        "move_stop_after_partial": bool(rng.choice([True, False], p=[0.85, 0.15])),
        "partial_exit_atr": float(rng.choice([0.8, 1.0, 1.2, 1.5, 2.0, 3.0])),
        "partial_exit_fraction": float(rng.choice([0.25, 0.33, 0.5, 0.67])),
        "pullback_atr": float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5])),
        "rsi_high": float(rng.uniform(58.0, 78.0)),
        "rsi_low": float(rng.uniform(22.0, 45.0)),
        "slope_bars": int(rng.choice([6, 12, 18, 24, 36])),
        "slope_min": float(rng.choice([0.0, 0.0005, 0.001, 0.0025, 0.005])),
        "trail_atr_mult": float(rng.choice([1.5, 2.0, 2.5, 3.0, 4.0])),
        "use_break_even": bool(rng.choice([True, False], p=[0.80, 0.20])),
        "use_partial_exit": bool(rng.choice([True, False], p=[0.65, 0.35])),
        "use_trailing_stop": bool(rng.choice([True, False], p=[0.65, 0.35])),
        "volatility_max": float(rng.choice([0.04, 0.05, 0.08, 0.12, 0.20, 1.0])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002, 0.003])),
        "volume_lookback": int(rng.choice([20, 30, 50, 75, 100])),
        "volume_min": float(rng.choice([0.0, 0.5, 0.75, 1.0, 1.25, 1.5])),
    }


def apply_strategy(base: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    df = base.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"].replace(0, np.nan)
    fast = ema(close, int(params["ema_fast"]))
    slow = ema(close, int(params["ema_slow"]))
    slope = slow.pct_change(int(params["slope_bars"]))
    vol_sma = df["volume"].rolling(int(params["volume_lookback"]), min_periods=1).mean()
    vol_ratio = (df["volume"] / vol_sma.replace(0, np.nan)).fillna(0)
    channel_high = high.rolling(int(params["channel_lookback"]), min_periods=3).max().shift(1)
    channel_low = low.rolling(int(params["channel_lookback"]), min_periods=3).min().shift(1)
    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    htf_slow_col = "htf_ema_slow_80" if int(params["htf_ema_slow"]) == 80 else "htf_ema_slow_200"
    daily_col = "daily_ema_slow_80" if int(params["daily_ema"]) == 80 else "daily_ema_slow_200"
    htf_up = (df["htf_ema_fast_20"] > df[htf_slow_col]) & (df["htf_macd_hist"] > float(params["htf_macd_min"]))
    htf_down = (df["htf_ema_fast_20"] < df[htf_slow_col]) & (df["htf_macd_hist"] < -float(params["htf_macd_min"]))
    daily_up = df["daily_close"] > df[daily_col]
    daily_down = df["daily_close"] < df[daily_col]
    ltf_up = (fast > slow) & (slope > float(params["slope_min"]))
    ltf_down = (fast < slow) & (slope < -float(params["slope_min"]))
    vol_ok = (df["daily_vol"] >= float(params["volatility_min"])) & (df["daily_vol"] <= float(params["volatility_max"]))
    volume_ok = vol_ratio >= float(params["volume_min"])

    family = str(params["family"])
    if family == "mtf_momentum_breakout":
        long_event = daily_up & htf_up & ltf_up & (close > channel_high)
        short_event = daily_down & htf_down & ltf_down & (close < channel_low)
    elif family == "mtf_trend_pullback":
        long_event = daily_up & htf_up & ltf_up & (low <= fast + float(params["pullback_atr"]) * atr) & (close > fast)
        short_event = daily_down & htf_down & ltf_down & (high >= fast - float(params["pullback_atr"]) * atr) & (close < fast)
    elif family == "mtf_bb_expansion":
        squeeze = df["htf_bb_width_rank"] <= 0.35
        long_event = daily_up & htf_up & squeeze & (close > bb_upper) & (df["rsi_14"] >= 50)
        short_event = daily_down & htf_down & squeeze & (close < bb_lower) & (df["rsi_14"] <= 50)
    elif family == "mtf_rsi_reclaim":
        long_event = daily_up & htf_up & (df["rsi_14"].shift(1) < float(params["rsi_low"])) & (df["rsi_14"] >= float(params["rsi_low"])) & (close > fast)
        short_event = daily_down & htf_down & (df["rsi_14"].shift(1) > float(params["rsi_high"])) & (df["rsi_14"] <= float(params["rsi_high"])) & (close < fast)
    elif family == "mtf_state_follow":
        long_event = daily_up & htf_up & ltf_up
        short_event = daily_down & htf_down & ltf_down
    else:
        raise ValueError(f"Unknown family: {family}")

    long_event = long_event & vol_ok & volume_ok & bool(params["allow_long"])
    short_event = short_event & vol_ok & volume_ok & bool(params["allow_short"])
    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[long_event] = 1
    signal.loc[short_event] = -1
    stop_loss = pd.Series(np.nan, index=df.index, dtype=float)
    take_profit = pd.Series(np.nan, index=df.index, dtype=float)
    stop_loss.loc[long_event] = close - float(params["atr_stop_mult"]) * atr
    stop_loss.loc[short_event] = close + float(params["atr_stop_mult"]) * atr
    take_profit.loc[long_event] = close + float(params["atr_tp_mult"]) * atr
    take_profit.loc[short_event] = close - float(params["atr_tp_mult"]) * atr

    df["signal"] = signal
    df["stop_loss"] = stop_loss
    df["take_profit"] = take_profit
    df["volume_ratio_mtf"] = vol_ratio
    return df.dropna(subset=["atr_14", "daily_vol", "htf_close", "daily_close"])


def metrics_for(featured: pd.DataFrame, symbol: str, args: argparse.Namespace, params: dict[str, Any], slippage: float) -> dict[str, Any]:
    runtime = argparse.Namespace(
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=slippage,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
    )
    trades = simulate_trades(featured, symbol, runtime, params)
    return metrics_from_trades(trades, featured.index, args.balance)


def score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 3:
        return -1000 + trades
    win = float(metrics.get("win_rate", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 8.0)
    exp = float(metrics.get("expectancy_r", 0))
    sharpe = min(float(metrics.get("sharpe_ratio", 0)), 6.0)
    dd = float(metrics.get("max_drawdown_r", 0))
    streak = float(metrics.get("max_consecutive_losses", 0))
    freq = min(trades_per_week(metrics), args.min_trades_per_week * 4)
    return win * 80 + exp * 260 + (pf - 1) * 45 + sharpe * 9 + freq * 5 - dd * 4 - streak * 5


def aggregate_score(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    total = sum(int(row.get("total_trades", 0)) for row in rows)
    if total < args.min_train_trades:
        return -1_000_000 + total, {}
    aggregate = {
        "wf_windows": len(rows),
        "wf_gate_windows": sum(1 for row in rows if deployment_gate(row, args)),
        "wf_positive_exp": sum(1 for row in rows if float(row.get("expectancy_r", 0)) > 0),
        "wf_mean_win_rate": sum(float(row.get("win_rate", 0)) for row in rows) / len(rows),
        "wf_mean_exp_r": sum(float(row.get("expectancy_r", 0)) for row in rows) / len(rows),
        "wf_mean_pf": sum(min(float(row.get("profit_factor", 0)), 10.0) for row in rows) / len(rows),
        "wf_max_dd_r": max(float(row.get("max_drawdown_r", 0)) for row in rows),
        "wf_total_trades": total,
    }
    total_score = (
        sum(score(row, args) for row in rows) / len(rows)
        + aggregate["wf_gate_windows"] / len(rows) * 220
        + aggregate["wf_positive_exp"] / len(rows) * 100
        + aggregate["wf_mean_exp_r"] * 140
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
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    out_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = out_root / f"mtf_confluence_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base = prepare_data(args.symbol, args.timeframe, args.htf, args.start, args.validation_end)
    rng = np.random.default_rng(args.seed)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")

    rows: list[dict[str, Any]] = []
    logger.info("Searching %d MTF candidates", args.trials)
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        featured = apply_strategy(base, params)
        metrics = [
            metrics_for(featured.loc[(featured.index >= test_start) & (featured.index <= test_end)], args.symbol, args, params, args.train_slippage_rate)
            for _, _, test_start, test_end in train_windows
        ]
        wf_score, aggregate = aggregate_score(metrics, args)
        rows.append({"trial": trial, "wf_score": wf_score, **{field: params.get(field) for field in PARAM_FIELDS}, **aggregate})
        if (trial + 1) % 100 == 0:
            best = max(rows, key=lambda row: row["wf_score"])
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

    rows.sort(key=lambda row: row["wf_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["wf_rank"] = rank
    top_rows = rows[: args.top]
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        featured = apply_strategy(base, params)
        validation_slice = featured.loc[(featured.index >= val_start) & (featured.index <= val_end)]
        validation = metrics_for(validation_slice, args.symbol, args, params, args.slippage_rate)
        stress = metrics_for(validation_slice, args.symbol, args, params, args.stress_slippage_rate)
        for prefix, metrics in [("validation", validation), ("stress", stress)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_trades_per_week"] = trades_per_week(metrics)
        row["validation_pass"] = deployment_gate(validation, args)
        row["stress_pass"] = deployment_gate(stress, args)
        row["deploy_score"] = (
            (650 if row["stress_pass"] else 0)
            + (250 if row["validation_pass"] else 0)
            + row["stress_win_rate"] * 100
            + min(row["stress_profit_factor"], 10.0) * 45
            + row["stress_expectancy_r"] * 280
            + min(row["stress_sharpe_ratio"], 6.0) * 10
            - row["stress_max_drawdown_r"] * 5
            - row["stress_max_consecutive_losses"] * 6
            + row["wf_score"] * 0.35
        )
    top_rows.sort(key=lambda row: row["deploy_score"], reverse=True)
    best = top_rows[0] if top_rows else rows[0]
    best_params = {field: best[field] for field in PARAM_FIELDS}
    best_featured = apply_strategy(base, best_params)
    wf_rows = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.validation_end, args.train_months, args.test_months),
        start=1,
    ):
        metrics = metrics_for(
            best_featured.loc[(best_featured.index >= test_start) & (best_featured.index <= test_end)],
            args.symbol,
            args,
            best_params,
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
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strategy": "mtf_confluence_edge_search",
        "seed": args.seed,
        "trials": args.trials,
        "best": best,
        "best_validation_ranked": top_rows[0] if top_rows else {},
        "aggregate": summarize_windows(wf_rows),
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "train_slippage_rate": args.train_slippage_rate,
            "stress_slippage_rate": args.stress_slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
        },
    }
    write_csv(run_dir / "training_candidates.csv", rows)
    write_csv(run_dir / "validated_top_candidates.csv", top_rows)
    write_csv(run_dir / "walk_forward.csv", wf_rows)
    (run_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Saved to %s | best %s rank %s stress pass %s win %.1f%% PF %.2f expR %.3f ddR %.2f trades/week %.2f WF gates %s/%s",
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
