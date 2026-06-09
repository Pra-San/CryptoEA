#!/usr/bin/env python3
"""Explore VWAP, volume-profile, and momentum strategy combinations."""

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

from scripts.explore_high_winrate_edge import (
    deployment_gate,
    ema,
    load_data,
    trades_per_week,
    walk_forward_windows,
    write_csv,
)
from scripts.explore_trailing_highwin_edge import PARAM_FIELDS as TRAILING_FIELDS
from scripts.explore_trailing_highwin_edge import run_metrics


logger = logging.getLogger("vwap_profile_explorer")


FAMILIES = [
    "vwap_reclaim",
    "vwap_pullback",
    "daily_vwap_pullback",
    "vwap_band_reversion",
    "profile_value_reject",
    "profile_breakout",
    "profile_poc_reversion",
    "vwap_value_rotation",
    "vwap_profile_confluence",
    "momentum_volume_breakout",
]

PARAM_FIELDS = [
    "family",
    "allow_long",
    "allow_short",
    "atr_stop_mult",
    "atr_tp_mult",
    "break_even_atr",
    "donchian_lookback",
    "ema_fast",
    "ema_slow",
    "exit_on_flat_signal",
    "max_holding_bars",
    "min_holding_bars",
    "mom_lookback",
    "mom_min",
    "move_stop_after_partial",
    "partial_exit_atr",
    "partial_exit_fraction",
    "poc_distance_max",
    "profile_lookback",
    "profile_value_area",
    "rsi_high",
    "rsi_low",
    "slope_bars",
    "slope_min",
    "target_mode",
    "trail_atr_mult",
    "trend_strength_max",
    "use_break_even",
    "use_partial_exit",
    "use_poc_target",
    "use_trailing_stop",
    "volatility_max",
    "volatility_min",
    "volume_lookback",
    "volume_min",
    "vwap_band_atr",
    "vwap_distance_max",
    "vwap_lookback",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore VWAP/profile/momentum strategy variants")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--top", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0010)
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-validation-trades", type=int, default=80)
    parser.add_argument("--min-trades-per-week", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def rolling_vwap(df: pd.DataFrame, lookback: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    dollar_volume = typical * df["volume"]
    vol_sum = df["volume"].rolling(lookback, min_periods=max(5, lookback // 4)).sum()
    return dollar_volume.rolling(lookback, min_periods=max(5, lookback // 4)).sum() / vol_sum.replace(0, np.nan)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, target, side="left")])


def profile_features(df: pd.DataFrame, lookback: int, value_area: float) -> pd.DataFrame:
    typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    poc = np.full(len(df), np.nan)
    val = np.full(len(df), np.nan)
    vah = np.full(len(df), np.nan)
    lower_q = max(0.0, (1.0 - value_area) / 2.0)
    upper_q = min(1.0, 1.0 - lower_q)
    for i in range(lookback - 1, len(df)):
        start = i - lookback + 1
        prices = typical[start : i + 1]
        weights = volume[start : i + 1]
        valid = np.isfinite(prices) & np.isfinite(weights) & (weights > 0)
        if valid.sum() < max(5, lookback // 5):
            continue
        prices = prices[valid]
        weights = weights[valid]
        poc[i] = float(prices[np.argmax(weights)])
        val[i] = weighted_quantile(prices, weights, lower_q)
        vah[i] = weighted_quantile(prices, weights, upper_q)
    return pd.DataFrame({"profile_poc": poc, "profile_val": val, "profile_vah": vah}, index=df.index)


def sample_params(rng: np.random.Generator, trial: int) -> dict[str, Any]:
    family = FAMILIES[trial % len(FAMILIES)] if trial < len(FAMILIES) else str(rng.choice(FAMILIES))
    side_mode = str(rng.choice(["long_only", "both", "short_only"], p=[0.72, 0.24, 0.04]))
    ema_fast = int(rng.choice([8, 10, 14, 20, 30, 50]))
    ema_slow = int(rng.choice([80, 100, 150, 200, 300, 400]))
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 80
    min_hold = int(rng.choice([1, 2, 3, 4, 6, 8]))
    max_hold = int(rng.choice([8, 12, 18, 24, 36, 48, 72, 96]))
    if max_hold <= min_hold:
        max_hold = min_hold + 12
    return {
        "family": family,
        "allow_long": side_mode in {"long_only", "both"},
        "allow_short": side_mode in {"short_only", "both"},
        "atr_stop_mult": float(rng.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])),
        "atr_tp_mult": float(rng.choice([0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])),
        "break_even_atr": float(rng.choice([0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5])),
        "donchian_lookback": int(rng.choice([10, 16, 20, 30, 40, 55])),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "exit_on_flat_signal": bool(rng.choice([True, False], p=[0.80, 0.20])),
        "max_holding_bars": max_hold,
        "min_holding_bars": min_hold,
        "mom_lookback": int(rng.choice([3, 6, 12, 18, 24, 36])),
        "mom_min": float(rng.choice([0.0005, 0.001, 0.0025, 0.005, 0.01, 0.015])),
        "move_stop_after_partial": bool(rng.choice([True, False], p=[0.90, 0.10])),
        "partial_exit_atr": float(rng.choice([0.25, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])),
        "partial_exit_fraction": float(rng.choice([0.25, 0.33, 0.5, 0.67])),
        "poc_distance_max": float(rng.choice([0.005, 0.01, 0.015, 0.025, 0.04, 0.08])),
        "profile_lookback": int(rng.choice([48, 72, 96, 120, 168, 240])),
        "profile_value_area": float(rng.choice([0.60, 0.68, 0.70, 0.75, 0.80])),
        "rsi_high": float(rng.uniform(58.0, 78.0)),
        "rsi_low": float(rng.uniform(22.0, 45.0)),
        "slope_bars": int(rng.choice([6, 12, 18, 24, 36, 48])),
        "slope_min": float(rng.choice([0.0, 0.0005, 0.001, 0.0025, 0.005])),
        "target_mode": str(rng.choice(["atr", "poc", "value_area", "session_vwap"], p=[0.55, 0.18, 0.17, 0.10])),
        "trail_atr_mult": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
        "trend_strength_max": float(rng.choice([0.006, 0.01, 0.015, 0.025, 0.04, 0.08])),
        "use_break_even": bool(rng.choice([True, False], p=[0.85, 0.15])),
        "use_partial_exit": bool(rng.choice([True, False], p=[0.75, 0.25])),
        "use_poc_target": bool(rng.choice([False, True], p=[0.80, 0.20])),
        "use_trailing_stop": bool(rng.choice([True, False], p=[0.75, 0.25])),
        "volatility_max": float(rng.choice([0.03, 0.04, 0.05, 0.08, 0.12, 1.0])),
        "volatility_min": float(rng.choice([0.0, 0.0005, 0.001, 0.002])),
        "volume_lookback": int(rng.choice([20, 30, 50, 75, 100, 150])),
        "volume_min": float(rng.choice([0.0, 0.5, 0.75, 1.0, 1.25, 1.5])),
        "vwap_band_atr": float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5])),
        "vwap_distance_max": float(rng.choice([0.005, 0.01, 0.015, 0.025, 0.04, 0.08])),
        "vwap_lookback": int(rng.choice([24, 48, 72, 96, 120, 168, 240])),
    }


def prepare_features(
    base: pd.DataFrame,
    params: dict[str, Any],
    cache: dict[tuple[int, float], pd.DataFrame],
) -> pd.DataFrame:
    featured = base.copy()
    profile_key = (int(params["profile_lookback"]), float(params["profile_value_area"]))
    if profile_key not in cache:
        cache[profile_key] = profile_features(base, profile_key[0], profile_key[1])
    featured = featured.join(cache[profile_key], how="left")
    featured["vwap"] = rolling_vwap(featured, int(params["vwap_lookback"]))
    typical = (featured["high"] + featured["low"] + featured["close"]) / 3.0
    session = featured.index.floor("D")
    session_volume = featured["volume"].groupby(session).cumsum()
    featured["session_vwap"] = (typical * featured["volume"]).groupby(session).cumsum() / session_volume.replace(0, np.nan)
    close = featured["close"]
    featured["ema_fast_vp"] = ema(close, int(params["ema_fast"]))
    featured["ema_slow_vp"] = ema(close, int(params["ema_slow"]))
    featured["momentum_vp"] = close.pct_change(int(params["mom_lookback"]))
    featured["trend_slope_vp"] = featured["ema_slow_vp"].pct_change(int(params["slope_bars"]))
    vol_sma = featured["volume"].rolling(int(params["volume_lookback"]), min_periods=1).mean()
    featured["volume_ratio_vp"] = (featured["volume"] / vol_sma.replace(0, np.nan)).fillna(0)
    return featured


def apply_strategy(featured: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    df = featured.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"].replace(0, np.nan)
    vwap = df["vwap"]
    session_vwap = df["session_vwap"]
    poc = df["profile_poc"]
    val = df["profile_val"]
    vah = df["profile_vah"]
    fast = df["ema_fast_vp"]
    slow = df["ema_slow_vp"]
    momentum = df["momentum_vp"]
    uptrend = (fast > slow) & (df["trend_slope_vp"] > float(params["slope_min"]))
    downtrend = (fast < slow) & (df["trend_slope_vp"] < -float(params["slope_min"]))
    trend_strength = ((fast - slow).abs() / slow.replace(0, np.nan)).fillna(0)
    range_ok = trend_strength <= float(params["trend_strength_max"])
    volume_ok = df["volume_ratio_vp"] >= float(params["volume_min"])
    volatility_ok = (df["daily_vol"] >= float(params["volatility_min"])) & (df["daily_vol"] <= float(params["volatility_max"]))
    near_vwap = ((close - vwap).abs() / close).fillna(1.0) <= float(params["vwap_distance_max"])
    near_session_vwap = ((close - session_vwap).abs() / close).fillna(1.0) <= float(params["vwap_distance_max"])
    near_poc = ((close - poc).abs() / close).fillna(1.0) <= float(params["poc_distance_max"])
    vwap_upper = vwap + float(params["vwap_band_atr"]) * atr
    vwap_lower = vwap - float(params["vwap_band_atr"]) * atr
    session_upper = session_vwap + float(params["vwap_band_atr"]) * atr
    session_lower = session_vwap - float(params["vwap_band_atr"]) * atr
    signal = pd.Series(0, index=df.index, dtype=int)
    stop_loss = pd.Series(np.nan, index=df.index, dtype=float)
    take_profit = pd.Series(np.nan, index=df.index, dtype=float)
    family = str(params["family"])

    if family == "vwap_reclaim":
        long_event = uptrend & (close.shift(1) < vwap.shift(1)) & (close > vwap) & (momentum > float(params["mom_min"]))
        short_event = downtrend & (close.shift(1) > vwap.shift(1)) & (close < vwap) & (momentum < -float(params["mom_min"]))
    elif family == "vwap_pullback":
        long_event = uptrend & (low <= vwap_upper) & (close > vwap) & near_vwap & (df["rsi_14"] >= 42)
        short_event = downtrend & (high >= vwap_lower) & (close < vwap) & near_vwap & (df["rsi_14"] <= 58)
    elif family == "daily_vwap_pullback":
        long_event = uptrend & (low <= session_upper) & (close > session_vwap) & near_session_vwap & (close > df["open"])
        short_event = downtrend & (high >= session_lower) & (close < session_vwap) & near_session_vwap & (close < df["open"])
    elif family == "vwap_band_reversion":
        long_event = range_ok & (low <= vwap_lower) & (close > vwap_lower) & (close < poc) & (df["rsi_14"] <= float(params["rsi_low"]) + 8)
        short_event = range_ok & (high >= vwap_upper) & (close < vwap_upper) & (close > poc) & (df["rsi_14"] >= float(params["rsi_high"]) - 8)
    elif family == "profile_value_reject":
        long_event = range_ok & (low <= val) & (close > val) & near_poc & (df["rsi_14"] <= float(params["rsi_low"]) + 12)
        short_event = range_ok & (high >= vah) & (close < vah) & near_poc & (df["rsi_14"] >= float(params["rsi_high"]) - 12)
    elif family == "profile_breakout":
        long_event = uptrend & (close > vah) & (close.shift(1) <= vah.shift(1)) & (momentum > float(params["mom_min"]))
        short_event = downtrend & (close < val) & (close.shift(1) >= val.shift(1)) & (momentum < -float(params["mom_min"]))
    elif family == "profile_poc_reversion":
        long_event = range_ok & (close < poc) & (low <= val) & (close > val) & (df["rsi_14"] <= float(params["rsi_low"]) + 15)
        short_event = range_ok & (close > poc) & (high >= vah) & (close < vah) & (df["rsi_14"] >= float(params["rsi_high"]) - 15)
    elif family == "vwap_value_rotation":
        long_event = (uptrend | range_ok) & (low <= val) & (close > session_vwap) & (close < vah) & (momentum > -float(params["mom_min"]))
        short_event = (downtrend | range_ok) & (high >= vah) & (close < session_vwap) & (close > val) & (momentum < float(params["mom_min"]))
    elif family == "vwap_profile_confluence":
        long_event = uptrend & (close > vwap) & (close > poc) & near_poc & (momentum > 0)
        short_event = downtrend & (close < vwap) & (close < poc) & near_poc & (momentum < 0)
    elif family == "momentum_volume_breakout":
        channel_high = high.rolling(int(params["donchian_lookback"]), min_periods=1).max().shift(1)
        channel_low = low.rolling(int(params["donchian_lookback"]), min_periods=1).min().shift(1)
        long_event = uptrend & (close > channel_high) & (close > vwap) & (momentum > float(params["mom_min"]))
        short_event = downtrend & (close < channel_low) & (close < vwap) & (momentum < -float(params["mom_min"]))
    else:
        raise ValueError(f"Unknown family: {family}")

    long_event = long_event & volume_ok & volatility_ok & bool(params["allow_long"])
    short_event = short_event & volume_ok & volatility_ok & bool(params["allow_short"])
    signal.loc[long_event] = 1
    signal.loc[short_event] = -1
    stop_loss.loc[long_event] = close - float(params["atr_stop_mult"]) * atr
    stop_loss.loc[short_event] = close + float(params["atr_stop_mult"]) * atr
    take_profit.loc[long_event] = close + float(params["atr_tp_mult"]) * atr
    take_profit.loc[short_event] = close - float(params["atr_tp_mult"]) * atr
    target_mode = str(params.get("target_mode", "atr"))
    if bool(params["use_poc_target"]) or target_mode == "poc":
        take_profit.loc[long_event] = poc.loc[long_event].where(poc.loc[long_event] > close.loc[long_event], take_profit.loc[long_event])
        take_profit.loc[short_event] = poc.loc[short_event].where(poc.loc[short_event] < close.loc[short_event], take_profit.loc[short_event])
    if target_mode == "value_area":
        take_profit.loc[long_event] = vah.loc[long_event].where(vah.loc[long_event] > close.loc[long_event], take_profit.loc[long_event])
        take_profit.loc[short_event] = val.loc[short_event].where(val.loc[short_event] < close.loc[short_event], take_profit.loc[short_event])
    if target_mode == "session_vwap":
        take_profit.loc[long_event] = session_vwap.loc[long_event].where(
            session_vwap.loc[long_event] > close.loc[long_event], take_profit.loc[long_event]
        )
        take_profit.loc[short_event] = session_vwap.loc[short_event].where(
            session_vwap.loc[short_event] < close.loc[short_event], take_profit.loc[short_event]
        )

    df["signal"] = signal
    df["stop_loss"] = stop_loss
    df["take_profit"] = take_profit
    return df.dropna(subset=["atr_14", "daily_vol", "vwap", "session_vwap", "profile_poc", "profile_val", "profile_vah"])


def score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 4:
        return -1000 + trades
    win = float(metrics.get("win_rate", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 8.0)
    exp = float(metrics.get("expectancy_r", 0))
    dd = float(metrics.get("max_drawdown_r", 0))
    streak = float(metrics.get("max_consecutive_losses", 0))
    freq = min(trades_per_week(metrics), args.min_trades_per_week * 4)
    if exp <= 0 or pf < 1.2:
        return win * 45 + exp * 220 + (pf - 1) * 25 - dd * 2.5 - streak * 3
    return win * 90 + exp * 240 + (pf - 1.0) * 38 + freq * 5 - dd * 3.5 - streak * 4


def aggregate_score(window_metrics: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    total_trades = sum(int(item.get("total_trades", 0)) for item in window_metrics)
    if total_trades < args.min_train_trades:
        return -1_000_000 + total_trades, {}
    exp_values = [float(item.get("expectancy_r", 0)) for item in window_metrics]
    win_values = [float(item.get("win_rate", 0)) for item in window_metrics]
    pf_values = [float(item.get("profit_factor", 0)) for item in window_metrics]
    aggregate = {
        "wf_windows": len(window_metrics),
        "wf_gate_windows": sum(1 for item in window_metrics if deployment_gate(item, args)),
        "wf_positive_exp": sum(1 for item in window_metrics if float(item.get("expectancy_r", 0)) > 0),
        "wf_mean_win_rate": sum(win_values) / len(win_values),
        "wf_mean_exp_r": sum(exp_values) / len(exp_values),
        "wf_mean_pf": sum(min(value, 10.0) for value in pf_values) / len(pf_values),
        "wf_max_dd_r": max(float(item.get("max_drawdown_r", 0)) for item in window_metrics),
        "wf_total_trades": total_trades,
    }
    total_score = (
        sum(score(item, args) for item in window_metrics) / len(window_metrics)
        + aggregate["wf_positive_exp"] / len(window_metrics) * 110
        + aggregate["wf_gate_windows"] / len(window_metrics) * 180
        + aggregate["wf_mean_exp_r"] * 130
    )
    return total_score, aggregate


def summarize_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    total = sum(row["total_trades"] for row in rows)
    weighted = lambda field: sum(row[field] * row["total_trades"] for row in rows) / total if total else 0
    return {
        "windows": len(rows),
        "gate_windows": sum(1 for row in rows if row["gate_pass"]),
        "positive_expectancy_windows": sum(1 for row in rows if row["expectancy_r"] > 0),
        "mean_win_rate": sum(row["win_rate"] for row in rows) / len(rows),
        "mean_expectancy_r": sum(row["expectancy_r"] for row in rows) / len(rows),
        "trade_weighted_expectancy_r": weighted("expectancy_r"),
        "max_drawdown_r": max(row["max_drawdown_r"] for row in rows),
        "max_drawdown_pct": max(row["max_drawdown_pct"] for row in rows),
        "max_consecutive_losses": max(row["max_consecutive_losses"] for row in rows),
        "total_trades": total,
        "weighted_trades_per_week": weighted("trades_per_day") * 7,
        "weighted_avg_win_r": weighted("avg_win_r"),
        "weighted_avg_loss_r": weighted("avg_loss_r"),
        "weighted_payoff_ratio": weighted("payoff_ratio"),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("data.loader").setLevel(logging.WARNING)
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"vwap_profile_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    profile_cache: dict[tuple[int, float], pd.DataFrame] = {}

    logger.info("Searching %d VWAP/profile/momentum candidates", args.trials)
    for trial in range(args.trials):
        params = sample_params(rng, trial)
        prepared = prepare_features(base, params, profile_cache)
        featured = apply_strategy(prepared, params)
        metrics = [
            run_metrics(featured.loc[(featured.index >= test_start) & (featured.index <= test_end)], args.symbol, args, params)
            for _, _, test_start, test_end in train_windows
        ]
        wf_score, aggregate = aggregate_score(metrics, args)
        rows.append({"trial": trial, "wf_score": wf_score, **{field: params.get(field) for field in PARAM_FIELDS}, **aggregate})
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
    for row in top_rows:
        params = {field: row[field] for field in PARAM_FIELDS}
        prepared = prepare_features(base, params, profile_cache)
        featured = apply_strategy(prepared, params)
        validation_slice = featured.loc[(featured.index >= val_start) & (featured.index <= val_end)]
        validation = run_metrics(validation_slice, args.symbol, args, params)
        stress = run_metrics(validation_slice, args.symbol, args, params, args.stress_slippage_rate)
        for prefix, metrics in [("validation", validation), ("stress", stress)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_trades_per_week"] = trades_per_week(metrics)
        row["validation_pass"] = deployment_gate(validation, args)
        row["stress_pass"] = deployment_gate(stress, args)
        row["deploy_score"] = (
            (500 if row["stress_pass"] else 0)
            + (250 if row["validation_pass"] else 0)
            + row["stress_win_rate"] * 110
            + min(row["stress_profit_factor"], 10.0) * 35
            + row["stress_expectancy_r"] * 210
            - row["stress_max_drawdown_r"] * 4
            - row["stress_max_consecutive_losses"] * 5
            + row["wf_score"] * 0.35
        )

    top_rows.sort(key=lambda item: item["deploy_score"], reverse=True)
    best = top_rows[0] if top_rows else rows[0]
    params = {field: best[field] for field in PARAM_FIELDS}
    prepared = prepare_features(base, params, profile_cache)
    featured = apply_strategy(prepared, params)
    wf_rows = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(args.start, args.validation_end, args.train_months, args.test_months),
        start=1,
    ):
        metrics = run_metrics(featured.loc[(featured.index >= test_start) & (featured.index <= test_end)], args.symbol, args, params)
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
        "strategy": "vwap_volume_profile_momentum_search",
        "seed": args.seed,
        "trials": args.trials,
        "best": best,
        "best_validation_ranked": top_rows[0] if top_rows else {},
        "aggregate": summarize_windows(wf_rows),
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
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
        "Saved to %s | best %s rank %s stress pass %s win %.1f%% PF %.2f expR %.3f ddR %.2f trades/week %.2f WF positive %s/%s",
        run_dir,
        best.get("family"),
        best.get("wf_rank"),
        best.get("stress_pass", False),
        best.get("stress_win_rate", 0) * 100,
        best.get("stress_profit_factor", 0),
        best.get("stress_expectancy_r", 0),
        best.get("stress_max_drawdown_r", 0),
        best.get("stress_trades_per_week", 0),
        summary["aggregate"].get("positive_expectancy_windows", 0),
        summary["aggregate"].get("windows", 0),
    )


if __name__ == "__main__":
    main()
