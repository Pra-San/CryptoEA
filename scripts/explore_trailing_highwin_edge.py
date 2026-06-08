#!/usr/bin/env python3
"""Search high-winrate variants with breakeven/trailing trade management."""

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
from scripts.explore_high_winrate_edge import (
    PARAM_FIELDS as ENTRY_PARAM_FIELDS,
    apply_strategy,
    deployment_gate,
    load_data,
    sample_params,
    trades_per_week,
    walk_forward_windows,
    write_csv,
)


logger = logging.getLogger("trailing_highwin_explorer")

PARAM_FIELDS = ENTRY_PARAM_FIELDS + [
    "break_even_atr",
    "move_stop_after_partial",
    "partial_exit_atr",
    "partial_exit_fraction",
    "trail_atr_mult",
    "use_break_even",
    "use_partial_exit",
    "use_trailing_stop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore high-winrate trailing-stop variants")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--stress-slippage-rate", type=float, default=0.0050)
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-validation-trades", type=int, default=80)
    parser.add_argument("--min-trades-per-week", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def sample_trailing_params(rng: np.random.Generator, trial: int) -> dict[str, Any]:
    params = sample_params(rng, trial)
    params["break_even_atr"] = float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))
    params["partial_exit_atr"] = float(rng.choice([0.6, 0.8, 1.0, 1.2, 1.5, 2.0]))
    params["partial_exit_fraction"] = float(rng.choice([0.25, 0.33, 0.5, 0.67]))
    params["trail_atr_mult"] = float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]))
    params["use_break_even"] = bool(rng.choice([True, False], p=[0.85, 0.15]))
    params["use_partial_exit"] = bool(rng.choice([True, False], p=[0.75, 0.25]))
    params["use_trailing_stop"] = bool(rng.choice([True, False], p=[0.70, 0.30]))
    params["move_stop_after_partial"] = bool(rng.choice([True, False], p=[0.90, 0.10]))
    params["exit_on_flat_signal"] = bool(params["exit_on_flat_signal"]) or params["use_trailing_stop"]
    return params


def empty_metrics() -> dict[str, Any]:
    calc = BacktestMetrics()
    return calc.to_dict(calc.calculate(SimpleNamespace(trades=[])))


def build_equity(trades: list[SimpleNamespace], timeline: pd.DatetimeIndex, balance: float) -> pd.Series:
    equity = balance
    pnl_by_time: dict[pd.Timestamp, float] = {}
    for trade in trades:
        ts = pd.Timestamp(trade.exit_time)
        pnl_by_time[ts] = pnl_by_time.get(ts, 0.0) + float(trade.pnl)
    values = []
    for ts in timeline:
        equity += pnl_by_time.get(pd.Timestamp(ts), 0.0)
        values.append(equity)
    return pd.Series(values, index=timeline, name="equity")


def metrics_from_trades(trades: list[SimpleNamespace], timeline: pd.DatetimeIndex, balance: float) -> dict[str, Any]:
    calc = BacktestMetrics()
    result = SimpleNamespace(
        trades=trades,
        equity_curve=build_equity(trades, timeline, balance),
        benchmark_curve=None,
        config=SimpleNamespace(initial_balance=balance),
        metadata={"start_date": str(timeline[0]) if len(timeline) else "", "end_date": str(timeline[-1]) if len(timeline) else ""},
    )
    return calc.to_dict(calc.calculate(result))


def position_size(
    entry_price: float,
    stop_loss: float,
    balance: float,
    args: argparse.Namespace,
    slippage_rate: float,
) -> tuple[float, float, float]:
    stop_fill = stop_loss * (1 - slippage_rate) if stop_loss < entry_price else stop_loss * (1 + slippage_rate)
    unit_risk = abs(entry_price - stop_fill) + 2 * entry_price * args.fee_rate
    if unit_risk <= 0:
        return 0.0, 0.0, 0.0
    risk_amount = balance * args.risk_per_trade
    quantity = risk_amount / unit_risk
    notional = quantity * entry_price
    max_notional = balance * args.max_position_pct
    if notional > max_notional:
        quantity = max_notional / entry_price
        notional = max_notional
        risk_amount = unit_risk * quantity
    return quantity, notional, risk_amount


def simulate_trades(
    featured: pd.DataFrame,
    symbol: str,
    args: argparse.Namespace,
    params: dict[str, Any],
    slippage_rate: float | None = None,
) -> list[SimpleNamespace]:
    if featured.empty:
        return []
    slip = args.slippage_rate if slippage_rate is None else slippage_rate
    signals = featured["signal"].astype(int).to_numpy()
    previous = np.zeros(len(signals), dtype=int)
    previous[1:] = signals[:-1]
    signal_bars = np.where((signals != 0) & (signals != previous))[0]
    entry_bars = signal_bars + 1
    valid = entry_bars < len(signals)
    pairs = list(zip(signal_bars[valid], entry_bars[valid]))
    if not pairs:
        return []

    opens = featured["open"].to_numpy(dtype=float)
    highs = featured["high"].to_numpy(dtype=float)
    lows = featured["low"].to_numpy(dtype=float)
    closes = featured["close"].to_numpy(dtype=float)
    stops = featured["stop_loss"].to_numpy(dtype=float)
    tps = featured["take_profit"].to_numpy(dtype=float)
    atr = featured["atr_14"].replace(0, np.nan).to_numpy(dtype=float)
    timestamps = featured.index
    min_hold = int(params["min_holding_bars"])
    max_hold = int(params["max_holding_bars"])
    last_exit = -1
    balance = args.balance
    trades: list[SimpleNamespace] = []

    for signal_idx, entry_idx in pairs:
        if entry_idx <= last_exit:
            continue
        sig = int(signals[signal_idx])
        if sig == 0 or np.isnan(stops[signal_idx]):
            continue
        raw_entry = opens[entry_idx]
        entry = raw_entry * (1 + slip * sig)
        stop = float(stops[signal_idx])
        tp = None if np.isnan(tps[signal_idx]) else float(tps[signal_idx])
        entry_atr = float(atr[signal_idx]) if not np.isnan(atr[signal_idx]) else abs(entry - stop)
        quantity, notional, risk_amount = position_size(entry, stop, balance, args, slip)
        if quantity <= 0 or notional <= 0:
            continue
        entry_fee = quantity * entry * args.fee_rate
        best_price = entry
        exit_idx = None
        exit_price_raw = None
        exit_reason = "max_hold"
        fee_be = entry * (1 + sig * (2 * args.fee_rate + 2 * slip))
        remaining_quantity = quantity
        realized_pnl = -entry_fee
        total_fees = entry_fee
        slippage_cost = abs(entry - raw_entry) * quantity
        partial_taken = False
        partial_target = entry + sig * float(params["partial_exit_atr"]) * entry_atr
        partial_fraction = min(max(float(params["partial_exit_fraction"]), 0.0), 0.95)

        search_end = min(entry_idx + max_hold, len(closes) - 1)
        for i in range(entry_idx + 1, search_end + 1):
            if sig == 1:
                if lows[i] <= stop and i >= entry_idx + min_hold:
                    exit_idx, exit_price_raw, exit_reason = i, stop, "stop_loss"
                    break
                best_price = max(best_price, highs[i])
                if params["use_break_even"] and best_price >= entry + params["break_even_atr"] * entry_atr:
                    stop = max(stop, fee_be)
                if params["use_trailing_stop"]:
                    stop = max(stop, best_price - params["trail_atr_mult"] * entry_atr)
                if (
                    params["use_partial_exit"]
                    and not partial_taken
                    and highs[i] >= partial_target
                    and i >= entry_idx + min_hold
                ):
                    part_qty = quantity * partial_fraction
                    part_exit = partial_target * (1 - slip)
                    realized_pnl += (part_exit - entry) * part_qty - part_qty * part_exit * args.fee_rate
                    total_fees += part_qty * part_exit * args.fee_rate
                    slippage_cost += abs(part_exit - partial_target) * part_qty
                    remaining_quantity -= part_qty
                    partial_taken = True
                    if params["move_stop_after_partial"]:
                        stop = max(stop, fee_be)
                if tp is not None and highs[i] >= tp and i >= entry_idx + min_hold:
                    exit_idx, exit_price_raw, exit_reason = i, tp, "take_profit"
                    break
            else:
                if highs[i] >= stop and i >= entry_idx + min_hold:
                    exit_idx, exit_price_raw, exit_reason = i, stop, "stop_loss"
                    break
                best_price = min(best_price, lows[i])
                if params["use_break_even"] and best_price <= entry - params["break_even_atr"] * entry_atr:
                    stop = min(stop, fee_be)
                if params["use_trailing_stop"]:
                    stop = min(stop, best_price + params["trail_atr_mult"] * entry_atr)
                if (
                    params["use_partial_exit"]
                    and not partial_taken
                    and lows[i] <= partial_target
                    and i >= entry_idx + min_hold
                ):
                    part_qty = quantity * partial_fraction
                    part_exit = partial_target * (1 + slip)
                    realized_pnl += (entry - part_exit) * part_qty - part_qty * part_exit * args.fee_rate
                    total_fees += part_qty * part_exit * args.fee_rate
                    slippage_cost += abs(part_exit - partial_target) * part_qty
                    remaining_quantity -= part_qty
                    partial_taken = True
                    if params["move_stop_after_partial"]:
                        stop = min(stop, fee_be)
                if tp is not None and lows[i] <= tp and i >= entry_idx + min_hold:
                    exit_idx, exit_price_raw, exit_reason = i, tp, "take_profit"
                    break
            if i >= entry_idx + min_hold:
                if bool(params["exit_on_flat_signal"]) and signals[i] == 0:
                    exit_idx, exit_price_raw, exit_reason = i, closes[i], "flat_signal"
                    break
                if signals[i] == -sig:
                    exit_idx, exit_price_raw, exit_reason = i, closes[i], "opposite_signal"
                    break
        if exit_idx is None:
            exit_idx = search_end
            exit_price_raw = closes[exit_idx]
        if exit_idx < entry_idx + min_hold:
            continue
        raw_exit = float(exit_price_raw)
        exit_price = raw_exit * (1 - slip * sig)
        gross = (exit_price - entry) * remaining_quantity * sig
        exit_fee = remaining_quantity * exit_price * args.fee_rate
        pnl = realized_pnl + gross - exit_fee
        total_fees += exit_fee
        slippage_cost += abs(exit_price - raw_exit) * remaining_quantity
        balance += pnl
        last_exit = exit_idx
        if partial_taken:
            exit_reason = f"partial_then_{exit_reason}"
        trades.append(SimpleNamespace(
            entry_time=timestamps[entry_idx],
            exit_time=timestamps[exit_idx],
            side="LONG" if sig == 1 else "SHORT",
            entry_price=entry,
            exit_price=exit_price,
            quantity=quantity,
            notional=notional,
            stop_loss=stop,
            take_profit=tp,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl / notional if notional else 0.0,
            risk_amount=risk_amount,
            r_multiple=pnl / risk_amount if risk_amount else 0.0,
            fees=total_fees,
            slippage=slippage_cost,
            holding_bars=exit_idx - entry_idx,
            metadata={"symbol": symbol},
        ))
    return trades


def run_metrics(
    featured: pd.DataFrame,
    symbol: str,
    args: argparse.Namespace,
    params: dict[str, Any],
    slippage_rate: float | None = None,
) -> dict[str, Any]:
    if featured.empty:
        return empty_metrics()
    trades = simulate_trades(featured, symbol, args, params, slippage_rate)
    return metrics_from_trades(trades, featured.index, args.balance)


def score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < 4:
        return -100 + trades
    win = float(metrics.get("win_rate", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 10.0)
    exp = float(metrics.get("expectancy_r", 0))
    dd = float(metrics.get("max_drawdown_r", 0))
    streak = float(metrics.get("max_consecutive_losses", 0))
    freq = min(trades_per_week(metrics), args.min_trades_per_week * 2)
    if exp <= 0 or pf < 1.2:
        return win * 40 + exp * 220 + (pf - 1) * 30 - dd * 3 - streak * 4
    return win * 100 + exp * 220 + (pf - 1) * 35 + freq * 8 - dd * 3 - streak * 5


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
        + aggregate["wf_gate_windows"] / len(window_metrics) * 250
        + aggregate["wf_positive_exp"] / len(window_metrics) * 90
        + aggregate["wf_mean_exp_r"] * 120
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
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_dir = output_root / f"partial_highwin_{args.symbol}_{args.timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base = load_data(args.symbol, args.timeframe, args.start, args.validation_end)
    train_windows = list(walk_forward_windows(args.start, args.train_end, args.train_months, args.test_months))
    val_start = pd.Timestamp(args.validation_start, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    logger.info("Searching %d partial/trailing high-winrate candidates", args.trials)
    for trial in range(args.trials):
        params = sample_trailing_params(rng, trial)
        featured = apply_strategy(base, params)
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
        featured = apply_strategy(base, params)
        validation = run_metrics(featured.loc[(featured.index >= val_start) & (featured.index <= val_end)], args.symbol, args, params)
        stress = run_metrics(
            featured.loc[(featured.index >= val_start) & (featured.index <= val_end)],
            args.symbol,
            args,
            params,
            args.stress_slippage_rate,
        )
        for prefix, metrics in [("validation", validation), ("stress", stress)]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_trades_per_week"] = trades_per_week(metrics)
        row["validation_pass"] = deployment_gate(validation, args)
        row["stress_pass"] = deployment_gate(stress, args)
        row["deploy_score"] = (
            (500 if row["stress_pass"] else 0)
            + (300 if row["validation_pass"] else 0)
            + row["stress_win_rate"] * 120
            + min(row["stress_profit_factor"], 10.0) * 35
            + row["stress_expectancy_r"] * 180
            - row["stress_max_drawdown_r"] * 3
            - row["stress_max_consecutive_losses"] * 5
            + row["wf_score"] * 0.3
        )
    top_rows.sort(key=lambda item: item["deploy_score"], reverse=True)
    best = top_rows[0] if top_rows and top_rows[0].get("stress_pass") else rows[0]
    params = {field: best[field] for field in PARAM_FIELDS}
    featured = apply_strategy(base, params)
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
        "strategy": "partial_trailing_highwin_edge_search",
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
