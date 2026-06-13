#!/usr/bin/env python3
"""Replay deployment execution on historical candles and compare to backtest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import DataLoader
from deployment.binance_futures import BinanceFuturesClient
from deployment.live_replay import live_replay_metrics
from deployment.replay_engine import ReplayConfig, replay_metrics
from deployment.strategy_runtime import CandidateRuntime
from scripts.explore_high_winrate_edge import trades_per_week
from scripts.explore_trailing_highwin_edge import simulate_trades
from scripts.explore_trailing_highwin_edge import metrics_from_trades

logger = logging.getLogger("replay_deployment_proxy")

DEFAULT_CANDIDATE = PROJECT_ROOT / "research/optimization/vwap_profile_SOLUSDT_1h_20260609_101440/best_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay deployment engine against historical proxy feed")
    parser.add_argument("--source", choices=["binance", "local"], default=os.getenv("REPLAY_SOURCE", "binance"))
    parser.add_argument("--exchange", choices=["mainnet", "demo"], default=os.getenv("BINANCE_EXCHANGE", "mainnet"))
    parser.add_argument("--candidate-summary", type=Path, default=Path(os.getenv("CANDIDATE_SUMMARY", DEFAULT_CANDIDATE)))
    parser.add_argument("--candidate-section", default=os.getenv("CANDIDATE_SECTION", "best"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("REPLAY_OUTPUT_DIR", "runtime/replay_proxy")))
    parser.add_argument("--balance", type=float, default=float(os.getenv("DEMO_INITIAL_BALANCE", "100000")))
    parser.add_argument("--risk-per-trade", type=float, default=float(os.getenv("RISK_PER_TRADE", "0.01")))
    parser.add_argument("--max-position-pct", type=float, default=float(os.getenv("MAX_POSITION_PCT", "1.0")))
    parser.add_argument("--fee-rate", type=float, default=float(os.getenv("FEE_RATE", "0.0005")))
    parser.add_argument("--slippage-rate", type=float, default=float(os.getenv("SLIPPAGE_RATE", "0.001")))
    parser.add_argument("--kline-limit", type=int, default=int(os.getenv("KLINE_LIMIT", "600")))
    parser.add_argument("--start", default=os.getenv("REPLAY_START", "2024-01-01"))
    parser.add_argument("--end", default=os.getenv("REPLAY_END", "2025-06-01"))
    parser.add_argument("--data-start", default=os.getenv("REPLAY_DATA_START"))
    parser.add_argument("--data-end", default=os.getenv("REPLAY_DATA_END"))
    parser.add_argument("--eval-start", default=os.getenv("REPLAY_EVAL_START"))
    parser.add_argument("--eval-end", default=os.getenv("REPLAY_EVAL_END"))
    parser.add_argument("--engine", choices=["model", "live", "both"], default=os.getenv("REPLAY_ENGINE", "model"))
    parser.add_argument("--quantity-mode", choices=["exact", "rounded"], default="exact")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def load_raw(args: argparse.Namespace, runtime: CandidateRuntime) -> pd.DataFrame:
    if args.source == "binance":
        client = BinanceFuturesClient.from_exchange(args.exchange)
        return client.closed_klines(runtime.symbol, runtime.timeframe, limit=args.kline_limit)
    loader = DataLoader(use_cache=True)
    raw = loader.load_symbol(runtime.symbol, "1m")
    if raw.index.tz is None:
        raw = raw.tz_localize("UTC")
    start = pd.Timestamp(args.data_start or args.start, tz="UTC")
    end = pd.Timestamp(args.data_end or args.end, tz="UTC")
    return raw.loc[(raw.index >= start) & (raw.index <= end), ["open", "high", "low", "close", "volume"]]


def evaluation_slice(featured: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.source != "local" and not (args.eval_start or args.eval_end):
        return featured
    start = pd.Timestamp(args.eval_start or args.start, tz="UTC")
    end = pd.Timestamp(args.eval_end or args.end, tz="UTC")
    return featured.loc[(featured.index >= start) & (featured.index <= end)]


def write_trades(path: Path, trades: list[object]) -> None:
    fields = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "quantity",
        "notional",
        "stop_loss",
        "take_profit",
        "exit_reason",
        "pnl",
        "pnl_pct",
        "risk_amount",
        "r_multiple",
        "fees",
        "slippage",
        "holding_bars",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: getattr(trade, field, None) for field in fields})


def trade_rows(trades: list[object]) -> list[dict[str, Any]]:
    fields = ["entry_time", "exit_time", "side", "exit_reason", "entry_price", "exit_price", "r_multiple", "pnl"]
    return [{field: str(getattr(trade, field, None)) for field in fields} for trade in trades]


def max_trade_delta(a: list[object], b: list[object], field: str) -> float:
    if len(a) != len(b):
        return float("inf")
    values = []
    for left, right in zip(a, b):
        try:
            values.append(abs(float(getattr(left, field)) - float(getattr(right, field))))
        except (TypeError, ValueError):
            values.append(float("inf"))
    return max(values) if values else 0.0


def backtest_from_featured(featured: pd.DataFrame, runtime: CandidateRuntime, balance: float) -> tuple[list[object], dict[str, Any]]:
    args = runtime.args(balance=balance)
    trades = simulate_trades(featured, runtime.symbol, args, runtime.params, runtime.slippage_rate)
    metrics = metrics_from_trades(trades, featured.index, args.balance)
    metrics["trades_per_week"] = trades_per_week(metrics)
    return trades, metrics


def compare(backtest_trades: list[object], replay_trades: list[object], backtest_metrics: dict[str, Any], replay_metrics_data: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for i, (left, right) in enumerate(zip(backtest_trades, replay_trades), start=1):
        for field in ["entry_time", "exit_time", "side", "exit_reason"]:
            if str(getattr(left, field)) != str(getattr(right, field)):
                mismatches.append({"trade": i, "field": field, "backtest": str(getattr(left, field)), "replay": str(getattr(right, field))})
        for field in ["entry_price", "exit_price", "quantity", "pnl", "r_multiple"]:
            if abs(float(getattr(left, field)) - float(getattr(right, field))) > 1e-7:
                mismatches.append({"trade": i, "field": field, "backtest": getattr(left, field), "replay": getattr(right, field)})
    return {
        "trade_count_match": len(backtest_trades) == len(replay_trades),
        "backtest_trade_count": len(backtest_trades),
        "replay_trade_count": len(replay_trades),
        "metric_deltas": {
            "total_trades": int(replay_metrics_data.get("total_trades", 0)) - int(backtest_metrics.get("total_trades", 0)),
            "win_rate": float(replay_metrics_data.get("win_rate", 0)) - float(backtest_metrics.get("win_rate", 0)),
            "profit_factor": float(replay_metrics_data.get("profit_factor", 0) or 0) - float(backtest_metrics.get("profit_factor", 0) or 0),
            "expectancy_r": float(replay_metrics_data.get("expectancy_r", 0) or 0) - float(backtest_metrics.get("expectancy_r", 0) or 0),
            "max_drawdown_r": float(replay_metrics_data.get("max_drawdown_r", 0) or 0) - float(backtest_metrics.get("max_drawdown_r", 0) or 0),
        },
        "max_trade_deltas": {
            "entry_price": max_trade_delta(backtest_trades, replay_trades, "entry_price"),
            "exit_price": max_trade_delta(backtest_trades, replay_trades, "exit_price"),
            "pnl": max_trade_delta(backtest_trades, replay_trades, "pnl"),
            "r_multiple": max_trade_delta(backtest_trades, replay_trades, "r_multiple"),
        },
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    runtime = CandidateRuntime.from_summary(
        args.candidate_summary,
        section=args.candidate_section,
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
    )
    raw = load_raw(args, runtime)
    featured_all = runtime.feature_frame(raw)
    featured = evaluation_slice(featured_all, args)
    backtest_trades, backtest_metrics = backtest_from_featured(featured, runtime, args.balance)
    config = ReplayConfig(exact_quantity=args.quantity_mode == "exact")
    replay_trades = []
    live_trades = []
    replay_metrics_data: dict[str, Any] = {}
    live_metrics_data: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if args.engine in {"model", "both"}:
        replay_trades, replay_metrics_data = replay_metrics(featured, runtime, config=config)
        comparisons["model"] = compare(backtest_trades, replay_trades, backtest_metrics, replay_metrics_data)
    if args.engine in {"live", "both"}:
        live_trades, live_metrics_data = live_replay_metrics(featured, runtime)
        comparisons["live"] = compare(backtest_trades, live_trades, backtest_metrics, live_metrics_data)
    primary = "live" if args.engine == "live" else "model"
    comparison = comparisons[primary]
    result = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "source": args.source,
        "exchange": args.exchange if args.source == "binance" else None,
        "engine": args.engine,
        "candidate": runtime.name,
        "symbol": runtime.symbol,
        "timeframe": runtime.timeframe,
        "raw_start": raw.index[0].isoformat() if len(raw) else None,
        "raw_end": raw.index[-1].isoformat() if len(raw) else None,
        "feature_warmup_start": featured_all.index[0].isoformat() if len(featured_all) else None,
        "feature_warmup_end": featured_all.index[-1].isoformat() if len(featured_all) else None,
        "featured_start": featured.index[0].isoformat() if len(featured) else None,
        "featured_end": featured.index[-1].isoformat() if len(featured) else None,
        "backtest_metrics": backtest_metrics,
        "replay_metrics": replay_metrics_data,
        "live_replay_metrics": live_metrics_data,
        "comparison": comparison,
        "comparisons": comparisons,
        "backtest_trades_head": trade_rows(backtest_trades[:5]),
        "replay_trades_head": trade_rows(replay_trades[:5]),
        "live_replay_trades_head": trade_rows(live_trades[:5]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trades(args.output_dir / "backtest_trades.csv", backtest_trades)
    if replay_trades:
        write_trades(args.output_dir / "replay_trades.csv", replay_trades)
    if live_trades:
        write_trades(args.output_dir / "live_replay_trades.csv", live_trades)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    logger.info(
        "Replay parity source=%s engine=%s backtest=%s replay=%s mismatch=%s expR %.6f/%.6f PF %.6f/%.6f",
        args.source,
        args.engine,
        len(backtest_trades),
        comparison["replay_trade_count"],
        comparison["mismatch_count"],
        float(backtest_metrics.get("expectancy_r", 0) or 0),
        float((live_metrics_data if primary == "live" else replay_metrics_data).get("expectancy_r", 0) or 0),
        float(backtest_metrics.get("profit_factor", 0) or 0),
        float((live_metrics_data if primary == "live" else replay_metrics_data).get("profit_factor", 0) or 0),
    )
    if args.fail_on_mismatch and any(
        not item["trade_count_match"] or item["mismatch_count"] > 0
        for item in comparisons.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
