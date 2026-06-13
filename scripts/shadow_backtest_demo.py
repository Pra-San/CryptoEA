#!/usr/bin/env python3
"""Run the exact research/backtest simulator beside deployment for comparison."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.binance_futures import BinanceFuturesClient
from deployment.state import SQLiteStateStore
from deployment.strategy_runtime import CandidateRuntime

logger = logging.getLogger("shadow_backtest_demo")

DEFAULT_CANDIDATE = PROJECT_ROOT / "research/optimization/vwap_profile_SOLUSDT_1h_20260609_101440/best_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact backtest shadow process on live Binance candles")
    parser.add_argument("--exchange", choices=["mainnet", "demo"], default=os.getenv("BINANCE_EXCHANGE", "mainnet"))
    parser.add_argument("--candidate-summary", type=Path, default=Path(os.getenv("CANDIDATE_SUMMARY", DEFAULT_CANDIDATE)))
    parser.add_argument("--candidate-section", default=os.getenv("CANDIDATE_SECTION", "best"))
    parser.add_argument("--state-db", type=Path, default=Path(os.getenv("CRYPTOEA_STATE_DB", "runtime/sol_momentum_deploy.sqlite")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("SHADOW_OUTPUT_DIR", "runtime/shadow_backtest")))
    parser.add_argument("--balance", type=float, default=float(os.getenv("DEMO_INITIAL_BALANCE", "100000")))
    parser.add_argument("--risk-per-trade", type=float, default=float(os.getenv("RISK_PER_TRADE", "0.01")))
    parser.add_argument("--max-position-pct", type=float, default=float(os.getenv("MAX_POSITION_PCT", "1.0")))
    parser.add_argument("--fee-rate", type=float, default=float(os.getenv("FEE_RATE", "0.0005")))
    parser.add_argument("--slippage-rate", type=float, default=float(os.getenv("SLIPPAGE_RATE", "0.001")))
    parser.add_argument("--kline-limit", type=int, default=int(os.getenv("KLINE_LIMIT", "1500")))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("SHADOW_POLL_SECONDS", "300")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def write_trades(path: Path, trades: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def run_once(args: argparse.Namespace) -> None:
    client = BinanceFuturesClient.from_exchange(args.exchange)
    runtime = CandidateRuntime.from_summary(
        args.candidate_summary,
        section=args.candidate_section,
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
    )
    raw = client.closed_klines(runtime.symbol, runtime.timeframe, limit=args.kline_limit)
    trades, metrics = runtime.backtest_snapshot(raw, balance=args.balance)
    store = SQLiteStateStore(args.state_db)
    deployment_summary = store.closed_trade_summary()
    shadow_total_pnl = sum(float(getattr(trade, "pnl", 0.0) or 0.0) for trade in trades)
    comparison = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "symbol": runtime.symbol,
        "timeframe": runtime.timeframe,
        "candidate": runtime.name,
        "exchange": args.exchange,
        "comparison_scope": "shadow is a rolling-window historical backtest; deployment_demo_summary is forward-only state since the demo database was created",
        "closed_kline_start": raw.index[0].isoformat() if len(raw) else None,
        "closed_kline_end": raw.index[-1].isoformat() if len(raw) else None,
        "shadow_metrics": metrics,
        "shadow_total_pnl": shadow_total_pnl,
        "deployment_demo_summary": deployment_summary,
        "differences": {
            "trade_count": deployment_summary.get("closed_trades", 0) - int(metrics.get("total_trades", 0)),
            "total_pnl": deployment_summary.get("total_pnl", 0.0) - shadow_total_pnl,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trades(args.output_dir / "shadow_trades.csv", trades)
    (args.output_dir / "shadow_metrics.json").write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str))
    logger.info(
        "Shadow rolling-window complete trades=%s win=%.2f%% pf=%.2f expR=%.3f deploy_closed_since_state_start=%s",
        metrics.get("total_trades", 0),
        float(metrics.get("win_rate", 0)) * 100,
        float(metrics.get("profit_factor", 0) or 0),
        float(metrics.get("expectancy_r", 0) or 0),
        deployment_summary.get("closed_trades", 0),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    while True:
        run_once(args)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
