#!/usr/bin/env python3
"""Run the selected SOLUSDT 1h momentum strategy in demo or live mode."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.binance_futures import BinanceFuturesClient
from deployment.brokers import DemoBroker, LiveBinanceBroker, RuntimePosition
from deployment.state import SQLiteStateStore
from deployment.strategy_runtime import CandidateRuntime, feature_tail_payload

logger = logging.getLogger("sol_momentum_deploy")

DEFAULT_CANDIDATE = PROJECT_ROOT / "research/optimization/vwap_profile_SOLUSDT_1h_20260609_101440/best_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy SOLUSDT 1h momentum-volume strategy")
    parser.add_argument("--mode", choices=["demo", "live"], default=os.getenv("CRYPTOEA_MODE", "demo"))
    parser.add_argument("--exchange", choices=["mainnet", "demo"], default=os.getenv("BINANCE_EXCHANGE", "mainnet"))
    parser.add_argument("--candidate-summary", type=Path, default=Path(os.getenv("CANDIDATE_SUMMARY", DEFAULT_CANDIDATE)))
    parser.add_argument("--candidate-section", default=os.getenv("CANDIDATE_SECTION", "best"))
    parser.add_argument("--state-db", type=Path, default=Path(os.getenv("CRYPTOEA_STATE_DB", "runtime/sol_momentum_deploy.sqlite")))
    parser.add_argument("--balance", type=float, default=float(os.getenv("DEMO_INITIAL_BALANCE", "100000")))
    parser.add_argument("--risk-per-trade", type=float, default=float(os.getenv("RISK_PER_TRADE", "0.01")))
    parser.add_argument("--max-position-pct", type=float, default=float(os.getenv("MAX_POSITION_PCT", "1.0")))
    parser.add_argument("--fee-rate", type=float, default=float(os.getenv("FEE_RATE", "0.0005")))
    parser.add_argument("--slippage-rate", type=float, default=float(os.getenv("SLIPPAGE_RATE", "0.001")))
    parser.add_argument("--max-notional-usdt", type=float, default=float(os.getenv("MAX_NOTIONAL_USDT", "1000")))
    parser.add_argument("--kline-limit", type=int, default=int(os.getenv("KLINE_LIMIT", "600")))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("POLL_SECONDS", "60")))
    parser.add_argument("--once", action="store_true", help="Run one reconciliation cycle and exit")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def load_active_position(store: SQLiteStateStore) -> RuntimePosition | None:
    payload = store.get_json("active_position")
    if not payload:
        return None
    fields = RuntimePosition.__dataclass_fields__
    return RuntimePosition(**{key: value for key, value in payload.items() if key in fields})


def save_active_position(store: SQLiteStateStore, position: RuntimePosition | None) -> None:
    store.set_json("active_position", asdict(position) if position else None)


def latest_signal_from_featured(featured: pd.DataFrame) -> dict[str, Any] | None:
    if len(featured) < 3:
        return None
    current = featured.iloc[-1]
    previous = featured.iloc[-2]
    signal = int(current["signal"])
    previous_signal = int(previous["signal"])
    base = {
        "signal": signal,
        "previous_signal": previous_signal,
        "signal_bar": featured.index[-1].isoformat(),
        "featured_tail": feature_tail_payload(featured),
    }
    if signal == 0 or signal == previous_signal:
        return {"action": "HOLD", **base}
    return {
        "action": "ENTER_LONG" if signal > 0 else "ENTER_SHORT",
        "stop_loss": float(current["stop_loss"]),
        "take_profit": float(current["take_profit"]) if pd.notna(current["take_profit"]) else None,
        "entry_atr": float(current["atr_14"]),
        "close": float(current["close"]),
        **base,
    }


def fee_break_even(position: RuntimePosition, fee_rate: float, slippage_rate: float) -> float:
    return position.entry_price * (1 + position.signal * (2 * fee_rate + 2 * slippage_rate))


def manage_position(
    position: RuntimePosition,
    featured: pd.DataFrame,
    broker: DemoBroker | LiveBinanceBroker,
    runtime: CandidateRuntime,
    rules: Any,
    mark_price: float,
) -> RuntimePosition | None:
    row = featured.iloc[-1]
    bar_time = featured.index[-1].isoformat()
    if position.last_managed_bar == bar_time:
        return position

    position.last_managed_bar = bar_time
    position.bars_held += 1
    signal = int(row["signal"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    min_hold = int(runtime.params["min_holding_bars"])
    max_hold = int(runtime.params["max_holding_bars"])
    old_stop = position.stop_loss

    def close_out(price: float, reason: str) -> None:
        broker.close_position(position, price, reason)

    if position.signal > 0:
        if low <= position.stop_loss and position.bars_held >= min_hold:
            close_out(position.stop_loss, "stop_loss")
            return None
        position.best_price = max(position.best_price, high)
        if runtime.params.get("use_break_even") and position.best_price >= position.entry_price + float(runtime.params["break_even_atr"]) * position.entry_atr:
            position.stop_loss = max(position.stop_loss, fee_break_even(position, runtime.fee_rate, runtime.slippage_rate))
        if runtime.params.get("use_trailing_stop"):
            position.stop_loss = max(position.stop_loss, position.best_price - float(runtime.params["trail_atr_mult"]) * position.entry_atr)
        if position.take_profit is not None and high >= position.take_profit and position.bars_held >= min_hold:
            close_out(position.take_profit, "take_profit")
            return None
    else:
        if high >= position.stop_loss and position.bars_held >= min_hold:
            close_out(position.stop_loss, "stop_loss")
            return None
        position.best_price = min(position.best_price, low)
        if runtime.params.get("use_break_even") and position.best_price <= position.entry_price - float(runtime.params["break_even_atr"]) * position.entry_atr:
            position.stop_loss = min(position.stop_loss, fee_break_even(position, runtime.fee_rate, runtime.slippage_rate))
        if runtime.params.get("use_trailing_stop"):
            position.stop_loss = min(position.stop_loss, position.best_price + float(runtime.params["trail_atr_mult"]) * position.entry_atr)
        if position.take_profit is not None and low <= position.take_profit and position.bars_held >= min_hold:
            close_out(position.take_profit, "take_profit")
            return None

    if position.bars_held >= min_hold and bool(runtime.params["exit_on_flat_signal"]) and signal == 0:
        close_out(close, "flat_signal")
        return None
    if position.bars_held >= min_hold and signal == -position.signal:
        close_out(close, "opposite_signal")
        return None
    if position.bars_held >= max_hold:
        close_out(close, "max_hold")
        return None
    if abs(position.stop_loss - old_stop) > 1e-12:
        broker.update_stop(position, position.stop_loss, rules)
    return position


def run_once(args: argparse.Namespace, store: SQLiteStateStore) -> None:
    exchange = "demo" if args.exchange == "demo" else "mainnet"
    client = BinanceFuturesClient.from_exchange(exchange)
    runtime = CandidateRuntime.from_summary(
        args.candidate_summary,
        section=args.candidate_section,
        balance=args.balance,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
    )
    if args.mode == "live" and args.exchange != "mainnet":
        logger.warning("Live mode is using Binance demo endpoint, not a production account")
    rules = client.symbol_rules(runtime.symbol)
    raw = client.closed_klines(runtime.symbol, runtime.timeframe, limit=args.kline_limit)
    featured = runtime.feature_frame(raw)
    if featured.empty:
        logger.warning("No featured bars available")
        return
    mark_price = client.mark_price(runtime.symbol)
    position = load_active_position(store)

    if args.mode == "live":
        broker: DemoBroker | LiveBinanceBroker = LiveBinanceBroker(
            client=client,
            store=store,
            risk_per_trade=args.risk_per_trade,
            max_position_pct=args.max_position_pct,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            max_notional_usdt=args.max_notional_usdt,
        )
    else:
        demo_balance = float(store.get_json("demo_balance", args.balance))
        broker = DemoBroker(
            store=store,
            balance=demo_balance,
            risk_per_trade=args.risk_per_trade,
            max_position_pct=args.max_position_pct,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )

    if position is not None:
        position = manage_position(position, featured, broker, runtime, rules, mark_price)
        save_active_position(store, position)

    decision = latest_signal_from_featured(featured)
    if position is None and decision and decision["action"].startswith("ENTER"):
        inserted = store.signal(decision["signal_bar"], runtime.symbol, int(decision["signal"]), decision)
        if inserted:
            position = broker.open_position(decision, mark_price, rules)
            save_active_position(store, position)
        else:
            logger.info("Signal bar already processed: %s", decision["signal_bar"])
    elif decision:
        store.event("signal_hold", {"signal_bar": decision["signal_bar"], "signal": decision["signal"], "action": decision["action"]})

    if isinstance(broker, DemoBroker):
        unrealized = 0.0
        if position:
            unrealized = (mark_price - position.entry_price) * position.quantity * position.signal
        store.set_json("demo_balance", broker.balance)
        store.equity(
            featured.index[-1].isoformat(),
            broker.balance + unrealized,
            broker.balance,
            unrealized,
            {"mark_price": mark_price, "position": asdict(position) if position else None},
        )
    else:
        store.equity(
            featured.index[-1].isoformat(),
            broker.balance,
            broker.balance,
            0.0,
            {"mark_price": mark_price, "position": asdict(position) if position else None},
        )
    store.set_json("last_run", {"ts": pd.Timestamp.now("UTC").isoformat(), "mode": args.mode, "exchange": args.exchange})
    logger.info("Cycle complete mode=%s exchange=%s active_position=%s metrics=%s", args.mode, args.exchange, bool(position), store.closed_trade_summary())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    store = SQLiteStateStore(args.state_db)
    while True:
        run_once(args, store)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
