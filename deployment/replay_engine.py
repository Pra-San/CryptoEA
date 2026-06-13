"""Historical proxy feed for deployment/backtest parity checks."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from deployment.brokers import position_size
from deployment.strategy_runtime import CandidateRuntime
from scripts.explore_high_winrate_edge import trades_per_week
from scripts.explore_trailing_highwin_edge import metrics_from_trades


@dataclass
class ReplayConfig:
    """Configuration for replaying deployment execution on historical bars."""

    exact_quantity: bool = True
    quantity_step: float = 0.0
    min_quantity: float = 0.0
    min_notional: float = 0.0


@dataclass
class ReplayPosition:
    """Internal active position for bar-by-bar replay."""

    signal: int
    entry_idx: int
    signal_idx: int
    entry_time: Any
    side: str
    entry_price: float
    quantity: float
    notional: float
    stop_loss: float
    take_profit: float | None
    entry_atr: float
    risk_amount: float
    realized_pnl: float
    fees: float
    slippage: float
    best_price: float
    partial_taken: bool = False
    remaining_quantity: float = 0.0


def _round_quantity(quantity: float, config: ReplayConfig) -> float:
    if config.exact_quantity or config.quantity_step <= 0:
        return quantity
    return np.floor(quantity / config.quantity_step) * config.quantity_step


def _new_trade(
    position: ReplayPosition,
    exit_idx: int,
    raw_exit: float,
    exit_reason: str,
    featured: pd.DataFrame,
    runtime: CandidateRuntime,
    balance: float,
) -> tuple[SimpleNamespace, float]:
    slip = runtime.slippage_rate
    args = runtime.args(balance=balance)
    exit_price = raw_exit * (1 - slip * position.signal)
    gross = (exit_price - position.entry_price) * position.remaining_quantity * position.signal
    exit_fee = position.remaining_quantity * exit_price * runtime.fee_rate
    pnl = position.realized_pnl + gross - exit_fee
    total_fees = position.fees + exit_fee
    slippage = position.slippage + abs(exit_price - raw_exit) * position.remaining_quantity
    reason = f"partial_then_{exit_reason}" if position.partial_taken else exit_reason
    trade = SimpleNamespace(
        entry_time=featured.index[position.entry_idx],
        exit_time=featured.index[exit_idx],
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        notional=position.notional,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        exit_reason=reason,
        pnl=pnl,
        pnl_pct=pnl / position.notional if position.notional else 0.0,
        risk_amount=position.risk_amount,
        r_multiple=pnl / position.risk_amount if position.risk_amount else 0.0,
        fees=total_fees,
        slippage=slippage,
        holding_bars=exit_idx - position.entry_idx,
        metadata={"symbol": runtime.symbol, "replay_balance": args.balance},
    )
    return trade, balance + pnl


def replay_deployment(featured: pd.DataFrame, runtime: CandidateRuntime, config: ReplayConfig | None = None) -> list[SimpleNamespace]:
    """Replay deployment execution over historical bars.

    The loop mirrors the intended live lifecycle:
    signal on closed bar i -> enter on bar i+1 open -> manage on completed bars.
    In exact-quantity mode it should match the research simulator trade-for-trade.
    """

    cfg = config or ReplayConfig()
    if featured.empty or len(featured) < 3:
        return []
    params = runtime.params
    args = runtime.args()
    signals = featured["signal"].astype(int).to_numpy()
    opens = featured["open"].to_numpy(dtype=float)
    highs = featured["high"].to_numpy(dtype=float)
    lows = featured["low"].to_numpy(dtype=float)
    closes = featured["close"].to_numpy(dtype=float)
    stops = featured["stop_loss"].to_numpy(dtype=float)
    tps = featured["take_profit"].to_numpy(dtype=float)
    atr = featured["atr_14"].replace(0, np.nan).to_numpy(dtype=float)
    min_hold = int(params["min_holding_bars"])
    max_hold = int(params["max_holding_bars"])
    balance = args.balance
    pending: dict[str, Any] | None = None
    position: ReplayPosition | None = None
    trades: list[SimpleNamespace] = []

    for i in range(1, len(featured)):
        if pending and position is None and i == pending["entry_idx"]:
            sig = int(pending["signal"])
            raw_entry = opens[i]
            entry = raw_entry * (1 + runtime.slippage_rate * sig)
            stop = float(pending["stop_loss"])
            entry_atr = float(pending["entry_atr"])
            quantity, notional, risk_amount = position_size(
                balance,
                entry,
                stop,
                runtime.risk_per_trade,
                runtime.max_position_pct,
                runtime.fee_rate,
                runtime.slippage_rate,
                sig,
            )
            quantity = _round_quantity(quantity, cfg)
            notional = quantity * entry
            if quantity > 0 and notional >= cfg.min_notional and quantity >= cfg.min_quantity:
                entry_fee = quantity * entry * runtime.fee_rate
                position = ReplayPosition(
                    signal=sig,
                    entry_idx=i,
                    signal_idx=int(pending["signal_idx"]),
                    entry_time=featured.index[i],
                    side="LONG" if sig == 1 else "SHORT",
                    entry_price=entry,
                    quantity=quantity,
                    notional=notional,
                    stop_loss=stop,
                    take_profit=pending["take_profit"],
                    entry_atr=entry_atr,
                    risk_amount=risk_amount,
                    realized_pnl=-entry_fee,
                    fees=entry_fee,
                    slippage=abs(entry - raw_entry) * quantity,
                    best_price=entry,
                    remaining_quantity=quantity,
                )
            pending = None

        if position is not None and i >= position.entry_idx + 1:
            exit_idx: int | None = None
            exit_raw: float | None = None
            exit_reason = "max_hold"
            fee_be = position.entry_price * (1 + position.signal * (2 * runtime.fee_rate + 2 * runtime.slippage_rate))
            partial_target = position.entry_price + position.signal * float(params["partial_exit_atr"]) * position.entry_atr
            partial_fraction = min(max(float(params["partial_exit_fraction"]), 0.0), 0.95)
            holding = i - position.entry_idx
            if position.signal == 1:
                if lows[i] <= position.stop_loss and holding >= min_hold:
                    exit_idx, exit_raw, exit_reason = i, position.stop_loss, "stop_loss"
                else:
                    position.best_price = max(position.best_price, highs[i])
                    if params["use_break_even"] and position.best_price >= position.entry_price + float(params["break_even_atr"]) * position.entry_atr:
                        position.stop_loss = max(position.stop_loss, fee_be)
                    if params["use_trailing_stop"]:
                        position.stop_loss = max(position.stop_loss, position.best_price - float(params["trail_atr_mult"]) * position.entry_atr)
                    if (
                        params["use_partial_exit"]
                        and not position.partial_taken
                        and highs[i] >= partial_target
                        and holding >= min_hold
                    ):
                        part_qty = position.quantity * partial_fraction
                        part_exit = partial_target * (1 - runtime.slippage_rate)
                        position.realized_pnl += (part_exit - position.entry_price) * part_qty - part_qty * part_exit * runtime.fee_rate
                        position.fees += part_qty * part_exit * runtime.fee_rate
                        position.slippage += abs(part_exit - partial_target) * part_qty
                        position.remaining_quantity -= part_qty
                        position.partial_taken = True
                        if params["move_stop_after_partial"]:
                            position.stop_loss = max(position.stop_loss, fee_be)
                    if position.take_profit is not None and highs[i] >= position.take_profit and holding >= min_hold:
                        exit_idx, exit_raw, exit_reason = i, position.take_profit, "take_profit"
            else:
                if highs[i] >= position.stop_loss and holding >= min_hold:
                    exit_idx, exit_raw, exit_reason = i, position.stop_loss, "stop_loss"
                else:
                    position.best_price = min(position.best_price, lows[i])
                    if params["use_break_even"] and position.best_price <= position.entry_price - float(params["break_even_atr"]) * position.entry_atr:
                        position.stop_loss = min(position.stop_loss, fee_be)
                    if params["use_trailing_stop"]:
                        position.stop_loss = min(position.stop_loss, position.best_price + float(params["trail_atr_mult"]) * position.entry_atr)
                    if (
                        params["use_partial_exit"]
                        and not position.partial_taken
                        and lows[i] <= partial_target
                        and holding >= min_hold
                    ):
                        part_qty = position.quantity * partial_fraction
                        part_exit = partial_target * (1 + runtime.slippage_rate)
                        position.realized_pnl += (position.entry_price - part_exit) * part_qty - part_qty * part_exit * runtime.fee_rate
                        position.fees += part_qty * part_exit * runtime.fee_rate
                        position.slippage += abs(part_exit - partial_target) * part_qty
                        position.remaining_quantity -= part_qty
                        position.partial_taken = True
                        if params["move_stop_after_partial"]:
                            position.stop_loss = min(position.stop_loss, fee_be)
                    if position.take_profit is not None and lows[i] <= position.take_profit and holding >= min_hold:
                        exit_idx, exit_raw, exit_reason = i, position.take_profit, "take_profit"
            if exit_idx is None and holding >= min_hold:
                if bool(params["exit_on_flat_signal"]) and signals[i] == 0:
                    exit_idx, exit_raw, exit_reason = i, closes[i], "flat_signal"
                elif signals[i] == -position.signal:
                    exit_idx, exit_raw, exit_reason = i, closes[i], "opposite_signal"
                elif holding >= max_hold:
                    exit_idx, exit_raw, exit_reason = i, closes[i], "max_hold"
            if exit_idx is not None and exit_raw is not None:
                trade, balance = _new_trade(position, exit_idx, float(exit_raw), exit_reason, featured, runtime, balance)
                trades.append(trade)
                position = None

        current_signal = int(signals[i])
        previous_signal = int(signals[i - 1])
        if (
            position is None
            and pending is None
            and current_signal != 0
            and current_signal != previous_signal
            and i + 1 < len(featured)
            and not np.isnan(stops[i])
        ):
            pending = {
                "signal": current_signal,
                "signal_idx": i,
                "entry_idx": i + 1,
                "stop_loss": float(stops[i]),
                "take_profit": None if np.isnan(tps[i]) else float(tps[i]),
                "entry_atr": float(atr[i]) if not np.isnan(atr[i]) else abs(float(closes[i]) - float(stops[i])),
            }

    if position is not None:
        final_idx = len(featured) - 1
        if final_idx >= position.entry_idx + min_hold:
            trade, balance = _new_trade(position, final_idx, float(closes[final_idx]), "max_hold", featured, runtime, balance)
            trades.append(trade)

    return trades


def replay_metrics(featured: pd.DataFrame, runtime: CandidateRuntime, config: ReplayConfig | None = None) -> tuple[list[SimpleNamespace], dict[str, Any]]:
    trades = replay_deployment(featured, runtime, config=config)
    metrics = metrics_from_trades(trades, featured.index, runtime.balance)
    metrics["trades_per_week"] = trades_per_week(metrics)
    return trades, metrics
