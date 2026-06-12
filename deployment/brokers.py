"""Demo and live brokers for the selected SOL momentum strategy."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Protocol

from deployment.binance_futures import BinanceFuturesClient, SymbolRules
from deployment.state import SQLiteStateStore

logger = logging.getLogger(__name__)

LIVE_ACK_VALUE = "I_UNDERSTAND_THIS_PLACES_REAL_BINANCE_FUTURES_ORDERS"


@dataclass
class RuntimePosition:
    trade_id: str
    symbol: str
    side: str
    signal: int
    quantity: float
    entry_price: float
    entry_time: str
    signal_bar: str
    stop_loss: float
    take_profit: float | None
    entry_atr: float
    risk_amount: float
    notional: float
    fees: float
    slippage: float
    bars_held: int = 0
    best_price: float = 0.0
    realized_pnl: float = 0.0
    protective_order_id: str | None = None
    last_managed_bar: str | None = None


class Broker(Protocol):
    balance: float

    def open_position(self, decision: dict[str, Any], price: float, rules: SymbolRules) -> RuntimePosition | None:
        ...

    def close_position(self, position: RuntimePosition, price: float, reason: str) -> RuntimePosition:
        ...

    def update_stop(self, position: RuntimePosition, new_stop: float, rules: SymbolRules) -> None:
        ...


def position_size(
    balance: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade: float,
    max_position_pct: float,
    fee_rate: float,
    slippage_rate: float,
    signal: int,
) -> tuple[float, float, float]:
    stop_fill = stop_loss * (1 - slippage_rate) if signal > 0 else stop_loss * (1 + slippage_rate)
    unit_risk = abs(entry_price - stop_fill) + 2 * entry_price * fee_rate
    if unit_risk <= 0:
        return 0.0, 0.0, 0.0
    risk_amount = balance * risk_per_trade
    quantity = risk_amount / unit_risk
    notional = quantity * entry_price
    max_notional = balance * max_position_pct
    if notional > max_notional:
        quantity = max_notional / entry_price
        notional = max_notional
        risk_amount = quantity * unit_risk
    return quantity, notional, risk_amount


class DemoBroker:
    """Paper broker that tracks signals, PnL, fees, slippage, and R metrics."""

    def __init__(
        self,
        store: SQLiteStateStore,
        balance: float,
        risk_per_trade: float,
        max_position_pct: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> None:
        self.store = store
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def open_position(self, decision: dict[str, Any], price: float, rules: SymbolRules) -> RuntimePosition | None:
        signal = int(decision["signal"])
        entry = price * (1 + self.slippage_rate * signal)
        quantity, notional, risk_amount = position_size(
            self.balance,
            entry,
            float(decision["stop_loss"]),
            self.risk_per_trade,
            self.max_position_pct,
            self.fee_rate,
            self.slippage_rate,
            signal,
        )
        rounded = rules.round_quantity(quantity)
        if rounded < rules.min_quantity or rounded * Decimal(str(entry)) < rules.min_notional:
            logger.warning("Skipping demo trade below exchange minimum: qty=%s notional=%s", rounded, rounded * Decimal(str(entry)))
            return None
        quantity = float(rounded)
        notional = quantity * entry
        fee = notional * self.fee_rate
        self.balance -= fee
        trade_id = f"DEMO-{int(time.time() * 1000)}"
        position = RuntimePosition(
            trade_id=trade_id,
            symbol=rules.symbol,
            side="LONG" if signal > 0 else "SHORT",
            signal=signal,
            quantity=quantity,
            entry_price=entry,
            entry_time=str(pd_now()),
            signal_bar=str(decision["signal_bar"]),
            stop_loss=float(decision["stop_loss"]),
            take_profit=float(decision["take_profit"]) if decision.get("take_profit") else None,
            entry_atr=float(decision["entry_atr"]),
            risk_amount=risk_amount,
            notional=notional,
            fees=fee,
            slippage=abs(entry - price) * quantity,
            best_price=entry,
            realized_pnl=-fee,
        )
        self._persist_open(position)
        return position

    def close_position(self, position: RuntimePosition, price: float, reason: str) -> RuntimePosition:
        exit_price = price * (1 - self.slippage_rate * position.signal)
        exit_fee = exit_price * position.quantity * self.fee_rate
        gross = (exit_price - position.entry_price) * position.quantity * position.signal
        pnl = position.realized_pnl + gross - exit_fee
        position.fees += exit_fee
        position.slippage += abs(exit_price - price) * position.quantity
        self.balance += gross - exit_fee
        payload = asdict(position)
        payload.update({
            "closed_at": str(pd_now()),
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl / position.notional if position.notional else 0.0,
            "r_multiple": pnl / position.risk_amount if position.risk_amount else 0.0,
            "exit_reason": reason,
            "status": "CLOSED",
            "opened_at": position.entry_time,
        })
        self.store.upsert_trade(position.trade_id, payload)
        self.store.event("demo_exit", payload)
        return position

    def update_stop(self, position: RuntimePosition, new_stop: float, rules: SymbolRules) -> None:
        position.stop_loss = new_stop

    def _persist_open(self, position: RuntimePosition) -> None:
        payload = asdict(position)
        payload.update({"status": "OPEN", "opened_at": position.entry_time})
        self.store.upsert_trade(position.trade_id, payload)
        self.store.event("demo_entry", payload)


class LiveBinanceBroker:
    """Real Binance USD-M Futures broker with explicit live-mode safeguards."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        store: SQLiteStateStore,
        risk_per_trade: float,
        max_position_pct: float,
        fee_rate: float,
        slippage_rate: float,
        max_notional_usdt: float,
    ) -> None:
        if os.getenv("CRYPTOEA_LIVE_TRADING_ACK") != LIVE_ACK_VALUE:
            raise RuntimeError(f"Refusing live trading without CRYPTOEA_LIVE_TRADING_ACK={LIVE_ACK_VALUE!r}")
        self.client = client
        self.store = store
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.max_notional_usdt = max_notional_usdt
        self.balance = client.usdt_available_balance()

    def open_position(self, decision: dict[str, Any], price: float, rules: SymbolRules) -> RuntimePosition | None:
        self.balance = self.client.usdt_available_balance()
        signal = int(decision["signal"])
        entry_estimate = price * (1 + self.slippage_rate * signal)
        quantity, notional, risk_amount = position_size(
            self.balance,
            entry_estimate,
            float(decision["stop_loss"]),
            self.risk_per_trade,
            self.max_position_pct,
            self.fee_rate,
            self.slippage_rate,
            signal,
        )
        if notional > self.max_notional_usdt:
            quantity = self.max_notional_usdt / entry_estimate
            notional = self.max_notional_usdt
        rounded = rules.round_quantity(quantity)
        if rounded < rules.min_quantity or rounded * Decimal(str(entry_estimate)) < rules.min_notional:
            logger.warning("Skipping live order below exchange minimum: qty=%s notional=%s", rounded, rounded * Decimal(str(entry_estimate)))
            return None
        side = "BUY" if signal > 0 else "SELL"
        trade_id = f"LIVE-{int(time.time() * 1000)}"
        response = self.client.new_order(
            symbol=rules.symbol,
            side=side,
            order_type="MARKET",
            quantity=rounded,
            reduce_only=False,
            client_order_id=trade_id,
        )
        fill_price = float(response.get("avgPrice") or response.get("price") or price)
        if fill_price <= 0:
            fill_price = price
        position = RuntimePosition(
            trade_id=trade_id,
            symbol=rules.symbol,
            side="LONG" if signal > 0 else "SHORT",
            signal=signal,
            quantity=float(rounded),
            entry_price=fill_price,
            entry_time=str(pd_now()),
            signal_bar=str(decision["signal_bar"]),
            stop_loss=float(decision["stop_loss"]),
            take_profit=float(decision["take_profit"]) if decision.get("take_profit") else None,
            entry_atr=float(decision["entry_atr"]),
            risk_amount=risk_amount,
            notional=float(rounded) * fill_price,
            fees=float(rounded) * fill_price * self.fee_rate,
            slippage=abs(fill_price - price) * float(rounded),
            best_price=fill_price,
            realized_pnl=-(float(rounded) * fill_price * self.fee_rate),
        )
        self.update_stop(position, position.stop_loss, rules)
        payload = asdict(position)
        payload.update({"status": "OPEN", "opened_at": position.entry_time, "entry_order": response})
        self.store.upsert_trade(position.trade_id, payload)
        self.store.event("live_entry", payload)
        return position

    def close_position(self, position: RuntimePosition, price: float, reason: str) -> RuntimePosition:
        side = "SELL" if position.signal > 0 else "BUY"
        try:
            self.client.cancel_all_orders(position.symbol)
        except Exception as exc:
            logger.warning("Could not cancel protective orders before close: %s", exc)
        response = self.client.new_order(
            symbol=position.symbol,
            side=side,
            order_type="MARKET",
            quantity=Decimal(str(position.quantity)),
            reduce_only=True,
            client_order_id=f"{position.trade_id}-CLOSE",
        )
        fill_price = float(response.get("avgPrice") or response.get("price") or price)
        gross = (fill_price - position.entry_price) * position.quantity * position.signal
        exit_fee = fill_price * position.quantity * self.fee_rate
        pnl = position.realized_pnl + gross - exit_fee
        payload = asdict(position)
        payload.update({
            "closed_at": str(pd_now()),
            "exit_price": fill_price,
            "pnl": pnl,
            "pnl_pct": pnl / position.notional if position.notional else 0.0,
            "r_multiple": pnl / position.risk_amount if position.risk_amount else 0.0,
            "exit_reason": reason,
            "exit_order": response,
            "status": "CLOSED",
            "opened_at": position.entry_time,
        })
        self.store.upsert_trade(position.trade_id, payload)
        self.store.event("live_exit", payload)
        return position

    def update_stop(self, position: RuntimePosition, new_stop: float, rules: SymbolRules) -> None:
        side = "SELL" if position.signal > 0 else "BUY"
        rounded_stop = rules.round_price(new_stop) if side == "SELL" else rules.round_price_up(new_stop)
        if rounded_stop <= 0:
            return
        try:
            self.client.cancel_all_orders(position.symbol)
        except Exception as exc:
            logger.warning("Could not cancel previous protective stop: %s", exc)
        response = self.client.new_order(
            symbol=position.symbol,
            side=side,
            order_type="STOP_MARKET",
            quantity=Decimal(str(position.quantity)),
            reduce_only=True,
            stop_price=rounded_stop,
            client_order_id=f"{position.trade_id}-STOP",
            response_type="ACK",
        )
        position.stop_loss = float(rounded_stop)
        position.protective_order_id = str(response.get("orderId", ""))
        self.store.event("protective_stop_update", {"trade_id": position.trade_id, "stop": str(rounded_stop), "response": response})


def pd_now() -> str:
    import pandas as pd

    return pd.Timestamp.now("UTC").isoformat()
