"""Live trading execution module with order management and reconciliation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from live.binance_client import BinanceClient
from live.state_manager import StateManager

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None
    stop_loss: float | None = None
    take_profit: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_time: datetime | None = None
    error_message: str | None = None


@dataclass
class ExecutionConfig:
    """Configuration for live trading execution."""

    symbol: str = "BTCUSDT"
    max_open_orders: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    maker_fee: float = 0.0004
    taker_fee: float = 0.0006
    slippage_tolerance: float = 0.0005
    state_dir: str = "data/state"
    heartbeat_interval: int = 60


class ExecutionEngine:
    """Live trading execution engine with order management and reconciliation.

    Handles order placement, monitoring, cancellation, and reconciliation
    with the exchange. Maintains order state and provides execution
    analytics.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        client: BinanceClient | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.client = client or BinanceClient()
        self.state_manager = StateManager(
            state_dir=self.config.state_dir
        )
        self.orders: dict[str, Order] = {}
        self.order_history: list[dict[str, Any]] = []

    def submit_order(
        self,
        symbol: str | None = None,
        side: OrderSide = OrderSide.BUY,
        order_type: OrderType = OrderType.MARKET,
        quantity: float = 0.0,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order | None:
        """Submit a trading order to the exchange.

        Args:
            symbol: Trading pair (overrides config if provided).
            side: Buy or sell side.
            order_type: Market, limit, stop loss, or take profit.
            quantity: Order quantity.
            price: Limit price (required for LIMIT orders).
            stop_loss: Stop loss price.
            take_profit: Take profit price.

        Returns:
            Order object with updated status, or None if submission fails.
        """
        symbol = symbol or self.config.symbol

        if quantity <= 0:
            logger.error("Invalid quantity: %s", quantity)
            return None

        if order_type == OrderType.LIMIT and price is None:
            logger.error("Limit order requires a price")
            return None

        order_id = f"{symbol}_{side.value}_{int(time.time() * 1000)}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        try:
            # Submit to exchange
            if order_type == OrderType.MARKET:
                result = self.client.place_market_order(
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                )
            elif order_type == OrderType.LIMIT:
                result = self.client.place_limit_order(
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    price=price,
                )
            else:
                logger.warning("Unsupported order type: %s", order_type)
                order.status = OrderStatus.REJECTED
                order.error_message = f"Unsupported order type: {order_type}"
                self._record_order(order)
                return order

            if result and result.get("status") == "FILLED":
                order.status = OrderStatus.FILLED
                order.filled_price = float(result.get("price", price or 0))
                order.filled_quantity = float(
                    result.get("executedQty", quantity)
                )
                order.commission = float(result.get("commission", 0))
                order.update_time = datetime.now(timezone.utc)
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get("msg", "Unknown error")

            self._record_order(order)
            logger.info(
                "Order %s submitted: %s %s %.4f @ %s",
                order_id,
                side.value,
                symbol,
                quantity,
                price or "MARKET",
            )

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error("Order submission failed: %s", e)
            self._record_order(order)

        return order

    def cancel_order(
        self,
        order_id: str,
    ) -> bool:
        """Cancel a pending order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        if order_id not in self.orders:
            logger.warning("Order %s not found", order_id)
            return False

        order = self.orders[order_id]

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            logger.info("Order %s already %s", order_id, order.status.value)
            return True

        try:
            result = self.client.cancel_order(
                symbol=order.symbol,
                order_id=order.order_id,
            )

            if result:
                order.status = OrderStatus.CANCELLED
                order.update_time = datetime.now(timezone.utc)
                logger.info("Order %s cancelled", order_id)
                return True
            else:
                order.error_message = "Cancellation returned no result"
                return False

        except Exception as e:
            order.error_message = str(e)
            logger.error("Order cancellation failed: %s", e)
            return False

    def cancel_all_orders(
        self,
        symbol: str | None = None,
    ) -> int:
        """Cancel all open orders for a symbol (or all symbols).

        Args:
            symbol: Symbol to cancel orders for (None for all).

        Returns:
            Number of orders cancelled.
        """
        symbol = symbol or self.config.symbol
        cancelled = 0

        for order_id, order in list(self.orders.items()):
            if order.symbol == symbol and order.status == OrderStatus.PENDING:
                if self.cancel_order(order_id):
                    cancelled += 1

        logger.info("Cancelled %d orders for %s", cancelled, symbol)
        return cancelled

    def reconcile_orders(
        self,
    ) -> dict[str, Any]:
        """Reconcile local order state with exchange state.

        Updates local order statuses based on exchange state.

        Returns:
            Dictionary containing reconciliation results.
        """
        logger.info("Starting order reconciliation")

        try:
            open_orders = self.client.get_open_orders(
                symbol=self.config.symbol
            )

            updated_count = 0

            for exchange_order in open_orders:
                exchange_id = str(exchange_order.get("orderId"))

                # Find matching local order
                for order_id, order in self.orders.items():
                    if order.status == OrderStatus.PENDING:
                        order.status = OrderStatus.PARTIALLY_FILLED
                        order.filled_quantity = float(
                            exchange_order.get("executedQty", 0)
                        )
                        order.filled_price = float(
                            exchange_order.get("price", 0)
                        )
                        order.update_time = datetime.now(timezone.utc)
                        updated_count += 1

            self.state_manager.save_state({
                "last_reconciliation": datetime.now(timezone.utc).isoformat(),
                "updated_orders": updated_count,
            })

            logger.info(
                "Reconciliation complete: %d orders updated",
                updated_count,
            )

            return {
                "updated_count": updated_count,
                "open_orders_count": len(open_orders),
            }

        except Exception as e:
            logger.error("Reconciliation failed: %s", e)
            return {"error": str(e)}

    def get_execution_report(
        self,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Generate execution analytics report.

        Args:
            symbol: Symbol to report on (None for all).

        Returns:
            Dictionary containing execution metrics.
        """
        symbol = symbol or self.config.symbol

        symbol_orders = [
            o for o in self.orders.values()
            if o.symbol == symbol
        ]

        if not symbol_orders:
            return {"message": "No orders found"}

        total_orders = len(symbol_orders)
        filled = sum(
            1
            for o in symbol_orders
            if o.status == OrderStatus.FILLED
        )
        cancelled = sum(
            1
            for o in symbol_orders
            if o.status == OrderStatus.CANCELLED
        )
        rejected = sum(
            1
            for o in symbol_orders
            if o.status == OrderStatus.REJECTED
        )

        total_commission = sum(o.commission for o in symbol_orders)

        return {
            "symbol": symbol,
            "total_orders": total_orders,
            "filled": filled,
            "cancelled": cancelled,
            "rejected": rejected,
            "fill_rate": filled / total_orders if total_orders > 0 else 0,
            "total_commission": total_commission,
            "orders": [
                {
                    "order_id": o.order_id,
                    "side": o.side.value,
                    "type": o.order_type.value,
                    "quantity": o.quantity,
                    "status": o.status.value,
                    "commission": o.commission,
                }
                for o in symbol_orders
            ],
        }

    def _record_order(
        self,
        order: Order,
    ) -> None:
        """Record order in local state and history.

        Args:
            order: Order object to record.
        """
        self.orders[order.order_id] = order

        self.order_history.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "filled_price": order.filled_price,
            "commission": order.commission,
            "timestamp": order.timestamp.isoformat(),
            "update_time": (
                order.update_time.isoformat()
                if order.update_time
                else None
            ),
            "error_message": order.error_message,
        })

        # Persist state
        self.state_manager.save_state({
            "last_order": order.order_id,
            "total_orders": len(self.orders),
        })
