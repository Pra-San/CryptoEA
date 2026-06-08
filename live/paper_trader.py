"""Paper trading module for CryptoEA.

Simulates live trading without executing real orders.
Validates signal generation matches backtest before going live.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Record of a paper trade."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""  # 'BUY' or 'SELL'
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""

    initial_balance: float = 10000.0
    fee_rate: float = 0.0006  # 0.06% taker fee
    slippage_rate: float = 0.0003  # 0.03% slippage
    max_positions: int = 3
    min_trade_size: float = 10.0


class PaperTrader:
    """Simulates live trading for validation before going live.

    Usage:
        trader = PaperTrader()
        trader.connect()  # Connect to WebSocket for real-time data
        trade = trader.execute_entry("BTCUSDT", "BUY", price, quantity)
        trader.execute_exit(trade.trade_id, price)
        stats = trader.get_performance()
    """

    def __init__(self, config: Optional[PaperTradingConfig] = None) -> None:
        """Initialize the paper trader.

        Args:
            config: Paper trading configuration.
        """
        self.config = config or PaperTradingConfig()
        self.balance = self.config.initial_balance
        self.open_trades: List[PaperTrade] = []
        self.completed_trades: List[PaperTrade] = []
        self.equity_curve: List[Dict] = []
        self._trade_counter = 0
        self._start_time = datetime.utcnow()

    def execute_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        metadata: Optional[Dict] = None,
    ) -> Optional[PaperTrade]:
        """Execute a paper trade entry.

        Args:
            symbol: Trading symbol.
            side: 'BUY' or 'SELL'.
            price: Entry price.
            quantity: Trade quantity.
            metadata: Additional trade metadata.

        Returns:
            PaperTrade if entry successful, None otherwise.
        """
        if len(self.open_trades) >= self.config.max_positions:
            logger.warning(
                f"Max positions ({self.config.max_positions}) reached. "
                f"Cannot open new trade for {symbol}."
            )
            return None

        # Check minimum trade size
        notional = price * quantity
        if notional < self.config.min_trade_size:
            logger.warning(
                f"Trade size {notional:,.2f} below minimum "
                f"{self.config.min_trade_size:,.2f}. Skipping."
            )
            return None

        # Apply slippage
        if side == "BUY":
            entry_price = price * (1 + self.config.slippage_rate)
        else:
            entry_price = price * (1 - self.config.slippage_rate)

        # Calculate fees
        fees = notional * self.config.fee_rate

        # Check balance
        if notional + fees > self.balance:
            logger.warning(
                f"Insufficient balance for {symbol}. "
                f"Need {notional + fees:,.2f}, have {self.balance:,.2f}"
            )
            return None

        self._trade_counter += 1
        trade = PaperTrade(
            trade_id=f"PT-{self._trade_counter:04d}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.utcnow(),
            fees=fees,
            metadata=metadata or {},
        )

        self.open_trades.append(trade)
        self.balance -= (notional + fees)

        logger.info(
            f"Paper trade ENTRY: {trade.trade_id} {side} {quantity} "
            f"{symbol} @ {entry_price:,.2f} (PnL so far: {self._get_unrealized_pnl(trade):,.2f})"
        )

        return trade

    def execute_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "signal_reversal",
    ) -> Optional[PaperTrade]:
        """Execute a paper trade exit.

        Args:
            trade_id: Trade ID to close.
            exit_price: Exit price.
            exit_reason: Reason for exit.

        Returns:
            Completed PaperTrade or None.
        """
        trade = next(
            (t for t in self.open_trades if t.trade_id == trade_id), None
        )

        if trade is None:
            logger.warning(f"Trade {trade_id} not found for exit.")
            return None

        # Apply slippage on exit
        if trade.side == "BUY":
            actual_exit = exit_price * (1 - self.config.slippage_rate)
        else:
            actual_exit = exit_price * (1 + self.config.slippage_rate)

        # Calculate PnL
        exit_notional = actual_exit * trade.quantity
        exit_fees = exit_notional * self.config.fee_rate

        if trade.side == "BUY":
            gross_pnl = (actual_exit - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - actual_exit) * trade.quantity

        net_pnl = gross_pnl - exit_fees - trade.fees
        pnl_pct = net_pnl / (trade.entry_price * trade.quantity)

        trade.exit_price = actual_exit
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = exit_reason
        trade.pnl = net_pnl
        trade.pnl_pct = pnl_pct
        trade.slippage = abs(actual_exit - exit_price) * trade.quantity

        # Return notional to balance
        self.balance += gross_pnl - exit_fees

        # Move from open to completed
        self.open_trades.remove(trade)
        self.completed_trades.append(trade)

        logger.info(
            f"Paper trade EXIT: {trade.trade_id} {trade.symbol} "
            f"PnL: {net_pnl:,.2f} ({pnl_pct:.2%}) | Reason: {exit_reason}"
        )

        return trade

    def _get_unrealized_pnl(self, trade: PaperTrade) -> float:
        """Calculate unrealized PnL for an open trade.

        Args:
            trade: Open trade.

        Returns:
            Unrealized PnL.
        """
        if trade.exit_price > 0:
            return trade.pnl  # Already closed

        if trade.side == "BUY":
            return (trade.entry_price - trade.entry_price) * trade.quantity
        return 0.0  # Simplified

    def update_equity(self, current_prices: Dict[str, float]) -> None:
        """Update equity curve with current market prices.

        Args:
            current_prices: Dictionary mapping symbol to current price.
        """
        unrealized_pnl = 0.0

        for trade in self.open_trades:
            current_price = current_prices.get(trade.symbol, trade.entry_price)

            if trade.side == "BUY":
                unrealized_pnl += (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl += (trade.entry_price - current_price) * trade.quantity

        current_equity = self.balance + unrealized_pnl

        self.equity_curve.append({
            "timestamp": datetime.utcnow(),
            "equity": current_equity,
            "balance": self.balance,
            "unrealized_pnl": unrealized_pnl,
            "open_positions": len(self.open_trades),
        })

    def get_performance(self) -> Dict:
        """Get paper trading performance statistics.

        Returns:
            Dictionary with performance metrics.
        """
        if not self.completed_trades:
            return {"status": "No trades yet"}

        pnls = [t.pnl for t in self.completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0

        return {
            "total_trades": len(self.completed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "current_balance": self.balance,
            "open_positions": len(self.open_trades),
            "days_running": (datetime.utcnow() - self._start_time).days,
            "trade_history": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                }
                for t in self.completed_trades
            ],
        }

    def reset(self) -> None:
        """Reset paper trading state."""
        self.balance = self.config.initial_balance
        self.open_trades.clear()
        self.completed_trades.clear()
        self.equity_curve.clear()
        self._trade_counter = 0
        self._start_time = datetime.utcnow()
        logger.info("Paper trading state reset.")
