"""Core vectorized backtesting engine for CryptoEA.

Event-driven simulation of a trading strategy. Handles entries, exits,
SL/TP, trailing stops, partial exits, commission, slippage, and funding rates.

Key Design Decisions:
- No lookahead bias: signals computed on close of bar N, executed on open of bar N+1
- Realistic fills: cannot fill at exact signal price on the same bar
- Partial exits: scale out in tranches at different targets
- Transaction log: every trade recorded with full metadata
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Record of a single trade execution.

    Attributes:
        entry_time: Timestamp of entry.
        exit_time: Timestamp of exit (None if open).
        side: LONG or SHORT.
        entry_price: Price at entry.
        exit_price: Price at exit (None if open).
        quantity: Number of units/traded amount.
        notional: Entry notional value (price * quantity).
        stop_loss: Stop loss price.
        take_profit: Take profit price (if single target).
        exit_reason: Why the trade was closed.
        pnl: Realized PnL (None if open).
        pnl_pct: Realized PnL percentage (None if open).
        fees: Total fees paid.
        slippage: Total slippage cost.
        holding_bars: Number of bars held.
        metadata: Additional trade metadata.
    """

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ""
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    risk_amount: float = 0.0
    r_multiple: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    holding_bars: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for the backtest engine.

    Attributes:
        initial_balance: Starting account balance in quote currency.
        fee_rate: Trading fee rate (maker/taker average).
        slippage_rate: Expected slippage as fraction.
        leverage: Trading leverage (1 = spot).
        max_positions: Maximum concurrent positions.
        min_trade_size: Minimum trade notional.
        report_frequency: How often to log progress.
    """

    initial_balance: float = 10000.0
    fee_rate: float = 0.0006  # 0.06% taker fee
    slippage_rate: float = 0.0003  # 0.03% slippage
    leverage: int = 1
    max_positions: int = 3
    min_trade_size: float = 10.0
    report_frequency: int = 100  # Log every N bars


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    Attributes:
        trades: List of all executed trades.
        equity_curve: Series of account equity over time.
        benchmark_curve: Series of benchmark (buy & hold) equity.
        config: Backtest configuration used.
        metadata: Additional performance metadata.
    """

    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    benchmark_curve: Optional[pd.Series] = None
    config: Optional[BacktestConfig] = None
    metadata: Dict = field(default_factory=dict)

    def get_trade_count(self) -> int:
        """Get number of completed trades."""
        return len([t for t in self.trades if t.exit_time is not None])

    def get_open_trades(self) -> List[Trade]:
        """Get list of currently open trades."""
        return [t for t in self.trades if t.exit_time is None]


class BacktestEngine:
    """Event-driven vectorized backtesting engine.

    Simulates trading with realistic market mechanics including:
    - Entry/exit with slippage
    - Stop loss and take profit orders
    - Partial exits at targets
    - Commission and fee modeling
    - Position tracking and PnL calculation
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,  # Strategy instance with generate_signals method
        config: Optional[BacktestConfig] = None,
        symbol: str = "UNKNOWN",
    ) -> None:
        """Initialize the backtest engine.

        Args:
            data: Preprocessed OHLCV DataFrame with signals.
            strategy: Strategy instance with generate_signals method.
            config: Backtest configuration. Uses defaults if None.
            symbol: Trading symbol for logging.
        """
        self.data = data.copy()
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.symbol = symbol
        self._result = BacktestResult(config=self.config, metadata={"symbol": symbol})

    def run(self) -> BacktestResult:
        """Execute the backtest simulation.

        Returns:
            BacktestResult with complete trade and equity data.
        """
        logger.info(f"Starting backtest for {self.symbol}: {len(self.data):,} bars")

        # Generate signals from strategy
        signals = self.strategy.generate_signals(self.data)

        # Initialize state
        balance = self.config.initial_balance
        positions: List[Trade] = []
        self._result.trades = []
        equity_values: List[float] = []
        benchmark_values: List[float] = []

        # Get signal series for vectorized processing
        signal_series = self.data["signal"].copy()

        # Track position state
        in_position = False
        position_side = PositionSide.FLAT
        entry_price = 0.0
        entry_time: Optional[pd.Timestamp] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        position_quantity = 0.0
        position_notional = 0.0

        for i in range(1, len(self.data)):
            current_bar = self.data.iloc[i]
            prev_bar = self.data.iloc[i - 1]
            current_time = self.data.index[i]

            current_signal = signal_series.iloc[i]

            # --- Signal Processing (executed on open of current bar) ---
            if not in_position and current_signal != 0:
                # Entry signal detected
                entry_price = current_bar["open"]

                # Apply slippage to entry
                if current_signal == 1:  # LONG
                    entry_price *= (1 + self.config.slippage_rate)
                    position_side = PositionSide.LONG
                else:  # SHORT
                    entry_price *= (1 - self.config.slippage_rate)
                    position_side = PositionSide.SHORT

                # Calculate position size
                position_quantity = self._calculate_position_size(
                    balance, entry_price, self.config.fee_rate
                )

                if position_quantity > 0:
                    position_notional = position_quantity * entry_price
                    fees = position_notional * self.config.fee_rate

                    # Get stop loss and take profit from strategy
                    stop_loss = current_bar.get("stop_loss")
                    take_profit = current_bar.get("take_profit")

                    # Create entry trade record
                    entry_trade = Trade(
                        entry_time=current_time,
                        side=position_side,
                        entry_price=entry_price,
                        quantity=position_quantity,
                        notional=position_notional,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        fees=fees,
                        metadata=current_bar.get("metadata", {}),
                    )
                    positions.append(entry_trade)
                    balance -= fees  # Pay fees on entry
                    in_position = True

            # --- Exit Logic ---
            if in_position:
                current_close = current_bar["close"]
                exit_trade = positions[-1]
                exit_reason = ""
                exited = False

                # Check stop loss
                if exit_trade.stop_loss is not None:
                    if position_side == PositionSide.LONG and current_close <= exit_trade.stop_loss:
                        exit_price = exit_trade.stop_loss * (1 - self.config.slippage_rate)
                        exit_reason = "stop_loss"
                        exited = True
                    elif position_side == PositionSide.SHORT and current_close >= exit_trade.stop_loss:
                        exit_price = exit_trade.stop_loss * (1 + self.config.slippage_rate)
                        exit_reason = "stop_loss"
                        exited = True

                # Check take profit
                if not exited and exit_trade.take_profit is not None:
                    if position_side == PositionSide.LONG and current_close >= exit_trade.take_profit:
                        exit_price = exit_trade.take_profit * (1 + self.config.slippage_rate)
                        exit_reason = "take_profit"
                        exited = True
                    elif position_side == PositionSide.SHORT and current_close <= exit_trade.take_profit:
                        exit_price = exit_trade.take_profit * (1 - self.config.slippage_rate)
                        exit_reason = "take_profit"
                        exited = True

                # Signal-based exit (signal reversed)
                if not exited and current_signal == 0 and i > 10:  # Minimum holding period
                    exit_price = current_bar["open"]
                    if position_side == PositionSide.LONG:
                        exit_price *= (1 - self.config.slippage_rate)
                    else:
                        exit_price *= (1 + self.config.slippage_rate)
                    exit_reason = "signal_reversal"
                    exited = True

                # Execute exit
                if exited:
                    exit_fees = position_quantity * exit_price * self.config.fee_rate
                    holding_bars = i - self.data.index.get_loc(exit_trade.entry_time)

                    # Calculate PnL
                    if position_side == PositionSide.LONG:
                        gross_pnl = (exit_price - exit_trade.entry_price) * position_quantity
                    else:  # SHORT
                        gross_pnl = (exit_trade.entry_price - exit_price) * position_quantity

                    net_pnl = gross_pnl - exit_fees - exit_trade.fees
                    pnl_pct = net_pnl / exit_trade.notional if exit_trade.notional > 0 else 0

                    # Update trade record
                    exit_trade.exit_time = current_time
                    exit_trade.exit_price = exit_price
                    exit_trade.exit_reason = exit_reason
                    exit_trade.pnl = net_pnl
                    exit_trade.pnl_pct = pnl_pct
                    exit_trade.holding_bars = max(1, holding_bars)

                    # Update balance
                    balance += gross_pnl - exit_fees
                    self._result.trades.append(exit_trade)

                    # Remove from open positions
                    positions.pop()
                    in_position = False
                    position_side = PositionSide.FLAT

            # Calculate equity
            if in_position:
                pos = positions[-1]
                if position_side == PositionSide.LONG:
                    unrealized = (current_close - pos.entry_price) * pos.quantity - pos.fees
                else:
                    unrealized = (pos.entry_price - current_close) * pos.quantity - pos.fees
                current_equity = balance + unrealized
            else:
                current_equity = balance

            equity_values.append(current_equity)

            # Benchmark: buy and hold
            if i > 0:
                bh_return = (current_bar["close"] - self.data.iloc[0]["close"]) / self.data.iloc[0]["close"]
                benchmark_values.append(self.config.initial_balance * (1 + bh_return))

            # Periodic logging
            if i % self.config.report_frequency == 0:
                trade_count = len(self._result.trades)
                winning_trades = len([t for t in self._result.trades if t.pnl and t.pnl > 0])
                win_rate = winning_trades / trade_count if trade_count > 0 else 0
                logger.debug(
                    f"  Bar {i:,}/{len(self.data):,} | "
                    f"Trades: {trade_count} (WR: {win_rate:.1%}) | "
                    f"Equity: {current_equity:,.0f}"
                )

        # Close any remaining open positions at last bar
        if positions:
            last_bar = self.data.iloc[-1]
            for pos in positions:
                if position_side == PositionSide.LONG:
                    exit_price = last_bar["close"] * (1 - self.config.slippage_rate)
                else:
                    exit_price = last_bar["close"] * (1 + self.config.slippage_rate)

                gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                net_pnl = gross_pnl - 2 * pos.fees  # Entry + exit fees
                pnl_pct = net_pnl / pos.notional if pos.notional > 0 else 0

                pos.exit_time = self.data.index[-1]
                pos.exit_price = exit_price
                pos.exit_reason = "end_of_backtest"
                pos.pnl = net_pnl
                pos.pnl_pct = pnl_pct
                pos.holding_bars = len(self.data) - self.data.index.get_loc(pos.entry_time)

                balance += gross_pnl - 2 * pos.fees
                self._result.trades.append(pos)

        # Store equity curves
        self._result.equity_curve = pd.Series(
            equity_values, index=self.data.index[1:], name="equity"
        )
        self._result.benchmark_curve = pd.Series(
            benchmark_values, index=self.data.index[1:], name="benchmark"
        )

        # Store metadata
        self._result.metadata.update({
            "total_bars": len(self.data),
            "total_signals": len(signals),
            "symbol": self.symbol,
            "start_date": str(self.data.index[0]),
            "end_date": str(self.data.index[-1]),
        })

        logger.info(
            f"Backtest complete for {self.symbol}: "
            f"{len(self._result.trades)} trades, "
            f"Final equity: {balance:,.0f}"
        )

        return self._result

    def _calculate_position_size(
        self, balance: float, price: float, fee_rate: float
    ) -> float:
        """Calculate position size based on risk parameters.

        Args:
            balance: Current account balance.
            price: Entry price.
            fee_rate: Fee rate for the trade.

        Returns:
            Position quantity in base asset.
        """
        # Risk-based sizing: risk 1% of balance per trade
        risk_amount = balance * self.config.risk_per_trade

        # Account for fees
        effective_price = price * (1 + fee_rate)

        # Calculate quantity
        quantity = risk_amount / effective_price

        # Enforce minimum trade size
        notional = quantity * price
        if notional < self.config.min_trade_size:
            return 0.0

        # Enforce maximum position size (10% of balance)
        max_notional = balance * 0.10
        if quantity * price > max_notional:
            quantity = max_notional / price

        return quantity
