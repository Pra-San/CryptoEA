"""Position sizing modules for CryptoEA.

Implements multiple position sizing strategies:
- ATR-based fixed fractional (default)
- Kelly Criterion (fractional)
- Volatility targeting
- Fixed dollar amount
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of a position sizing calculation.

    Attributes:
        quantity: Number of base asset units to trade.
        notional: Total trade value in quote currency.
        risk_amount: Amount of equity at risk.
        stop_loss_price: Calculated stop loss price.
        take_profit_price: Calculated take profit price.
        risk_pct: Risk as percentage of equity.
        sizing_method: Which method was used.
    """

    quantity: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_pct: float = 0.0
    sizing_method: str = "unknown"


@dataclass
class RiskLimits:
    """Portfolio-level risk limits.

    Attributes:
        max_risk_per_trade: Maximum risk per trade as fraction of equity.
        max_portfolio_risk: Maximum total open risk as fraction of equity.
        max_correlated_exposure: Max exposure to correlated assets.
        daily_loss_limit: Daily loss limit that halts trading.
        weekly_loss_limit: Weekly loss limit that reduces size.
        max_drawdown_halt: Max DD from peak that stops all trading.
        max_position_size_pct: Max single position as % of equity.
    """

    max_risk_per_trade: float = 0.01  # 1% of equity
    max_portfolio_risk: float = 0.05  # 5% total open risk
    max_correlated_exposure: float = 0.03  # 3% in correlated assets
    daily_loss_limit: float = 0.03  # 3% daily loss -> halt
    weekly_loss_limit: float = 0.06  # 6% weekly loss -> reduce 50%
    max_drawdown_halt: float = 0.10  # 10% DD -> full halt
    max_position_size_pct: float = 0.10  # Max 10% of equity


class PositionSizer:
    """Calculate optimal position sizes based on various methods.

    The default method is ATR-based fixed fractional sizing,
    which adjusts position size based on market volatility.
    """

    def __init__(
        self,
        equity: float = 10000.0,
        risk_limits: Optional[RiskLimits] = None,
        default_method: str = "atr",
    ) -> None:
        """Initialize the position sizer.

        Args:
            equity: Current account equity in quote currency.
            risk_limits: Portfolio-level risk constraints.
            default_method: Default sizing method ('atr', 'kelly', 'vol_target', 'fixed').
        """
        self.equity = equity
        self.risk_limits = risk_limits or RiskLimits()
        self.default_method = default_method
        self._method = default_method

    def set_method(self, method: str) -> None:
        """Set the sizing method.

        Args:
            method: One of 'atr', 'kelly', 'vol_target', 'fixed'.
        """
        valid_methods = ['atr', 'kelly', 'vol_target', 'fixed']
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'. Valid: {valid_methods}")
        self._method = method

    def calculate(
        self,
        price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> PositionSizeResult:
        """Calculate position size using the selected method.

        Args:
            price: Current asset price.
            stop_loss_price: Stop loss price (for ATR/fixed methods).
            take_profit_price: Take profit price.
            atr: Current ATR value (for ATR method).
            volatility: Current volatility (for vol_target method).
            win_rate: Historical win rate (for Kelly method).
            avg_win_loss_ratio: Avg win/avg loss ratio (for Kelly method).

        Returns:
            PositionSizeResult with calculated sizing.
        """
        if self._method == 'atr':
            return self._atr_sizing(
                price, stop_loss_price, take_profit_price, atr
            )
        elif self._method == 'kelly':
            return self._kelly_sizing(
                price, stop_loss_price, take_profit_price,
                win_rate, avg_win_loss_ratio
            )
        elif self._method == 'vol_target':
            return self._vol_target_sizing(
                price, stop_loss_price, volatility
            )
        elif self._method == 'fixed':
            return self._fixed_sizing(price)
        else:
            raise ValueError(f"Unknown sizing method: {self._method}")

    def _atr_sizing(
        self,
        price: float,
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        atr: Optional[float],
    ) -> PositionSizeResult:
        """ATR-based fixed fractional position sizing.

        position_size = (account_risk_per_trade * account_equity) /
                        (atr_multiplier * current_atr)

        Args:
            price: Current asset price.
            stop_loss_price: Stop loss price.
            take_profit_price: Take profit price.
            atr: Current ATR value.

        Returns:
            PositionSizeResult.
        """
        result = PositionSizeResult(sizing_method='atr')

        if atr is None or atr <= 0:
            logger.warning("ATR is None or <= 0, using default sizing")
            atr = price * 0.02  # Default 2% of price

        # Calculate stop distance
        if stop_loss_price is not None:
            stop_distance = abs(price - stop_loss_price)
        else:
            stop_distance = atr * 2  # Default 2x ATR stop

        if stop_distance <= 0:
            logger.warning("Stop distance is <= 0, skipping trade")
            return result

        # Risk per trade (percentage of equity)
        risk_pct = min(
            self.risk_limits.max_risk_per_trade,
            self.equity * 0.01  # Cap at 1%
        )

        # Risk amount in quote currency
        risk_amount = self.equity * risk_pct

        # Position size in base asset
        quantity = risk_amount / stop_distance

        # Enforce max position size
        max_notional = self.equity * self.risk_limits.max_position_size_pct
        max_quantity = max_notional / price
        quantity = min(quantity, max_quantity)

        # Calculate notional
        result.notional = quantity * price
        result.quantity = quantity
        result.risk_amount = risk_amount
        result.risk_pct = risk_amount / self.equity
        result.stop_loss_price = stop_loss_price or price - stop_distance
        result.take_profit_price = take_profit_price

        return result

    def _kelly_sizing(
        self,
        price: float,
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float],
    ) -> PositionSizeResult:
        """Fractional Kelly Criterion position sizing.

        Kelly % = win_rate - ((1 - win_rate) / win_loss_ratio)

        We use fractional Kelly (25-50% of full Kelly) to reduce volatility.

        Args:
            price: Current asset price.
            stop_loss_price: Stop loss price.
            take_profit_price: Take profit price.
            win_rate: Historical win rate.
            avg_win_loss_ratio: Avg win / Avg loss ratio.

        Returns:
            PositionSizeResult.
        """
        result = PositionSizeResult(sizing_method='kelly')

        if win_rate is None or avg_win_loss_ratio is None:
            logger.warning("Kelly requires win_rate and win_loss_ratio")
            return self._atr_sizing(price, stop_loss_price, take_profit_price, None)

        # Full Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / avg_win_loss_ratio)

        # Fractional Kelly (use 50% of full Kelly)
        fractional_kelly = kelly_pct * 0.5

        # Cap at maximum allowed
        fractional_kelly = min(fractional_kelly, self.risk_limits.max_risk_per_trade)

        # Don't trade if Kelly is negative
        if fractional_kelly <= 0:
            logger.info("Kelly suggests no trade (negative edge)")
            return result

        # Calculate position size
        risk_amount = self.equity * fractional_kelly

        # Stop distance
        if stop_loss_price is not None:
            stop_distance = abs(price - stop_loss_price)
        else:
            stop_distance = price * 0.02  # Default 2%

        if stop_distance <= 0:
            return result

        quantity = risk_amount / stop_distance

        # Enforce max position size
        max_notional = self.equity * self.risk_limits.max_position_size_pct
        max_quantity = max_notional / price
        quantity = min(quantity, max_quantity)

        result.notional = quantity * price
        result.quantity = quantity
        result.risk_amount = risk_amount
        result.risk_pct = fractional_kelly
        result.stop_loss_price = stop_loss_price
        result.take_profit_price = take_profit_price

        return result

    def _vol_target_sizing(
        self,
        price: float,
        stop_loss_price: Optional[float],
        volatility: Optional[float],
    ) -> PositionSizeResult:
        """Volatility-targeting position sizing.

        Adjusts position size to maintain constant volatility exposure.

        Args:
            price: Current asset price.
            stop_loss_price: Stop loss price.
            volatility: Current asset volatility.

        Returns:
            PositionSizeResult.
        """
        result = PositionSizeResult(sizing_method='vol_target')

        if volatility is None or volatility <= 0:
            logger.warning("Volatility is None or <= 0, using default sizing")
            return self._atr_sizing(price, stop_loss_price, None, None)

        # Target annual volatility (e.g., 15%)
        target_vol = 0.15

        # Scale position by inverse of current volatility
        vol_ratio = target_vol / volatility

        # Base position (1% risk)
        base_risk = self.equity * self.risk_limits.max_risk_per_trade

        # Adjusted position
        adjusted_risk = base_risk * vol_ratio

        # Cap at maximum
        adjusted_risk = min(adjusted_risk, self.equity * self.risk_limits.max_risk_per_trade)

        # Calculate quantity
        if stop_loss_price is not None:
            stop_distance = abs(price - stop_loss_price)
        else:
            stop_distance = price * volatility * np.sqrt(24)  # 24 1h bars

        if stop_distance <= 0:
            return result

        quantity = adjusted_risk / stop_distance

        # Enforce max position size
        max_notional = self.equity * self.risk_limits.max_position_size_pct
        max_quantity = max_notional / price
        quantity = min(quantity, max_quantity)

        result.notional = quantity * price
        result.quantity = quantity
        result.risk_amount = adjusted_risk
        result.risk_pct = adjusted_risk / self.equity
        result.stop_loss_price = stop_loss_price

        return result

    def _fixed_sizing(self, price: float) -> PositionSizeResult:
        """Fixed dollar amount position sizing.

        Simple approach: trade a fixed dollar amount regardless of market conditions.

        Args:
            price: Current asset price.

        Returns:
            PositionSizeResult.
        """
        result = PositionSizeResult(sizing_method='fixed')

        # Trade 5% of equity in this asset
        notional = self.equity * 0.05

        # Enforce max position size
        max_notional = self.equity * self.risk_limits.max_position_size_pct
        notional = min(notional, max_notional)

        quantity = notional / price

        result.notional = notional
        result.quantity = quantity
        result.risk_amount = notional  # Full notional is at risk
        result.risk_pct = notional / self.equity

        return result

    def update_equity(self, new_equity: float) -> None:
        """Update the current equity for position sizing.

        Args:
            new_equity: New account equity value.
        """
        self.equity = new_equity
