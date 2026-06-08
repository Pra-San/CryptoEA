"""Drawdown and circuit breaker management for CryptoEA.

Implements circuit breakers that halt or reduce trading
when predefined drawdown thresholds are breached.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    NORMAL = "normal"
    WARNING = "warning"  # Reduce positions
    HALT = "halt"  # Stop all trading
    REVIEW = "review"  # Full halt, requires manual review


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers.

    Attributes:
        daily_loss_limit: Daily loss % that triggers halt.
        weekly_loss_limit: Weekly loss % that triggers size reduction.
        monthly_loss_limit: Monthly loss % that triggers full halt.
        max_drawdown_from_peak: Max DD from peak that halts trading.
        consecutive_loss_limit: Max consecutive losses before halting.
        recovery_bars: Bars of no-trading after halt before resuming.
    """

    daily_loss_limit: float = 0.03  # 3% daily loss
    weekly_loss_limit: float = 0.06  # 6% weekly loss
    monthly_loss_limit: float = 0.10  # 10% monthly loss
    max_drawdown_from_peak: float = 0.10  # 10% DD from peak
    consecutive_loss_limit: int = 5  # 5 consecutive losses
    recovery_bars: int = 24  # 24 bars after halt before resuming


@dataclass
class CircuitBreakerStatus:
    """Current status of circuit breakers.

    Attributes:
        state: Current circuit breaker state.
        daily_pnl: Today's PnL as fraction of equity.
        weekly_pnl: This week's PnL as fraction of equity.
        monthly_pnl: This month's PnL as fraction of equity.
        max_dd_from_peak: Max drawdown from peak equity.
        consecutive_losses: Number of consecutive losing trades.
        last_halt_time: When the last halt was triggered.
        size_multiplier: Current position size multiplier (0-1).
        reason: Why the circuit breaker is in current state.
    """

    state: CircuitBreakerState = CircuitBreakerState.NORMAL
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_dd_from_peak: float = 0.0
    consecutive_losses: int = 0
    last_halt_time: Optional[datetime] = None
    size_multiplier: float = 1.0
    reason: str = ""


class DrawdownManager:
    """Manages circuit breakers and risk limits.

    Monitors daily, weekly, monthly, and peak-to-trough drawdowns,
    and triggers appropriate risk reductions or halts.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        """Initialize the drawdown manager.

        Args:
            config: Circuit breaker configuration. Uses defaults if None.
        """
        self.config = config or CircuitBreakerConfig()
        self.status = CircuitBreakerStatus()
        self._equity_history: list = []
        self._trade_history: list = []

    def update(
        self,
        current_equity: float,
        peak_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        recent_trades: Optional[list] = None,
    ) -> CircuitBreakerStatus:
        """Update circuit breaker status based on current conditions.

        Args:
            current_equity: Current account equity.
            peak_equity: Peak equity (all-time high of account).
            daily_pnl: Today's PnL as fraction of starting equity.
            weekly_pnl: This week's PnL as fraction of starting equity.
            monthly_pnl: This month's PnL as fraction of starting equity.
            recent_trades: List of recent trade PnL values.

        Returns:
            Updated CircuitBreakerStatus.
        """
        self._equity_history.append(current_equity)

        # Calculate max drawdown from peak
        if peak_equity > 0:
            self.status.max_dd_from_peak = (peak_equity - current_equity) / peak_equity
        else:
            self.status.max_dd_from_peak = 0

        # Update PnL metrics
        self.status.daily_pnl = daily_pnl
        self.status.weekly_pnl = weekly_pnl
        self.status.monthly_pnl = monthly_pnl

        # Update consecutive losses
        if recent_trades:
            losses = sum(1 for t in recent_trades if t < 0)
            # Count consecutive losses from the end
            consecutive = 0
            for t in reversed(recent_trades):
                if t < 0:
                    consecutive += 1
                else:
                    break
            self.status.consecutive_losses = consecutive
        else:
            self.status.consecutive_losses = 0

        # Determine circuit breaker state
        self._evaluate_state()

        return self.status

    def should_trade(self) -> bool:
        """Check if trading should be allowed.

        Returns:
            True if trading is allowed, False if halted.
        """
        return self.status.state in (
            CircuitBreakerState.NORMAL,
            CircuitBreakerState.WARNING,
        )

    def get_size_multiplier(self) -> float:
        """Get the current position size multiplier.

        Returns:
            Multiplier for position sizes (0.0 to 1.0).
        """
        return self.status.size_multiplier

    def get_status(self) -> str:
        """Get a human-readable status string.

        Returns:
            Status description.
        """
        return (
            f"Circuit Breaker: {self.status.state.value} | "
            f"Size Multiplier: {self.status.size_multiplier:.0%} | "
            f"Max DD: {self.status.max_dd_from_peak:.1%} | "
            f"Consecutive Losses: {self.status.consecutive_losses} | "
            f"Reason: {self.status.reason}"
        )

    def _evaluate_state(self) -> None:
        """Evaluate all circuit breaker conditions and set state."""
        # Reset to normal
        state = CircuitBreakerState.NORMAL
        size_mult = 1.0
        reason = ""

        # Check daily loss limit
        if self.status.daily_pnl <= -self.config.daily_loss_limit:
            state = CircuitBreakerState.HALT
            size_mult = 0.0
            reason = f"Daily loss limit hit ({self.status.daily_pnl:.1%})"

        # Check weekly loss limit
        elif self.status.weekly_pnl <= -self.config.weekly_loss_limit:
            state = CircuitBreakerState.WARNING
            size_mult = 0.5
            reason = f"Weekly loss limit hit ({self.status.weekly_pnl:.1%})"

        # Check monthly loss limit
        elif self.status.monthly_pnl <= -self.config.monthly_loss_limit:
            state = CircuitBreakerState.HALT
            size_mult = 0.0
            reason = f"Monthly loss limit hit ({self.status.monthly_pnl:.1%})"

        # Check max drawdown from peak
        elif self.status.max_dd_from_peak >= self.config.max_drawdown_from_peak:
            state = CircuitBreakerState.HALT
            size_mult = 0.0
            reason = f"Max drawdown from peak hit ({self.status.max_dd_from_peak:.1%})"

        # Check consecutive losses
        elif self.status.consecutive_losses >= self.config.consecutive_loss_limit:
            state = CircuitBreakerState.WARNING
            size_mult = 0.5
            reason = f"Consecutive losses ({self.status.consecutive_losses})"

        self.status.state = state
        self.status.size_multiplier = size_mult
        self.status.reason = reason

        if state != CircuitBreakerState.NORMAL:
            logger.warning(f"Circuit breaker triggered: {reason}")
        else:
            logger.debug(f"Circuit breaker: {reason}")
