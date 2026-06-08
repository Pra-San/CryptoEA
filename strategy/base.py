"""Abstract base strategy class for CryptoEA.

Defines the interface all strategies must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class SignalResult:
    """Result of signal generation for a single bar.

    Attributes:
        signal: Trading signal (LONG, SHORT, FLAT).
        strength: Signal strength 0-1 (confidence).
        stop_loss: Optional stop loss price.
        take_profit: Optional take profit price.
        position_size: Optional position size as fraction of equity.
        metadata: Additional context for the signal.
    """

    signal: Signal = Signal.FLAT
    strength: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Base configuration for all strategies.

    Attributes:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        timeframe: Timeframe (e.g., '1h').
        risk_per_trade: Risk per trade as fraction of equity.
        max_positions: Maximum simultaneous positions.
        min_trade_size: Minimum trade size in quote currency.
        max_trade_size: Maximum trade size in quote currency.
        use_stop_loss: Whether to use stop losses.
        use_take_profit: Whether to use take profits.
        trailing_stop: Whether to use trailing stops.
        trailing_stop_pct: Trailing stop distance as fraction.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    risk_per_trade: float = 0.01  # 1% of equity per trade
    max_positions: int = 3
    min_trade_size: float = 10.0  # 10 USDT minimum
    max_trade_size: float = 50000.0  # 50,000 USDT maximum
    use_stop_loss: bool = True
    use_take_profit: bool = True
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.02  # 2% trailing distance


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement:
    - initialize(): Setup indicators and parameters
    - generate_signals(): Produce trading signals from data
    - get_config(): Return strategy configuration
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize the base strategy.

        Args:
            config: Strategy configuration. Uses defaults if None.
        """
        self.config = config or StrategyConfig()
        self._initialized = False
        self._history: List[SignalResult] = []

    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Initialize and enrich the data with strategy-specific features.

        Args:
            data: Raw OHLCV DataFrame.

        Returns:
            Enriched DataFrame with strategy features.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from the enriched data.

        Args:
            data: Enriched DataFrame from initialize().

        Returns:
            Series of Signal values indexed by timestamp.
        """
        pass

    def get_config(self) -> StrategyConfig:
        """Return the strategy configuration.

        Returns:
            StrategyConfig instance.
        """
        return self.config

    def get_name(self) -> str:
        """Return the strategy name.

        Returns:
            Strategy name string.
        """
        return self.__class__.__name__

    def get_version(self) -> str:
        """Return the strategy version.

        Returns:
            Version string.
        """
        return "v1"

    def record_signal(self, signal: SignalResult) -> None:
        """Record a signal for tracking.

        Args:
            signal: SignalResult to record.
        """
        self._history.append(signal)

    def get_statistics(self) -> Dict:
        """Get strategy execution statistics.

        Returns:
            Dictionary with signal statistics.
        """
        if not self._history:
            return {"total_signals": 0}

        signals = [s.signal for s in self._history]
        long_count = sum(1 for s in signals if s == Signal.LONG)
        short_count = sum(1 for s in signals if s == Signal.SHORT)
        flat_count = sum(1 for s in signals if s == Signal.FLAT)

        return {
            "total_signals": len(self._history),
            "long_signals": long_count,
            "short_signals": short_count,
            "flat_signals": flat_count,
            "avg_strength": np.mean([s.strength for s in self._history]),
        }
