"""V1: Volatility-Adjusted Momentum Strategy with Regime Filtering.

Hypothesis:
  Crypto markets exhibit persistent momentum that is amplified in
  trending regimes and dampened in ranging regimes. By combining:
  1. Volatility-adjusted momentum (momentum / volatility)
  2. Regime detection (trending vs ranging via Hurst exponent)
  3. Volume confirmation (volume spike on breakout)
  
  We can filter out low-quality signals during choppy markets,
  improving win rate and Sharpe ratio while maintaining trade count.

Edge Rationale:
  - Momentum persistence in crypto is well-documented (hours to days)
  - Most retail traders chase momentum too late; we enter early
  - Volatility adjustment prevents over-leveraging during high-vol
  - Regime filter avoids trading during low-volatility ranges
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from strategy.base import BaseStrategy, Signal, SignalResult, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class MomentumConfig(StrategyConfig):
    """Configuration for the volatility-adjusted momentum strategy.

    Attributes:
        momentum_lookback: Bars to look back for momentum calculation.
        volatility_lookback: Bars for rolling volatility.
        hurst_lookback: Bars for Hurst exponent calculation.
        hurst_trending_threshold: Hurst > this = trending (0.5 is random walk).
        volume_spike_threshold: Volume ratio > this = confirmed breakout.
        min_volatility: Minimum volatility to trade (avoid dead markets).
        max_volatility: Maximum volatility (avoid crash zones).
        atr_stop_multiplier: Stop loss distance in ATR multiples.
        atr_tp_multiplier: Take profit distance in ATR multiples.
        pyramiding: Whether to add to winning positions.
        max_pyramid_levels: Max additional positions when pyramiding.
    """

    # Momentum parameters
    momentum_lookback: int = 24  # 24 bars (24h on 1h chart)
    volatility_lookback: int = 50
    hurst_lookback: int = 100
    hurst_trending_threshold: float = 0.55

    # Volume confirmation
    volume_spike_threshold: float = 1.5  # 1.5x average volume

    # Volatility filters
    min_volatility: float = 0.001  # 0.1% daily vol
    max_volatility: float = 0.05  # 5% daily vol (avoid crashes)

    # Risk parameters
    atr_stop_multiplier: float = 2.0  # 2x ATR stop
    atr_tp_multiplier: float = 3.0  # 3x ATR target (1:1.5 R:R)

    # Pyramiding
    pyramiding: bool = False
    max_pyramid_levels: int = 2


class VolatilityAdjustedMomentumStrategy(BaseStrategy):
    """Volatility-Adjusted Momentum with Regime Filtering.

    Strategy Logic:
    1. Calculate volatility-adjusted momentum score
    2. Detect market regime using Hurst exponent
    3. Only trade when: trending regime + strong momentum + volume confirmation
    4. Use ATR-based dynamic stops and targets
    5. Optional pyramiding on winning trades
    """

    def __init__(self, config: Optional[MomentumConfig] = None) -> None:
        """Initialize the momentum strategy.

        Args:
            config: Strategy configuration with momentum-specific params.
        """
        super().__init__(config or MomentumConfig())
        self._config: MomentumConfig = (
            config if isinstance(config, MomentumConfig) else MomentumConfig()
        )

    def get_name(self) -> str:
        """Return strategy name."""
        return "VolatilityAdjustedMomentum"

    def get_version(self) -> str:
        """Return strategy version."""
        return "v1"

    def initialize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Initialize data with momentum, regime, and volume features.

        Args:
            data: Raw OHLCV DataFrame.

        Returns:
            DataFrame with all strategy features computed.
        """
        df = data.copy()

        # --- Momentum Score ---
        # Momentum = returns over lookback, adjusted by volatility
        momentum = df["close"].pct_change(self._config.momentum_lookback)
        volatility = df["log_return"].rolling(
            self._config.volatility_lookback, min_periods=1
        ).std()

        # Volatility-adjusted momentum (Sharpe-like ratio per bar)
        df["vol_adj_momentum"] = np.where(
            volatility > 0,
            momentum / (volatility * np.sqrt(self._config.momentum_lookback)),
            0,
        )

        # --- Regime Detection: Hurst Exponent ---
        # Rolling Hurst exponent using rescaled range (R/S) analysis
        # H > 0.5 = trending (persistent), H < 0.5 = mean-reverting
        df["hurst_exponent"] = self._calculate_hurst(
            df["close"], self._config.hurst_lookback
        )
        df["is_trending"] = df["hurst_exponent"] > self._config.hurst_trending_threshold

        # --- Volatility Regime ---
        daily_vol = df["log_return"].rolling(24, min_periods=1).std() * np.sqrt(24)
        df["daily_volatility"] = daily_vol
        df["in_volatility_zone"] = (
            (daily_vol > self._config.min_volatility)
            & (daily_vol < self._config.max_volatility)
        )

        # --- Volume Confirmation ---
        vol_sma = df["volume"].rolling(50, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / vol_sma
        df["volume_confirmed"] = df["volume_ratio"] > self._config.volume_spike_threshold

        # --- Dynamic ATR for SL/TP ---
        df["atr_14"] = self._calculate_atr(df, 14)

        # --- Composite Signal Score (0-1) ---
        # Combine momentum, regime, and volume into a single score
        mom_score = df["vol_adj_momentum"].clip(-3, 3) / 3  # Normalize to [-1, 1]
        regime_score = df["is_trending"].astype(float)  # 0 or 1
        volume_score = (df["volume_ratio"] / self._config.volume_spike_threshold).clip(0, 1)

        df["composite_score"] = (
            0.5 * mom_score
            + 0.3 * regime_score * (mom_score > 0).astype(float)
            + 0.2 * volume_score
        )

        # --- Signal Generation ---
        # Long: composite_score > 0.3, trending, volume confirmed
        # Short: composite_score < -0.3, trending, volume confirmed
        df["signal"] = 0
        df.loc[
            (df["composite_score"] > 0.3)
            & (df["is_trending"])
            & (df["volume_confirmed"])
            & (df["in_volatility_zone"]),
            "signal",
        ] = 1  # LONG

        df.loc[
            (df["composite_score"] < -0.3)
            & (df["is_trending"])
            & (df["volume_confirmed"])
            & (df["in_volatility_zone"]),
            "signal",
        ] = -1  # SHORT

        # --- Stop Loss & Take Profit ---
        if self._config.use_stop_loss:
            df["stop_loss"] = np.where(
                df["signal"] == 1,
                df["close"] - self._config.atr_stop_multiplier * df["atr_14"],
                np.where(
                    df["signal"] == -1,
                    df["close"] + self._config.atr_stop_multiplier * df["atr_14"],
                    np.nan,
                ),
            )

        if self._config.use_take_profit:
            df["take_profit"] = np.where(
                df["signal"] == 1,
                df["close"] + self._config.atr_tp_multiplier * df["atr_14"],
                np.where(
                    df["signal"] == -1,
                    df["close"] - self._config.atr_tp_multiplier * df["atr_14"],
                    np.nan,
                ),
            )

        # Drop rows with NaN from rolling calculations
        initial_len = len(df)
        df = df.dropna()
        logger.debug(f"  After dropping NaN: {len(df):,} bars (from {initial_len:,})")

        return df

    def generate_signals(self, data: pd.DataFrame) -> List[SignalResult]:
        """Generate trading signals from the enriched data.

        Args:
            data: DataFrame from initialize().

        Returns:
            List of SignalResult objects.
        """
        signals = []

        for idx, row in data.iterrows():
            signal_result = SignalResult()

            if row.get("signal", 0) == 1:
                signal_result.signal = Signal.LONG
                signal_result.strength = min(1.0, abs(row.get("composite_score", 0)))
                signal_result.stop_loss = row.get("stop_loss")
                signal_result.take_profit = row.get("take_profit")
                signal_result.position_size = self._config.risk_per_trade
                signal_result.metadata = {
                    "momentum": row.get("vol_adj_momentum", 0),
                    "hurst": row.get("hurst_exponent", 0.5),
                    "volume_ratio": row.get("volume_ratio", 1.0),
                    "atr": row.get("atr_14", 0),
                }

            elif row.get("signal", 0) == -1:
                signal_result.signal = Signal.SHORT
                signal_result.strength = min(1.0, abs(row.get("composite_score", 0)))
                signal_result.stop_loss = row.get("stop_loss")
                signal_result.take_profit = row.get("take_profit")
                signal_result.position_size = self._config.risk_per_trade
                signal_result.metadata = {
                    "momentum": row.get("vol_adj_momentum", 0),
                    "hurst": row.get("hurst_exponent", 0.5),
                    "volume_ratio": row.get("volume_ratio", 1.0),
                    "atr": row.get("atr_14", 0),
                }

            else:
                signal_result.signal = Signal.FLAT

            self.record_signal(signal_result)
            signals.append(signal_result)

        return signals

    def _calculate_hurst(
        self, series: pd.Series, window: int
    ) -> pd.Series:
        """Calculate rolling Hurst exponent using R/S analysis.

        The Hurst exponent H indicates market behavior:
        - H > 0.5: Trending (persistent)
        - H = 0.5: Random walk
        - H < 0.5: Mean-reverting (anti-persistent)

        Args:
            series: Price series.
            window: Rolling window for calculation.

        Returns:
            Series of Hurst exponents.
        """
        result = pd.Series(np.nan, index=series.index)

        # Use a vectorized approximation for speed
        # For each window, compute R/S ratio
        returns = series.pct_change().dropna()

        for start in range(0, len(returns) - window, max(1, window // 10)):
            end = min(start + window, len(returns))
            window_data = returns.iloc[start:end]

            if len(window_data) < 20:
                continue

            # Rescaled range calculation
            cum_returns = window_data.cumsum()
            R = (cum_returns.max() - cum_returns.min())  # Range
            S = window_data.std()  # Standard deviation

            if S > 0:
                RS = R / S
                # Hurst = log(RS) / log(n), but we use a simpler approximation
                # A more robust method uses log-log regression of RS vs n
                h = np.log(RS + 1e-10) / np.log(len(window_data) + 1e-10)
                result.iloc[start:end // 2 + start] = np.clip(h, 0, 1)

        # Forward fill small gaps
        result = result.ffill(limit=5).bfill()

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range.

        Args:
            df: DataFrame with OHLC data.
            period: Lookback period.

        Returns:
            Series with ATR values.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()

        return atr
