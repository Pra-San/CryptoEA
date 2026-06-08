"""Data preprocessing and feature engineering for CryptoEA.

Handles cleaning, resampling, and computing technical indicators
for backtesting and live trading.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    max_gap_fill_bars: int = 3  # Max consecutive NaN bars to forward-fill
    min_volume_ratio: float = 0.0  # Minimum volume ratio to keep (0 = keep all)
    log_returns: bool = True  # Compute log returns
    add_indicators: bool = True  # Compute technical indicators
    fee_rate_maker: float = 0.0004  # 0.04% maker fee
    fee_rate_taker: float = 0.0006  # 0.06% taker fee
    slippage_rate: float = 0.0003  # 0.03% default slippage
    resample_to: Optional[str] = None  # Resample to this timeframe (e.g., '1h')


class DataPreprocessor:
    """Clean, resample, and engineer features from raw OHLCV data.

    Handles missing data, computes returns, technical indicators,
    and applies realistic fee/slippage models.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()

    def preprocess(
        self, df: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """Run full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame.
            symbol: Symbol name for logging.

        Returns:
            Preprocessed DataFrame with features.
        """
        logger.info(f"Preprocessing {symbol}: {len(df):,} bars")

        # Step 1: Clean data
        df = self._clean_data(df)

        # Step 2: Resample if requested
        if self.config.resample_to:
            df = self._resample(df, self.config.resample_to)

        # Step 3: Engineer features
        df = self._engineer_features(df)

        # Step 4: Add fee/slippage metadata
        df = self._add_cost_model(df)

        logger.info(f"  Preprocessed {symbol}: {len(df):,} bars, "
                    f"{len(df.columns)} columns")

        return df

    def process(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Backward-compatible alias for preprocess()."""
        return self.preprocess(df, symbol=symbol)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data: handle missing values, outliers, duplicates.

        Args:
            df: Raw DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()

        if "timestamp" in df.columns:
            timestamp = df["timestamp"]
            if pd.api.types.is_numeric_dtype(timestamp):
                unit = "ms" if timestamp.dropna().astype("int64").median() > 10**11 else "s"
                df["timestamp"] = pd.to_datetime(timestamp, unit=unit, utc=True)
            else:
                df["timestamp"] = pd.to_datetime(timestamp, utc=True)
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or timestamp column")

        original_len = len(df)

        # Sort by index
        df = df.sort_index()

        # Forward-fill small gaps (up to max_gap_fill_bars)
        df = df.ffill(limit=self.config.max_gap_fill_bars)

        # Backward-fill any remaining NaN at the start
        df = df.bfill()

        # Drop any remaining rows with NaN in OHLC
        df = df.dropna(subset=["open", "high", "low", "close"])

        filled = original_len - len(df)
        if filled > 0:
            logger.debug(f"  Filled/dropped {filled} rows")

        # Remove obvious outliers: high < low, or price changes > 50% in 1m
        df = self._remove_outliers(df)

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove obvious data quality outliers.

        Args:
            df: DataFrame to filter.

        Returns:
            Filtered DataFrame.
        """
        before_len = len(df)

        # Remove bars where high < low (impossible)
        mask = df["high"] >= df["low"]
        if mask.sum() < len(df):
            df = df[mask]

        # Remove bars with > 50% price change (likely data errors in 1m)
        if len(df) > 1:
            returns = df["close"].pct_change(fill_method=None).abs()
            extreme_mask = returns.isna() | (returns <= 0.50)
            extreme_count = (~extreme_mask).sum()
            if extreme_count > 0:
                logger.debug(f"  Removed {extreme_count} extreme moves (>50%)")

            df = df[extreme_mask]

        after_len = len(df)
        removed = before_len - after_len
        if removed > 0:
            logger.debug(f"  Removed {removed} outlier bars")

        return df

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to a different timeframe.

        Args:
            df: DataFrame to resample.
            timeframe: Target timeframe (e.g., '1h', '4h').

        Returns:
            Resampled DataFrame.
        """
        if df.index.tz is None:
            df = df.tz_localize("UTC")

        # Resample OHLC
        df_ohlc = df[["open", "high", "low", "close"]].resample(timeframe).ohlc()
        # Resample volume separately (ohlc doesn't handle volume)
        df_vol = df[["volume"]].resample(timeframe).sum()

        # Flatten multi-level columns to single level
        df_ohlc.columns = [col[1] for col in df_ohlc.columns]
        # Add volume from the separately-resampled volume DataFrame
        df_ohlc = df_ohlc.join(df_vol, rsuffix="_vol")
        # Rename volume column
        if "volume_vol" in df_ohlc.columns:
            df_ohlc = df_ohlc.rename(columns={"volume_vol": "volume"})

        # Drop rows with NaN
        df_resampled = df_ohlc.dropna()

        logger.info(f"  Resampled to {timeframe}: {len(df_resampled):,} bars "
                    f"(from {len(df):,})")

        return df_resampled

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators and features.

        Args:
            df: Cleaned DataFrame.

        Returns:
            DataFrame with added features.
        """
        if not self.config.add_indicators:
            return df

        close = df["close"].squeeze() if hasattr(df["close"], "squeeze") else df["close"]
        high = df["high"].squeeze() if hasattr(df["high"], "squeeze") else df["high"]
        low = df["low"].squeeze() if hasattr(df["low"], "squeeze") else df["low"]
        volume = df["volume"].squeeze() if hasattr(df["volume"], "squeeze") else df["volume"]

        # --- Returns ---
        if self.config.log_returns:
            df["log_return"] = np.log(close / close.shift(1))

        df["pct_change"] = close.pct_change(fill_method=None)

        # --- Volatility features ---
        for period in [14, 30, 50, 100]:
            df[f"volatility_{period}"] = df["log_return"].rolling(
                period, min_periods=1
            ).std()

        # --- ATR (Average True Range) ---
        df["atr_14"] = self._calculate_atr(df, 14)
        df["atr_7"] = self._calculate_atr(df, 7)

        # --- RSI (Relative Strength Index) ---
        df["rsi_14"] = self._calculate_rsi(close, 14)
        df["rsi_7"] = self._calculate_rsi(close, 7)

        # --- MACD ---
        ema_fast = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=1).mean()
        df["macd"] = ema_fast - ema_slow

        # Signal line (9-period EMA of MACD)
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- Moving averages ---
        for period in [20, 50, 100, 200]:
            df[f"sma_{period}"] = close.rolling(
                period, min_periods=1
            ).mean()
            df[f"ema_{period}"] = close.ewm(
                span=period, adjust=False, min_periods=1
            ).mean()

        # --- Volume features ---
        df["volume_sma_20"] = volume.rolling(20, min_periods=1).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # Volume-weighted price (cumulative)
        df["vwp"] = (close * volume).cumsum() / volume.cumsum()

        # --- Price position within range ---
        for period in [20, 50, 100]:
            rolling_high = high.rolling(period, min_periods=1).max()
            rolling_low = low.rolling(period, min_periods=1).min()
            price_range = rolling_high - rolling_low
            df[f"price_position_{period}"] = np.where(
                price_range > 0,
                (close - rolling_low) / price_range,
                0.5,
            )

        # --- Bollinger Bands ---
        bb_std = df["log_return"].rolling(20, min_periods=1).std() * np.sqrt(20)
        df["bb_middle"] = close.rolling(20, min_periods=1).mean()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_middle"] * bb_std / close
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_middle"] * bb_std / close
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # --- Donchian Channels ---
        for period in [20, 50]:
            df[f"donchian_high_{period}"] = high.rolling(
                period, min_periods=1
            ).max()
            df[f"donchian_low_{period}"] = low.rolling(
                period, min_periods=1
            ).min()
            df[f"donchian_mid_{period}"] = (
                df[f"donchian_high_{period}"]
                + df[f"donchian_low_{period}"]
            ) / 2

        # Drop rows with NaN from indicators
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.debug(f"  Dropped {dropped} rows from indicator computation")

        return df

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

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            close: Price series.
            period: Lookback period.

        Returns:
            Series with RSI values.
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(1.0)  # Default when no loss

        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _add_cost_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fee and slippage metadata for realistic backtesting.

        Args:
            df: DataFrame to annotate.

        Returns:
            DataFrame with cost metadata columns.
        """
        # Round-trip cost (entry + exit)
        df["round_trip_cost"] = (
            self.config.fee_rate_maker + self.config.fee_rate_taker
        ) + (2 * self.config.slippage_rate)

        # Per-trade cost as percentage
        df["entry_fee"] = self.config.fee_rate_taker  # Assume market orders
        df["exit_fee"] = self.config.fee_rate_taker
        df["slippage_bps"] = self.config.slippage_rate * 10000  # in basis points

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names.

        Returns:
            List of feature column names.
        """
        feature_patterns = [
            "log_return", "pct_change", "volatility_", "atr_",
            "rsi_", "macd", "sma_", "ema_", "volume_", "vwp",
            "price_position_", "bb_", "donchian_", "round_trip_cost"
        ]

        all_columns = [
            "open", "high", "low", "close", "volume",
            "log_return", "pct_change",
            "atr_14", "atr_7",
            "rsi_14", "rsi_7",
            "macd", "macd_signal", "macd_hist",
            "volume_sma_20", "volume_ratio", "vwp",
            "round_trip_cost", "entry_fee", "exit_fee", "slippage_bps"
        ]

        return all_columns
