"""Data loader for Binance historical OHLCV data.

Handles loading, validation, and caching of Binance kline data
in multiple formats (CSV, Parquet).
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import get_binance_data_path, get_cache_path

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality for a symbol/timeframe combination."""

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_bars: int
    missing_bars: int
    gap_count: int
    duplicate_count: int
    zero_volume_bars: int
    zero_trade_bars: int
    quality_score: float  # 0-1, higher is better
    issues: List[str] = field(default_factory=list)


@dataclass
class SymbolInfo:
    """Metadata about a symbol from the manifest."""

    symbol: str
    market: str
    timeframe: str
    rows: int
    first_time: str
    last_time: str
    source: str


class DataLoader:
    """Load and validate Binance historical OHLCV data.

    Supports CSV and Parquet formats with automatic caching.
    Validates data integrity including gaps, duplicates, and outliers.
    """

    # Standard Binance column names
    COLUMNS = ["timestamp", "time", "open", "high", "low", "close", "volume"]

    # Required OHLCV columns
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(
        self,
        use_cache: bool = True,
        cache_format: str = "parquet",
        verbose: bool = False,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            use_cache: Whether to use parquet cache for faster loading.
            cache_format: Format for cached data (parquet, csv).
            verbose: Whether to print detailed loading info.
        """
        self.use_cache = use_cache
        self.cache_format = cache_format
        self.verbose = verbose
        self._manifest: Optional[Dict] = None

    def discover_symbols(self, data_dir: Optional[str] = None) -> List[SymbolInfo]:
        """Discover available symbols from the manifest file.

        Args:
            data_dir: Optional path to data directory. Defaults to binance root.

        Returns:
            List of SymbolInfo objects describing available data.
        """
        if data_dir is None:
            data_dir = get_binance_data_path()

        manifest_path = Path(data_dir) / "manifest.json"
        if not manifest_path.exists():
            logger.warning(f"No manifest found at {manifest_path}")
            return []

        try:
            with open(manifest_path, "r") as f:
                manifest = pd.read_json(f)

            self._manifest = manifest
            symbols = []
            for sym_data in manifest.get("symbols", []):
                info = SymbolInfo(
                    symbol=sym_data["symbol"],
                    market=sym_data.get("market", "spot"),
                    timeframe=sym_data.get("timeframe", "1m"),
                    rows=sym_data.get("rows", 0),
                    first_time=sym_data.get("firstTime", ""),
                    last_time=sym_data.get("lastTime", ""),
                    source=sym_data.get("source", ""),
                )
                symbols.append(info)
                logger.info(
                    f"  Symbol: {info.symbol} | {info.timeframe} | "
                    f"{info.rows:,} bars | {info.first_time} to {info.last_time}"
                )

            return symbols

        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return []

    def load_symbol(
        self,
        symbol: str,
        timeframe: str = "1m",
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Load historical data for a single symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT').
            timeframe: Timeframe (e.g., '1m', '5m', '1h').
            use_cache: Override instance setting.

        Returns:
            DataFrame with OHLCV data indexed by timestamp.
        """
        if use_cache is None:
            use_cache = self.use_cache

        cache_path = get_cache_path(f"{symbol}_{timeframe}")
        cached_file = cache_path / f"{symbol}_{timeframe}.{self.cache_format}"

        # Try cache first
        if use_cache and cached_file.exists():
            try:
                if self.verbose:
                    logger.info(f"Loading cached data: {cached_file}")
                start = time.time()
                df = pd.read_parquet(cached_file)
                # Restore timestamp as index (cache was written with index=False)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp")
                # Verify we got a proper DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"  Cache has invalid index ({type(df.index).__name__}), rebuilding from CSV")
                    raise FileNotFoundError("Invalid cache index")
                elapsed = time.time() - start
                if self.verbose:
                    logger.info(f"  Loaded {len(df):,} bars in {elapsed:.2f}s")
                return df
            except Exception as e:
                logger.warning(f"Cache load failed, falling back to CSV: {e}")

        # Load from CSV
        csv_path = get_binance_data_path(symbol, timeframe)
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        if self.verbose:
            logger.info(f"Loading from CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f}MB)")

        start = time.time()
        df = pd.read_csv(
            csv_path,
            usecols=["timestamp", "time", "open", "high", "low", "close", "volume"],
        )
        elapsed = time.time() - start
        logger.info(f"  Loaded {len(df):,} bars in {elapsed:.2f}s")

        # Process and cache
        df = self._process_raw(df)

        # Save to cache
        if use_cache:
            try:
                cached_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cached_file, index=False)
                if self.verbose:
                    cache_size = cached_file.stat().st_size / 1e6
                    logger.info(f"  Cached to {cached_file} ({cache_size:.1f}MB)")
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

        return df

    def _process_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw CSV data into standard format.

        Args:
            df: Raw DataFrame from CSV.

        Returns:
            Processed DataFrame with proper types and index.
        """
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Set timestamp as index
        df = df.set_index("timestamp")

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where essential columns are NaN
        initial_len = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning(f"  Dropped {dropped} rows with NaN values")

        # Sort by timestamp
        df = df.sort_index()

        # Remove duplicates (keep first)
        initial_len = len(df)
        df = df[~df.index.duplicated(keep="first")]
        dupes = initial_len - len(df)
        if dupes > 0:
            logger.warning(f"  Removed {dupes} duplicate timestamps")

        return df

    def validate_data(
        self, df: pd.DataFrame, symbol: str, timeframe: str = "1m"
    ) -> DataQualityReport:
        """Validate data quality and generate a quality report.

        Args:
            df: DataFrame to validate.
            symbol: Trading symbol.
            timeframe: Timeframe.

        Returns:
            DataQualityReport with findings.
        """
        issues = []
        missing_bars = 0
        gap_count = 0
        duplicate_count = 0
        zero_volume_count = 0
        zero_trades_count = 0

        # Check for duplicates
        initial_len = len(df)
        dup_mask = df.index.duplicated()
        duplicate_count = dup_mask.sum()
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate timestamps found")

        # Check for gaps based on timeframe
        expected_interval = self._get_expected_interval(timeframe)
        actual_diffs = df.index.to_series().diff().dt.total_seconds()
        gaps = actual_diffs[actual_diffs > expected_interval * 1.5]
        gap_count = len(gaps)

        if gap_count > 0:
            missing_bars = int(gaps.sum() / expected_interval) - gap_count
            issues.append(
                f"{gap_count} gaps detected (~{missing_bars} missing bars)"
            )

        # Check for zero volume bars (common in early BTC data)
        zero_volume_count = (df["volume"] == 0).sum()
        if zero_volume_count > 0:
            pct = zero_volume_count / len(df) * 100
            issues.append(f"{zero_volume_count:,} bars with zero volume ({pct:.1f}%)")

        # Check for zero trade bars (high == low == open == close)
        zero_trades_count = (df["high"] == df["low"]).sum()
        if zero_trades_count > 0:
            pct = zero_trades_count / len(df) * 100
            issues.append(
                f"{zero_trades_count:,} bars with zero trades ({pct:.1f}%)"
            )

        # Calculate quality score
        total_bars = len(df)
        penalty = 0
        if gap_count > 0:
            penalty += min(gap_count / total_bars * 100, 30)
        if zero_volume_count > 0:
            penalty += min(zero_volume_count / total_bars * 100, 20)
        if zero_trades_count > 0:
            penalty += min(zero_trades_count / total_bars * 100, 20)
        if duplicate_count > 0:
            penalty += min(duplicate_count / total_bars * 100, 10)

        quality_score = max(0, min(100, 100 - penalty))

        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            start_date=str(df.index.min()),
            end_date=str(df.index.max()),
            total_bars=total_bars,
            missing_bars=missing_bars,
            gap_count=gap_count,
            duplicate_count=duplicate_count,
            zero_volume_bars=zero_volume_count,
            zero_trade_bars=zero_trades_count,
            quality_score=quality_score,
            issues=issues,
        )

        return report

    def _get_expected_interval(self, timeframe: str) -> int:
        """Get expected interval in seconds for a timeframe string.

        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h').

        Returns:
            Interval in seconds.
        """
        interval_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "1w": 604800,
        }
        return interval_map.get(timeframe, 60)

    def load_and_validate(
        self, symbol: str, timeframe: str = "1m"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Load data and generate quality report in one call.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe.

        Returns:
            Tuple of (DataFrame, DataQualityReport).
        """
        df = self.load_symbol(symbol, timeframe)
        report = self.validate_data(df, symbol, timeframe)

        logger.info(
            f"Data Quality for {symbol} ({timeframe}): "
            f"Score={report.quality_score:.0f}/100, "
            f"{report.total_bars:,} bars, "
            f"{report.start_date} to {report.end_date}"
        )

        if report.issues:
            for issue in report.issues:
                logger.warning(f"  - {issue}")

        return df, report

    def load_multiple_symbols(
        self, symbols: List[str], timeframe: str = "1m"
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols.

        Args:
            symbols: List of symbol strings.
            timeframe: Timeframe.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load_symbol(symbol, timeframe)
                logger.info(f"  Loaded {symbol}: {len(result[symbol]):,} bars")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        return result
