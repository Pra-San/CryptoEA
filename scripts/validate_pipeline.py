#!/usr/bin/env python3
"""Validation script to test data pipeline and initial backtest."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from strategy.v1.strategy import VolatilityAdjustedMomentumStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics


def setup_logging() -> None:
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def validate_data(symbol: str, data_dir: str) -> pd.DataFrame:
    """Validate and preprocess data for a symbol.

    Args:
        symbol: Trading pair (e.g., 'btcusdt').
        data_dir: Directory containing CSV files.

    Returns:
        Preprocessed DataFrame with features.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DATA VALIDATION: %s", symbol.upper())
    logger.info("=" * 60)

    # Use DataLoader class API
    loader = DataLoader(use_cache=False, verbose=True)

    try:
        df = loader.load_symbol(symbol, timeframe="1m")
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        return pd.DataFrame()

    if df.empty:
        logger.error("Failed to load data")
        return pd.DataFrame()

    logger.info("Raw data loaded: %d rows", len(df))
    logger.info("Date range: %s to %s",
                str(df.index.min()),
                str(df.index.max()))
    logger.info("Columns: %s", list(df.columns))

    # Preprocess - use preprocess() method
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df, symbol=symbol)

    if df.empty:
        logger.error("Preprocessing failed")
        return pd.DataFrame()

    logger.info("Preprocessed data: %d rows", len(df))
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:15])  # Show first 15

    # Print basic statistics
    logger.info("\nPrice Statistics:")
    logger.info("  Open: %.2f - %.2f (mean: %.2f)",
                df['open'].min(), df['open'].max(), df['open'].mean())
    logger.info("  Close: %.2f - %.2f (mean: %.2f)",
                df['close'].min(), df['close'].max(), df['close'].mean())

    logger.info("\nVolume Statistics:")
    logger.info("  Volume: %.2f - %.2f (mean: %.2f)",
                df['volume'].min(), df['volume'].max(), df['volume'].mean())

    return df


def run_backtest(symbol: str, df: pd.DataFrame, data_dir: str) -> dict:
    """Run backtest on preprocessed data.

    Args:
        symbol: Trading pair.
        df: Preprocessed DataFrame.
        data_dir: Directory for saving results.

    Returns:
        Dictionary containing backtest results.
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BACKTEST: %s", symbol.upper())
    logger.info("=" * 60)

    # Create strategy
    strategy = VolatilityAdjustedMomentumStrategy()
    logger.info("Strategy: %s", strategy.__class__.__name__)
    logger.info("Strategy parameters: %s", strategy.get_params())

    # Run backtest
    initial_capital = 10000.0
    logger.info("Initial capital: $%.2f", initial_capital)

    engine = BacktestEngine(
        data=df,
        strategy=strategy,
        initial_capital=initial_capital,
    )

    logger.info("Running backtest...")
    trade_log, equity_curve = engine.run()

    # Calculate metrics
    metrics = BacktestMetrics(trade_log, equity_curve)
    results = metrics.calculate_all()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            logger.info("  %-25s: %.4f", key, value)
        else:
            logger.info("  %-25s: %s", key, value)

    # Save trade log for further analysis
    output_path = Path(data_dir).parent / f"{symbol}_backtest_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\nResults saved to: %s", output_path)

    return results


def main() -> None:
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    data_dir = "/Users/speketi/Projects/TEA/data/binance/m1"

    # Symbols to test
    symbols = ["btcusdt", "ethusdt", "solusdt"]

    all_results = {}

    for symbol in symbols:
        logger.info("\n" + "#" * 80)
        logger.info("# PHASE 1: VALIDATION & BACKTEST")
        logger.info("#" * 80)

        # Validate data
        df = validate_data(symbol, data_dir)

        if df.empty:
            logger.warning("Skipping %s due to validation failure", symbol)
            continue

        # Run backtest
        results = run_backtest(symbol, df, data_dir)
        all_results[symbol] = results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        calmar = results.get("calmar_ratio", 0)
        sharpe = results.get("sharpe_ratio", 0)
        max_dd = results.get("max_drawdown_pct", 0)
        win_rate = results.get("win_rate", 0)
        total_trades = results.get("total_trades", 0)

        logger.info("%-10s | Calmar: %.2f | Sharpe: %.2f | MaxDD: %.2f%% | WinRate: %.2f%% | Trades: %d",
                    symbol.upper(), calmar, sharpe, max_dd, win_rate * 100, total_trades)

    # Edge criteria check
    logger.info("\n" + "=" * 60)
    logger.info("EDGE CRITERIA CHECK")
    logger.info("=" * 60)
    logger.info("Target: Calmar > 1.5, MaxDD < 15%, WinRate > 50%")

    for symbol, results in all_results.items():
        calmar = results.get("calmar_ratio", 0)
        max_dd = results.get("max_drawdown_pct", 0)
        win_rate = results.get("win_rate", 0)

        passes = (
            calmar > 1.5 and
            max_dd < 15.0 and
            win_rate > 0.50
        )

        status = "✅ PASS" if passes else "❌ FAIL"
        logger.info("%-10s: %s (Calmar: %.2f, MaxDD: %.2f%%, WinRate: %.2f%%)",
                    symbol.upper(), status, calmar, max_dd, win_rate * 100)


if __name__ == "__main__":
    main()
