#!/usr/bin/env python3
"""CLI script for walk-forward analysis."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from data.loader import load_csv
from data.preprocessor import DataPreprocessor
from strategy.v1.strategy import VolatilityAdjustedMomentumStrategy


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Whether to enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_walk_forward(
    symbol: str,
    data_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
    train_months: int = 12,
    test_months: int = 3,
    overlap_months: int = 1,
    initial_capital: float = 10000.0,
    verbose: bool = False,
) -> dict:
    """Run walk-forward analysis for a symbol.

    Args:
        symbol: Trading pair (e.g., 'btcusdt').
        data_dir: Directory containing CSV files.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).
        train_months: Training window size in months.
        test_months: Testing window size in months.
        overlap_months: Overlap between windows in months.
        initial_capital: Starting capital.
        verbose: Whether to enable debug logging.

    Returns:
        Dictionary containing walk-forward results.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("WALK-FORWARD ANALYSIS")
    logger.info("=" * 60)

    # Load and preprocess data
    csv_path = Path(data_dir) / f"{symbol}_1m_spot.csv"

    if not csv_path.exists():
        logger.error("Data file not found: %s", csv_path)
        sys.exit(1)

    raw_df = load_csv(str(csv_path), start_date=start_date, end_date=end_date)

    if raw_df.empty:
        logger.error("No data available after filtering")
        sys.exit(1)

    preprocessor = DataPreprocessor()
    df = preprocessor.process(raw_df)

    if df.empty:
        logger.error("No data available after preprocessing")
        sys.exit(1)

    logger.info(
        "Data loaded: %d rows, %s to %s",
        len(df),
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )

    # Create strategy
    strategy = VolatilityAdjustedMomentumStrategy()

    # Create walk-forward config
    config = WalkForwardConfig(
        train_months=train_months,
        test_months=test_months,
        overlap_months=overlap_months,
    )

    # Run walk-forward analysis
    analyzer = WalkForwardAnalyzer(config=config)
    results = analyzer.run(
        symbol=symbol,
        strategy=strategy,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        verbose=verbose,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info("Total Windows: %d", results.get("total_windows", 0))

    aggregate = results.get("aggregate", {})

    if aggregate:
        logger.info("\nAggregate Statistics:")
        for key, value in aggregate.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", key, value)
            else:
                logger.info("  %s: %s", key, value)

    windows = results.get("windows", [])

    if windows:
        logger.info("\nPer-Window Results:")
        for window in windows:
            idx = window.get("window_index", 0)
            metrics = window.get("metrics", {})
            calmar = metrics.get("calmar_ratio", 0)
            max_dd = metrics.get("max_drawdown_pct", 0)
            win_rate = metrics.get("win_rate", 0)

            logger.info(
                "  Window %d: Calmar=%.2f, MaxDD=%.2f%%, WinRate=%.2f%%",
                idx,
                calmar,
                max_dd,
                win_rate * 100,
            )

    # Save results
    output_path = Path(data_dir).parent / "walk_forward_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\nResults saved to: %s", output_path)

    return results


def main() -> None:
    """Main entry point for walk-forward CLI."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward analysis for a Crypto EA"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="btcusdt",
        help="Trading pair (default: btcusdt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/binance/m1",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=12,
        help="Training window in months (default: 12)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Testing window in months (default: 3)",
    )
    parser.add_argument(
        "--overlap-months",
        type=int,
        default=1,
        help="Overlap between windows in months (default: 1)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    run_walk_forward(
        symbol=args.symbol,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=args.train_months,
        test_months=args.test_months,
        overlap_months=args.overlap_months,
        initial_capital=args.initial_capital,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
