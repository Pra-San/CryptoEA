#!/usr/bin/env python3
"""CLI script for parameter optimization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from backtest.optimizer import ParameterOptimizer
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


def run_optimization(
    symbol: str,
    data_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
    n_trials: int = 100,
    verbose: bool = False,
) -> dict:
    """Run parameter optimization for a symbol.

    Args:
        symbol: Trading pair (e.g., 'btcusdt').
        data_dir: Directory containing CSV files.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).
        n_trials: Number of Optuna trials.
        verbose: Whether to enable debug logging.

    Returns:
        Dictionary containing optimization results.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PARAMETER OPTIMIZATION")
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

    # Run optimization
    optimizer = ParameterOptimizer()
    results = optimizer.calibrate(
        train_df=df,
        strategy=strategy,
        initial_capital=10000.0,
        n_trials=n_trials,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)

    best_params = results.get("best_params", {})
    best_value = results.get("best_value", 0)

    logger.info("Best Calmar Ratio: %.4f", best_value)
    logger.info("Best Parameters:")

    for key, value in best_params.items():
        logger.info("  %s: %s", key, value)

    importance = results.get("parameter_importance", {})

    if importance:
        logger.info("\nParameter Importance:")
        for param, score in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("  %s: %.4f", param, score)

    # Save results
    output_path = Path(data_dir).parent / "optimization_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\nResults saved to: %s", output_path)

    return results


def main() -> None:
    """Main entry point for optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Run parameter optimization for a Crypto EA"
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
        help="Directory containing CSV files (default: data/binance/m1)",
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
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    run_optimization(
        symbol=args.symbol,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        n_trials=args.n_trials,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
