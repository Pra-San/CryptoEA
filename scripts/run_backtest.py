#!/usr/bin/env python3
"""CLI script to run backtests on CryptoEA strategies.

Usage:
    python run_backtest.py --symbol BTCUSDT --timeframe 1h --version 1
    python run_backtest.py --symbol ETHUSDT --timeframe 4h --start 2023-01-01 --end 2025-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor, PreprocessingConfig
from strategy.v1.strategy import (
    VolatilityAdjustedMomentumStrategy,
    MomentumConfig,
)
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import BacktestMetrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtests on CryptoEA strategies"
    )
    parser.add_argument(
        "--symbol", "-s", type=str, default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--timeframe", "-t", type=str, default="1h",
        help="Timeframe (default: 1h)"
    )
    parser.add_argument(
        "--version", "-v", type=int, default=1,
        help="Strategy version (default: 1)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0,
        help="Initial balance (default: 10000)"
    )
    parser.add_argument(
        "--risk-per-trade", type=float, default=0.01,
        help="Risk per trade as fraction (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--resample", type=str, default=None,
        help="Resample to this timeframe (e.g., '4h')"
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for backtesting."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("backtest")

    logger.info(f"=" * 60)
    logger.info(f"  CryptoEA Backtest Runner")
    logger.info(f"  Symbol: {args.symbol} | Timeframe: {args.timeframe}")
    logger.info(f"  Balance: {args.balance:,.0f} | Risk/Trade: {args.risk_per_trade:.1%}")
    logger.info(f"=" * 60)

    # Step 1: Load data
    logger.info("\n[1/5] Loading data...")
    loader = DataLoader(use_cache=True, verbose=args.verbose)
    df = loader.load_symbol(args.symbol, args.timeframe, use_cache=True)

    # Apply date filter
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
        logger.info(f"  Filtered to start: {args.start}")
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]
        logger.info(f"  Filtered to end: {args.end}")

    logger.info(f"  Loaded {len(df):,} bars: {df.index[0]} to {df.index[-1]}")

    # Step 2: Preprocess data
    logger.info("\n[2/5] Preprocessing data...")
    preproc_config = PreprocessingConfig(
        resample_to=args.resample,
        add_indicators=True,
    )
    preprocessor = DataPreprocessor(preproc_config)
    df = preprocessor.preprocess(df, args.symbol)

    logger.info(f"  After preprocessing: {len(df):,} bars, {len(df.columns)} columns")

    # Step 3: Initialize strategy
    logger.info("\n[3/5] Initializing strategy...")
    strategy_config = MomentumConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_per_trade=args.risk_per_trade,
        max_positions=3,
    )
    strategy = VolatilityAdjustedMomentumStrategy(strategy_config)

    # Initialize strategy features
    df = strategy.initialize(df)
    logger.info(f"  Strategy features computed: {len(df.columns)} columns")

    # Step 4: Run backtest
    logger.info("\n[4/5] Running backtest...")
    bt_config = BacktestConfig(
        initial_balance=args.balance,
        fee_rate=0.0006,  # 0.06% taker fee
        slippage_rate=0.0003,  # 0.03% slippage
        leverage=1,
        max_positions=3,
        report_frequency=1000,
    )

    engine = BacktestEngine(
        data=df,
        strategy=strategy,
        config=bt_config,
        symbol=args.symbol,
    )

    result = engine.run()
    logger.info(f"  Completed: {result.get_trade_count()} trades")

    # Step 5: Calculate and display metrics
    logger.info("\n[5/5] Calculating metrics...")
    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate(result, df)
    metrics_calc.print_report(metrics)

    # Save results
    reports_dir = project_root / "backtest" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"{args.symbol}_{args.timeframe}_v{args.version}.txt"
    with open(report_file, "w") as f:
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write(f"Period: {df.index[0]} to {df.index[-1]}\n")
        f.write(f"Bars: {len(df):,}\n")
        f.write(f"Strategy: {strategy.get_name()} v{strategy.get_version()}\n\n")
        f.write(metrics.summary + "\n")

    logger.info(f"  Report saved to: {report_file}")


if __name__ == "__main__":
    import pandas as pd
    main()
