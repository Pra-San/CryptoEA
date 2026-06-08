#!/usr/bin/env python3
"""CLI script for live/paper trading."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from live.execution import ExecutionEngine, ExecutionConfig, OrderSide, OrderType
from live.paper_trader import PaperTrader
from live.data_feed import DataFeed, DataFeedConfig
from live.monitor import TradingMonitor, MonitorConfig
from live.state_manager import StateManager
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


async def run_paper_trading(
    symbol: str,
    initial_capital: float,
    data_dir: str,
    state_dir: str,
    use_telegram: bool,
    telegram_token: str | None,
    telegram_chat_id: str | None,
    verbose: bool = False,
) -> None:
    """Run paper trading simulation.

    Args:
        symbol: Trading pair.
        initial_capital: Starting capital.
        data_dir: Directory containing historical CSV files.
        state_dir: Directory for state persistence.
        use_telegram: Whether to enable Telegram alerts.
        telegram_token: Telegram bot token.
        telegram_chat_id: Telegram chat ID.
        verbose: Whether to enable debug logging.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PAPER TRADING SIMULATION")
    logger.info("=" * 60)
    logger.info("Symbol: %s", symbol)
    logger.info("Initial Capital: $%.2f", initial_capital)
    logger.info("State Directory: %s", state_dir)

    # Initialize components
    state_manager = StateManager(state_dir=state_dir)
    execution = ExecutionEngine(
        config=ExecutionConfig(
            symbol=symbol,
            state_dir=state_dir,
        )
    )
    paper_trader = PaperTrader(
        execution=execution,
        initial_capital=initial_capital,
        state_manager=state_manager,
    )
    monitor = TradingMonitor(
        config=MonitorConfig(
            state_dir=state_dir,
            use_telegram=use_telegram,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
        )
    )

    # Load historical data for backtest
    from data.loader import load_csv
    from data.preprocessor import DataPreprocessor
    from backtest.engine import BacktestEngine
    from backtest.metrics import BacktestMetrics

    csv_path = Path(data_dir) / f"{symbol}_1m_spot.csv"

    if not csv_path.exists():
        logger.error("Data file not found: %s", csv_path)
        sys.exit(1)

    raw_df = load_csv(str(csv_path))
    preprocessor = DataPreprocessor()
    df = preprocessor.process(raw_df)

    if df.empty:
        logger.error("No data available after preprocessing")
        sys.exit(1)

    logger.info(
        "Loaded %d rows of data",
        len(df),
    )

    # Run initial backtest on historical data
    strategy = VolatilityAdjustedMomentumStrategy()

    logger.info("\nRunning initial backtest on historical data...")

    engine = BacktestEngine(
        data=df,
        strategy=strategy,
        initial_capital=initial_capital,
    )

    trade_log, equity_curve = engine.run()
    metrics = BacktestMetrics(trade_log, equity_curve)
    metrics_dict = metrics.calculate_all()

    logger.info("=" * 60)
    logger.info("HISTORICAL BACKTEST RESULTS")
    logger.info("=" * 60)

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    # Start paper trading (simulated real-time)
    logger.info("\nStarting paper trading simulation...")
    logger.info("Paper trading is simulated. No real orders are placed.")
    logger.info("Press Ctrl+C to stop and view results.")

    try:
        await paper_trader.run_simulation(
            df=df,
            strategy=strategy,
            monitor=monitor,
        )

        # Print final report
        report = paper_trader.get_performance_report()
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING FINAL REPORT")
        logger.info("=" * 60)

        for key, value in report.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", key, value)
            else:
                logger.info("  %s: %s", key, value)

        # Save state
        paper_trader.save_state()
        monitor.save_state()

        logger.info("\nPaper trading complete. State saved to %s", state_dir)

    except KeyboardInterrupt:
        logger.info("\nPaper trading interrupted by user")

        report = paper_trader.get_performance_report()
        logger.info("\nFinal Report:")

        for key, value in report.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", key, value)
            else:
                logger.info("  %s: %s", key, value)

        paper_trader.save_state()
        monitor.save_state()


def main() -> None:
    """Main entry point for live trading CLI."""
    parser = argparse.ArgumentParser(
        description="Run paper/live trading for a Crypto EA"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="btcusdt",
        help="Trading pair (default: btcusdt)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/binance/m1",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="data/state",
        help="Directory for state persistence",
    )
    parser.add_argument(
        "--use-telegram",
        action="store_true",
        help="Enable Telegram alerts",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=None,
        help="Telegram bot token",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=None,
        help="Telegram chat ID",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    import asyncio

    asyncio.run(
        run_paper_trading(
            symbol=args.symbol,
            initial_capital=args.initial_capital,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
            use_telegram=args.use_telegram,
            telegram_token=args.telegram_token,
            telegram_chat_id=args.telegram_chat_id,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
