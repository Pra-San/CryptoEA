"""Walk-forward analysis module for robust strategy validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.optimizer import ParameterOptimizer
from data.loader import load_csv
from data.preprocessor import DataPreprocessor
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    train_months: int = 12
    test_months: int = 3
    overlap_months: int = 1
    min_train_days: int = 252
    metrics: tuple[str, ...] = (
        "calmar_ratio",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate",
        "total_trades",
        "profit_factor",
    )


class WalkForwardAnalyzer:
    """Sliding window walk-forward analysis with out-of-sample validation.

    Splits historical data into overlapping training and testing windows,
    optimizes parameters on training data, and evaluates on out-of-sample
    test data. Provides robustness metrics across all windows.
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
    ) -> None:
        self.config = config or WalkForwardConfig()

    def run(
        self,
        symbol: str,
        strategy: BaseStrategy,
        data_dir: str | Path = "data/binance/m1",
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Execute walk-forward analysis across all windows.

        Args:
            symbol: Trading pair (e.g., 'btcusdt').
            strategy: Strategy instance implementing BaseStrategy.
            data_dir: Directory containing CSV files.
            start_date: Optional start date filter (YYYY-MM-DD).
            end_date: Optional end date filter (YYYY-MM-DD).
            initial_capital: Starting capital for backtest.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing window results, aggregate statistics,
            and parameter importance rankings.
        """
        logger.info("Starting walk-forward analysis for %s", symbol)

        csv_path = Path(data_dir) / f"{symbol}_1m_spot.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        raw_df = load_csv(str(csv_path), start_date=start_date, end_date=end_date)
        preprocessor = DataPreprocessor()
        df = preprocessor.process(raw_df)

        if df.empty:
            raise ValueError("No data available after filtering")

        windows = self._generate_windows(df.index)

        if not windows:
            raise ValueError(
                "Insufficient data for walk-forward analysis. "
                f"Need at least {self.config.train_months + self.config.test_months} months"
            )

        all_results: list[dict[str, Any]] = []
        all_params: list[dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if verbose:
                logger.info(
                    "Window %d/%d: Train [%s -> %s], Test [%s -> %s]",
                    i + 1,
                    len(windows),
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )

            train_df = df.loc[train_start:train_end]
            test_df = df.loc[test_start:test_end]

            if len(train_df) < self.config.min_train_days:
                logger.warning(
                    "Skipping window %d: insufficient training data (%d days)",
                    i + 1,
                    len(train_df),
                )
                continue

            optimizer = ParameterOptimizer()
            optimized_params = optimizer.calibrate(
                train_df=train_df,
                strategy=strategy,
                initial_capital=initial_capital,
            )

            all_params.append(optimized_params)

            engine = BacktestEngine(
                data=test_df,
                strategy=strategy,
                initial_capital=initial_capital,
                params=optimized_params,
            )

            trade_log, equity_curve = engine.run()
            metrics = BacktestMetrics(trade_log, equity_curve)
            metrics_dict = metrics.calculate_all()

            window_result = {
                "window_index": i + 1,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "optimized_params": optimized_params,
                "metrics": metrics_dict,
            }

            all_results.append(window_result)

            if verbose:
                calmar = metrics_dict.get("calmar_ratio", 0)
                max_dd = metrics_dict.get("max_drawdown_pct", 0)
                logger.info(
                    "  Out-of-sample Calmar: %.2f, Max DD: %.2f%%",
                    calmar,
                    max_dd,
                )

        aggregate = self._aggregate_results(all_results)

        return {
            "windows": all_results,
            "aggregate": aggregate,
            "total_windows": len(all_results),
            "symbol": symbol,
        }

    def _generate_windows(
        self,
        index: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate overlapping train/test windows.

        Args:
            index: DatetimeIndex of the dataset.

        Returns:
            List of tuples (train_start, train_end, test_start, test_end).
        """
        timestamps = sorted(index)
        min_ts = timestamps[0]
        max_ts = timestamps[-1]

        windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

        current_start = min_ts

        while current_start <= max_ts:
            train_end = current_start + pd.Timedelta(
                months=self.config.train_months
            )
            test_start = train_end + pd.Timedelta(
                months=self.config.overlap_months
            )
            test_end = test_start + pd.Timedelta(
                months=self.config.test_months
            )

            if test_end > max_ts:
                break

            windows.append(
                (current_start, train_end, test_start, test_end)
            )

            current_start += pd.Timedelta(
                months=self.config.train_months - self.config.overlap_months
            )

        return windows

    def _aggregate_results(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate metrics across all windows.

        Args:
            results: List of per-window result dictionaries.

        Returns:
            Dictionary of aggregated statistics.
        """
        if not results:
            return {}

        metric_keys = self.config.metrics
        aggregated: dict[str, list[float]] = {k: [] for k in metric_keys}

        for r in results:
            m = r.get("metrics", {})
            for k in metric_keys:
                val = m.get(k)
                if val is not None:
                    aggregated[k].append(float(val))

        summary: dict[str, Any] = {}

        for k, values in aggregated.items():
            if values:
                summary[f"{k}_mean"] = float(np.mean(values))
                summary[f"{k}_std"] = float(np.std(values))
                summary[f"{k}_min"] = float(np.min(values))
                summary[f"{k}_max"] = float(np.max(values))

        profitable_windows = sum(
            1
            for r in results
            if r.get("metrics", {}).get("total_return_pct", 0) > 0
        )
        summary["out_of_sample_win_rate"] = (
            profitable_windows / len(results)
        )

        return summary
