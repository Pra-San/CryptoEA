"""Monte Carlo simulation for strategy robustness testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 1000
    confidence_levels: tuple[float, ...] = (0.90, 0.95, 0.99)
    resample_method: str = "block"  # 'block' or 'bootstrap'
    block_size: int = 100  # For block bootstrap


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy robustness testing.

    Generates thousands of possible equity curves by resampling trade
    returns with bootstrapping or block resampling. Provides confidence
    intervals for key metrics and probability of ruin estimates.
    """

    def __init__(
        self,
        config: MonteCarloConfig | None = None,
    ) -> None:
        self.config = config or MonteCarloConfig()

    def run(
        self,
        trade_log: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Execute Monte Carlo simulation on trade log.

        Args:
            trade_log: DataFrame of executed trades.
            strategy: Strategy instance (used for parameter sampling).
            initial_capital: Starting capital.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing simulation results, confidence intervals,
            and probability estimates.
        """
        if trade_log.empty:
            raise ValueError("Trade log is empty, cannot run simulation")

        logger.info(
            "Running Monte Carlo simulation (%d simulations)",
            self.config.num_simulations,
        )

        all_final_equities: list[float] = []
        all_max_drawdowns: list[float] = []
        all_sharpe_ratios: list[float] = []

        for i in range(self.config.num_simulated):
            # Resample trades
            resampled_trades = self._resample(trade_log)

            # Simulate equity curve
            equity = self._simulate_equity(resampled_trades, initial_capital)
            final_equity = equity.iloc[-1]

            # Calculate metrics for this simulation
            max_dd = self._calculate_max_drawdown(equity)
            sharpe = self._calculate_sharpe(equity)

            all_final_equities.append(float(final_equity))
            all_max_drawdowns.append(float(max_dd))
            all_sharpe_ratios.append(float(sharpe))

            if verbose and (i + 1) % 100 == 0:
                logger.info("Completed %d/%d simulations", i + 1, self.config.num_simulations)

        # Calculate confidence intervals
        ci_results = self._calculate_confidence_intervals(
            all_final_equities,
            all_max_drawdowns,
            all_sharpe_ratios,
        )

        # Probability estimates
        prob_ruin = self._calculate_probability_of_ruin(
            all_final_equities, initial_capital
        )
        prob_profit = self._calculate_probability_of_profit(
            all_final_equities, initial_capital
        )

        return {
            "num_simulations": self.config.num_simulations,
            "final_equity_stats": {
                "mean": float(np.mean(all_final_equities)),
                "median": float(np.median(all_final_equities)),
                "std": float(np.std(all_final_equities)),
                "min": float(np.min(all_final_equities)),
                "max": float(np.max(all_final_equities)),
            },
            "max_drawdown_stats": {
                "mean": float(np.mean(all_max_drawdowns)),
                "median": float(np.median(all_max_drawdowns)),
                "std": float(np.std(all_max_drawdowns)),
                "p95": float(np.percentile(all_max_drawdowns, 95)),
            },
            "sharpe_ratio_stats": {
                "mean": float(np.mean(all_sharpe_ratios)),
                "median": float(np.median(all_sharpe_ratios)),
                "std": float(np.std(all_sharpe_ratios)),
            },
            "confidence_intervals": ci_results,
            "probability_of_ruin": prob_ruin,
            "probability_of_profit": prob_profit,
            "simulated_equities": all_final_equities,
        }

    def _resample(
        self,
        trade_log: pd.DataFrame,
    ) -> pd.DataFrame:
        """Resample trades using block bootstrap or standard bootstrap.

        Args:
            trade_log: Original trade log DataFrame.

        Returns:
            Resampled trade log DataFrame.
        """
        n = len(trade_log)

        if self.config.resample_method == "block":
            return self._block_bootstrap(trade_log, n)
        else:
            return self._standard_bootstrap(trade_log, n)

    def _standard_bootstrap(
        self,
        trade_log: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Standard bootstrap resampling.

        Args:
            trade_log: Original trade log.
            n: Number of trades.

        Returns:
            Resampled trade log.
        """
        indices = np.random.choice(n, size=n, replace=True)
        return trade_log.iloc[indices].reset_index(drop=True)

    def _block_bootstrap(
        self,
        trade_log: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Block bootstrap resampling for time-series data.

        Preserves local correlation structure by resampling blocks of trades.

        Args:
            trade_log: Original trade log.
            n: Number of trades.

        Returns:
            Resampled trade log.
        """
        block_size = min(self.config.block_size, n)
        resampled_trades: list[pd.DataFrame] = []

        current_pos = 0

        while current_pos < n:
            block_end = min(current_pos + block_size, n)
            block = trade_log.iloc[current_pos:block_end]

            if len(block) > 0:
                resampled_trades.append(block)

            current_pos = np.random.randint(0, max(1, n - block_size + 1))

        if resampled_trades:
            return pd.concat(resampled_trades, ignore_index=True)
        return trade_log.head(1)

    def _simulate_equity(
        self,
        trades: pd.DataFrame,
        initial_capital: float,
    ) -> pd.Series:
        """Simulate equity curve from resampled trades.

        Args:
            trades: Resampled trade DataFrame.
            initial_capital: Starting capital.

        Returns:
            Series of equity values over time.
        """
        equity = [initial_capital]

        for _, trade in trades.iterrows():
            pnl = trade.get("pnl", 0.0)
            if isinstance(pnl, (int, float)):
                equity.append(equity[-1] + float(pnl))
            else:
                equity.append(equity[-1])

        return pd.Series(equity, index=range(len(equity)))

    def _calculate_max_drawdown(
        self,
        equity: pd.Series,
    ) -> float:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity: Series of equity values.

        Returns:
            Maximum drawdown as a percentage.
        """
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(abs(drawdown.min()))

    def _calculate_sharpe(
        self,
        equity: pd.Series,
    ) -> float:
        """Calculate annualized Sharpe ratio from equity curve.

        Args:
            equity: Series of equity values.

        Returns:
            Annualized Sharpe ratio.
        """
        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        annualized_sharpe = (mean_return / std_return) * np.sqrt(252 * 24 * 60)
        return float(annualized_sharpe)

    def _calculate_confidence_intervals(
        self,
        final_equities: list[float],
        max_drawdowns: list[float],
        sharpe_ratios: list[float],
    ) -> dict[str, Any]:
        """Calculate confidence intervals for key metrics.

        Args:
            final_equities: List of final equity values from simulations.
            max_drawdowns: List of maximum drawdowns from simulations.
            sharpe_ratios: List of Sharpe ratios from simulations.

        Returns:
            Dictionary of confidence intervals for each metric.
        """
        ci_results: dict[str, Any] = {}

        for name, values in [
            ("final_equity", final_equities),
            ("max_drawdown", max_drawdowns),
            ("sharpe_ratio", sharpe_ratios),
        ]:
            ci_results[name] = {}

            for level in self.config.confidence_levels:
                alpha = 1 - level
                lower = float(np.percentile(values, alpha / 2 * 100))
                upper = float(np.percentile(values, (1 - alpha / 2) * 100))
                ci_results[name][f"{level:.0%}"] = {
                    "lower": lower,
                    "upper": upper,
                }

        return ci_results

    def _calculate_probability_of_ruin(
        self,
        final_equities: list[float],
        initial_capital: float,
        ruin_threshold: float = 0.5,
    ) -> float:
        """Calculate probability of ruin (equity falls below threshold).

        Args:
            final_equities: List of final equity values.
            initial_capital: Starting capital.
            ruin_threshold: Fraction of initial capital considered ruin.

        Returns:
            Probability of ruin (0.0 to 1.0).
        """
        ruin_level = initial_capital * ruin_threshold
        ruin_count = sum(1 for e in final_equities if e < ruin_level)
        return float(ruin_count / len(final_equities))

    def _calculate_probability_of_profit(
        self,
        final_equities: list[float],
        initial_capital: float,
    ) -> float:
        """Calculate probability of ending in profit.

        Args:
            final_equities: List of final equity values.
            initial_capital: Starting capital.

        Returns:
            Probability of profit (0.0 to 1.0).
        """
        profit_count = sum(1 for e in final_equities if e > initial_capital)
        return float(profit_count / len(final_equities))
