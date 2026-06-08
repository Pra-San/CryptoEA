"""Performance metrics calculation for CryptoEA backtests.

Computes comprehensive performance statistics including:
- Core performance (Sharpe, Sortino, Calmar, CAGR)
- Drawdown analysis (max DD, recovery factor, Ulcer Index)
- Trade statistics (win rate, profit factor, expectancy)
- Risk metrics (VaR, CVaR, beta, alpha)
- Execution quality (fees, slippage, monthly returns)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from backtest.engine import BacktestResult, Trade

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a backtest.

    Attributes:
        Core performance metrics
        drawdown_metrics: Drawdown analysis results
        trade_metrics: Individual trade statistics
        risk_metrics: Risk-adjusted measures
        execution_metrics: Trading quality measures
    """

    # Core Performance
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Drawdown Analysis
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0

    # Trade Statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0
    expectancy: float = 0.0
    avg_holding_bars: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Risk Metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    max_open_positions: int = 0

    # Execution Quality
    total_fees: float = 0.0
    total_slippage: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    profitable_months: int = 0
    total_months: int = 0

    # Summary
    summary: str = ""


class BacktestMetrics:
    """Calculate comprehensive performance metrics from backtest results.

    Usage:
        metrics = BacktestMetrics()
        result = metrics.calculate(backtest_result, data)
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (0 for crypto).
        """
        self.risk_free_rate = risk_free_rate

    def calculate(
        self, result: BacktestResult, data: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """Calculate all performance metrics from backtest results.

        Args:
            result: BacktestResult from running the engine.
            data: Optional DataFrame for additional calculations.

        Returns:
            PerformanceMetrics with all computed values.
        """
        metrics = PerformanceMetrics()

        trades = result.trades
        completed_trades = [t for t in trades if t.exit_time is not None]

        if not completed_trades:
            metrics.summary = "No completed trades."
            return metrics

        metrics.total_trades = len(completed_trades)

        # --- Trade Statistics ---
        pnls = [t.pnl for t in completed_trades if t.pnl is not None]
        pnl_pcts = [t.pnl_pct for t in completed_trades if t.pnl_pct is not None]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(pnls) if pnls else 0

        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.avg_pnl = np.mean(pnls) if pnls else 0
        metrics.expectancy = metrics.avg_pnl * metrics.win_rate + metrics.avg_loss * (1 - metrics.win_rate)

        # Average holding time
        holding_times = [t.holding_bars for t in completed_trades]
        metrics.avg_holding_bars = int(np.mean(holding_times)) if holding_times else 0

        # Consecutive wins/losses
        metrics.max_consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
        metrics.max_consecutive_losses = self._max_consecutive(pnls, lambda x: x <= 0)

        # --- Core Performance ---
        equity = result.equity_curve
        if equity is not None and len(equity) > 1:
            # Total return
            initial = result.config.initial_balance if result.config else 10000
            final = equity.iloc[-1]
            metrics.total_return = (final - initial) / initial

            # CAGR
            start_date = result.metadata.get("start_date", "")
            end_date = result.metadata.get("end_date", "")
            if start_date and end_date:
                try:
                    from datetime import datetime
                    start_dt = pd.Timestamp(start_date)
                    end_dt = pd.Timestamp(end_date)
                    years = (end_dt - start_dt).days / 365.25
                    if years > 0:
                        metrics.cagr = (final / initial) ** (1 / years) - 1
                except Exception:
                    metrics.cagr = metrics.total_return

            # Returns series for ratio calculations
            returns = equity.pct_change().dropna()

            # Sharpe Ratio (annualized, risk-free = 0)
            if len(returns) > 1 and returns.std() > 0:
                metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # 24 1h bars per day

            # Sortino Ratio (downside deviation)
            downside = returns[returns < 0]
            if len(downside) > 1 and downside.std() > 0:
                metrics.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252 * 24)

            # Calmar Ratio (CAGR / Max Drawdown)
            metrics.calmar_ratio = metrics.cagr / abs(self._max_drawdown_from_equity(equity)) if self._max_drawdown_from_equity(equity) != 0 else 0

            # Profit Factor
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # --- Drawdown Analysis ---
        if equity is not None:
            dd = self._calculate_drawdown(equity)
            metrics.max_drawdown = dd.max() if len(dd) > 0 else 0
            metrics.max_drawdown_duration = self._max_drawdown_duration(dd)
            metrics.avg_drawdown = dd.mean() if len(dd) > 0 else 0
            metrics.recovery_factor = abs(metrics.total_return / metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
            metrics.ulcer_index = np.sqrt(np.mean(dd ** 2))

        # --- Risk Metrics ---
        if equity is not None:
            returns = equity.pct_change().dropna()
            if len(returns) > 0:
                # Value at Risk (95%)
                metrics.var_95 = np.percentile(returns, 5)

                # Expected Shortfall (CVaR)
                tail_returns = returns[returns <= metrics.var_95]
                metrics.cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else metrics.var_95

                # Beta vs benchmark
                if result.benchmark_curve is not None:
                    bench_returns = result.benchmark_curve.pct_change().dropna()
                    common_idx = equity.index.intersection(bench_returns.index)
                    if len(common_idx) > 10:
                        asset_ret = equity.loc[common_idx].pct_change().dropna()
                        bench_ret = bench_returns.loc[common_idx].dropna()
                        # Align lengths by truncating to minimum
                        min_len = min(len(asset_ret), len(bench_ret))
                        if min_len > 1:
                            asset_ret = asset_ret.iloc[:min_len]
                            bench_ret = bench_ret.iloc[:min_len]
                            cov_matrix = np.cov(asset_ret.values, bench_ret.values)
                            if cov_matrix[1, 1] > 0:
                                metrics.beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                                # Alpha (excess return)
                                market_return = bench_ret.mean() * 252 * 24
                                asset_return = returns.mean() * 252 * 24
                                metrics.alpha = metrics.total_return - (metrics.beta * market_return)

        # --- Execution Quality ---
        total_fees = sum(t.fees for t in trades)
        total_slippage = sum(t.slippage for t in trades)
        metrics.total_fees = total_fees
        metrics.total_slippage = total_slippage

        # Monthly returns
        if equity is not None:
            monthly = equity.resample("ME").last()
            monthly_returns = monthly.pct_change().dropna()
            if len(monthly_returns) > 0:
                metrics.best_month = monthly_returns.max()
                metrics.worst_month = monthly_returns.min()
                metrics.profitable_months = (monthly_returns > 0).sum()
                metrics.total_months = len(monthly_returns)

        # Generate summary
        metrics.summary = self._generate_summary(metrics)

        return metrics

    def _generate_summary(self, m: PerformanceMetrics) -> str:
        """Generate a human-readable summary of metrics.

        Args:
            m: PerformanceMetrics to summarize.

        Returns:
            Summary string.
        """
        if m.total_trades == 0:
            return "No trades executed."

        lines = [
            f"Total Trades: {m.total_trades} (Win Rate: {m.win_rate:.1%})",
            f"Total Return: {m.total_return:.1%} | CAGR: {m.cagr:.1%}",
            f"Sharpe: {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f} | Calmar: {m.calmar_ratio:.2f}",
            f"Max DD: {m.max_drawdown:.1%} | Profit Factor: {m.profit_factor:.2f}",
            f"Avg PnL: {m.avg_pnl:,.2f} | Expectancy: {m.expectancy:,.2f}",
            f"Max Consecutive Wins: {m.max_consecutive_wins} | Losses: {m.max_consecutive_losses}",
        ]

        if m.profitable_months > 0:
            lines.append(f"Monthly Win Rate: {m.profitable_months}/{m.total_months} months profitable")

        return "\n".join(lines)

    def _max_drawdown_from_equity(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity: Equity curve Series.

        Returns:
            Maximum drawdown as a fraction.
        """
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series.

        Args:
            equity: Equity curve Series.

        Returns:
            Drawdown series (negative values).
        """
        running_max = equity.cummax()
        return (equity - running_max) / running_max

    def _max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in bars.

        Args:
            drawdown: Drawdown series.

        Returns:
            Maximum duration in bars.
        """
        # Find periods where drawdown is below zero
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0

        # Find longest consecutive True sequence
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _max_consecutive(self, values: List[float], condition) -> int:
        """Find maximum consecutive values meeting a condition.

        Args:
            values: List of values to check.
            condition: Function that returns True/False.

        Returns:
            Maximum consecutive count.
        """
        max_count = 0
        current_count = 0

        for v in values:
            if condition(v):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def print_report(self, metrics: PerformanceMetrics) -> None:
        """Print a formatted metrics report.

        Args:
            metrics: PerformanceMetrics to display.
        """
        print("\n" + "=" * 60)
        print("  BACKTEST PERFORMANCE REPORT")
        print("=" * 60)
        print(f"\n{metrics.summary}")
        print(f"\n  Trade Statistics:")
        print(f"    Total Trades:     {metrics.total_trades}")
        print(f"    Winning Trades:   {metrics.winning_trades} ({metrics.win_rate:.1%})")
        print(f"    Losing Trades:    {metrics.losing_trades}")
        print(f"    Win Rate:         {metrics.win_rate:.1%}")
        print(f"    Avg Win:          {metrics.avg_win:,.2f}")
        print(f"    Avg Loss:         {metrics.avg_loss:,.2f}")
        print(f"    Profit Factor:    {metrics.profit_factor:.2f}")
        print(f"    Expectancy:       {metrics.expectancy:,.2f}")
        print(f"    Avg Holding Bars: {metrics.avg_holding_bars}")

        print(f"\n  Performance Ratios:")
        print(f"    Total Return:     {metrics.total_return:.1%}")
        print(f"    CAGR:             {metrics.cagr:.1%}")
        print(f"    Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        print(f"    Sortino Ratio:    {metrics.sortino_ratio:.2f}")
        print(f"    Calmar Ratio:     {metrics.calmar_ratio:.2f}")

        print(f"\n  Drawdown Analysis:")
        print(f"    Max Drawdown:     {metrics.max_drawdown:.1%}")
        print(f"    Max DD Duration:  {metrics.max_drawdown_duration} bars")
        print(f"    Recovery Factor:  {metrics.recovery_factor:.2f}")
        print(f"    Ulcer Index:      {metrics.ulcer_index:.2f}")

        print(f"\n  Risk Metrics:")
        print(f"    VaR (95%):        {metrics.var_95:.2%}")
        print(f"    CVaR (95%):       {metrics.cvar_95:.2%}")
        print(f"    Beta:             {metrics.beta:.2f}")
        print(f"    Alpha:            {metrics.alpha:.2%}")

        print(f"\n  Execution Quality:")
        print(f"    Total Fees:       {metrics.total_fees:,.2f}")
        print(f"    Total Slippage:   {metrics.total_slippage:,.2f}")
        if metrics.total_months > 0:
            print(f"    Best Month:       {metrics.best_month:.1%}")
            print(f"    Worst Month:      {metrics.worst_month:.1%}")
            print(f"    Profitable Months: {metrics.profitable_months}/{metrics.total_months}")

        print("\n" + "=" * 60 + "\n")
