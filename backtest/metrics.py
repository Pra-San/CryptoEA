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
    gross_profit: float = 0.0
    gross_loss: float = 0.0

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
    median_pnl: float = 0.0
    expectancy: float = 0.0
    avg_trade_return: float = 0.0
    median_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    payoff_ratio: float = 0.0
    trades_per_day: float = 0.0
    avg_holding_bars: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    exposure_pct: float = 0.0

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

    def __init__(
        self,
        trade_log: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        """Initialize the metrics calculator.

        Args:
            trade_log: Optional legacy trade log DataFrame.
            equity_curve: Optional legacy equity curve Series.
            risk_free_rate: Annual risk-free rate (0 for crypto).
        """
        if isinstance(trade_log, (int, float)) and equity_curve is None:
            risk_free_rate = float(trade_log)
            trade_log = None

        self.trade_log = trade_log
        self.equity_curve = equity_curve
        self.risk_free_rate = risk_free_rate

    def calculate_all(self) -> Dict[str, float]:
        """Calculate metrics for the legacy DataFrame/Series API."""
        trade_log = self.trade_log
        equity = self.equity_curve

        if trade_log is None or trade_log.empty:
            return self._metrics_to_dict(PerformanceMetrics())

        pnl_col = "pnl" if "pnl" in trade_log.columns else None
        return_col = (
            "pnl_pct"
            if "pnl_pct" in trade_log.columns
            else "return_pct"
            if "return_pct" in trade_log.columns
            else None
        )

        pnls = trade_log[pnl_col].dropna().astype(float) if pnl_col else pd.Series(dtype=float)
        returns = trade_log[return_col].dropna().astype(float) if return_col else pd.Series(dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        m = PerformanceMetrics()
        m.total_trades = int(len(trade_log))
        m.winning_trades = int(len(wins))
        m.losing_trades = int(len(losses))
        m.win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
        m.avg_win = float(wins.mean()) if len(wins) else 0.0
        m.avg_loss = float(losses.mean()) if len(losses) else 0.0
        m.avg_pnl = float(pnls.mean()) if len(pnls) else 0.0
        m.median_pnl = float(pnls.median()) if len(pnls) else 0.0
        m.best_trade = float(pnls.max()) if len(pnls) else 0.0
        m.worst_trade = float(pnls.min()) if len(pnls) else 0.0
        m.gross_profit = float(wins.sum()) if len(wins) else 0.0
        m.gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        m.profit_factor = (
            m.gross_profit / m.gross_loss
            if m.gross_loss > 0
            else float("inf")
            if m.gross_profit > 0
            else 0.0
        )
        m.expectancy = m.avg_win * m.win_rate + m.avg_loss * (1 - m.win_rate)
        m.avg_trade_return = float(returns.mean()) if len(returns) else 0.0
        m.median_trade_return = float(returns.median()) if len(returns) else 0.0
        m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss else 0.0
        m.max_consecutive_wins = self._max_consecutive(pnls.tolist(), lambda x: x > 0)
        m.max_consecutive_losses = self._max_consecutive(pnls.tolist(), lambda x: x <= 0)

        if equity is not None and len(equity) > 1:
            equity = equity.astype(float)
            initial = float(equity.iloc[0])
            final = float(equity.iloc[-1])
            m.total_return = (final - initial) / initial if initial else 0.0
            eq_returns = equity.pct_change().dropna()
            if len(eq_returns) > 1 and eq_returns.std() > 0:
                m.sharpe_ratio = float((eq_returns.mean() / eq_returns.std()) * np.sqrt(252 * 24))
            dd = self._calculate_drawdown(equity)
            m.max_drawdown = float(dd.min()) if len(dd) else 0.0
            m.max_drawdown_duration = self._max_drawdown_duration(dd)
            m.calmar_ratio = abs(m.total_return / m.max_drawdown) if m.max_drawdown else 0.0

        return self._metrics_to_dict(m)

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
        metrics.median_pnl = np.median(pnls) if pnls else 0
        metrics.best_trade = max(pnls) if pnls else 0
        metrics.worst_trade = min(pnls) if pnls else 0
        metrics.gross_profit = sum(wins)
        metrics.gross_loss = abs(sum(losses))
        metrics.expectancy = metrics.avg_win * metrics.win_rate + metrics.avg_loss * (1 - metrics.win_rate)
        metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else 0

        if pnl_pcts:
            metrics.avg_trade_return = float(np.mean(pnl_pcts))
            metrics.median_trade_return = float(np.median(pnl_pcts))

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

            if isinstance(equity.index, pd.DatetimeIndex):
                days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1)
                metrics.trades_per_day = metrics.total_trades / days
                metrics.exposure_pct = min(
                    1.0,
                    sum(t.holding_bars for t in completed_trades) / len(equity),
                )

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
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf') if metrics.gross_profit > 0 else 0

        # --- Drawdown Analysis ---
        if equity is not None:
            dd = self._calculate_drawdown(equity)
            metrics.max_drawdown = dd.min() if len(dd) > 0 else 0
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
            f"Max DD: {m.max_drawdown:.4%} | Profit Factor: {m.profit_factor:.2f}",
            f"Avg PnL: {m.avg_pnl:,.2f} | Expectancy: {m.expectancy:,.2f}",
            f"Trades/Day: {m.trades_per_day:.2f} | Exposure: {m.exposure_pct:.1%}",
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

    def _metrics_to_dict(self, m: PerformanceMetrics) -> Dict[str, float]:
        """Convert metrics to a flat dictionary for JSON/tests."""
        return {
            "total_return": float(m.total_return),
            "total_return_pct": float(m.total_return * 100),
            "cagr": float(m.cagr),
            "cagr_pct": float(m.cagr * 100),
            "sharpe_ratio": float(m.sharpe_ratio),
            "sortino_ratio": float(m.sortino_ratio),
            "calmar_ratio": float(m.calmar_ratio),
            "profit_factor": float(m.profit_factor),
            "total_trades": int(m.total_trades),
            "winning_trades": int(m.winning_trades),
            "losing_trades": int(m.losing_trades),
            "gross_profit": float(m.gross_profit),
            "gross_loss": float(m.gross_loss),
            "max_drawdown": float(m.max_drawdown),
            "max_drawdown_pct": float(abs(m.max_drawdown) * 100),
            "max_drawdown_duration": int(m.max_drawdown_duration),
            "avg_drawdown": float(m.avg_drawdown),
            "recovery_factor": float(m.recovery_factor),
            "ulcer_index": float(m.ulcer_index),
            "win_rate": float(m.win_rate),
            "avg_win": float(m.avg_win),
            "avg_loss": float(m.avg_loss),
            "avg_pnl": float(m.avg_pnl),
            "median_pnl": float(m.median_pnl),
            "expectancy": float(m.expectancy),
            "avg_trade_return": float(m.avg_trade_return),
            "median_trade_return": float(m.median_trade_return),
            "best_trade": float(m.best_trade),
            "worst_trade": float(m.worst_trade),
            "payoff_ratio": float(m.payoff_ratio),
            "trades_per_day": float(m.trades_per_day),
            "avg_holding_bars": int(m.avg_holding_bars),
            "max_consecutive_wins": int(m.max_consecutive_wins),
            "max_consecutive_losses": int(m.max_consecutive_losses),
            "exposure_pct": float(m.exposure_pct),
            "var_95": float(m.var_95),
            "cvar_95": float(m.cvar_95),
            "beta": float(m.beta),
            "alpha": float(m.alpha),
            "total_fees": float(m.total_fees),
            "total_slippage": float(m.total_slippage),
            "best_month": float(m.best_month),
            "worst_month": float(m.worst_month),
            "profitable_months": int(m.profitable_months),
            "total_months": int(m.total_months),
        }

    def to_dict(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Public serializer for PerformanceMetrics."""
        return self._metrics_to_dict(metrics)

    def format_report(self, metrics: PerformanceMetrics) -> str:
        """Return a formatted metrics report as text."""
        lines = [
            "",
            "=" * 60,
            "  BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            metrics.summary,
            "",
            "  Trade Statistics:",
            f"    Total Trades:       {metrics.total_trades}",
            f"    Winning Trades:     {metrics.winning_trades} ({metrics.win_rate:.1%})",
            f"    Losing Trades:      {metrics.losing_trades}",
            f"    Win Rate:           {metrics.win_rate:.1%}",
            f"    Avg Win:            {metrics.avg_win:,.2f}",
            f"    Avg Loss:           {metrics.avg_loss:,.2f}",
            f"    Median PnL:         {metrics.median_pnl:,.2f}",
            f"    Best Trade:         {metrics.best_trade:,.2f}",
            f"    Worst Trade:        {metrics.worst_trade:,.2f}",
            f"    Profit Factor:      {metrics.profit_factor:.2f}",
            f"    Payoff Ratio:       {metrics.payoff_ratio:.2f}",
            f"    Expectancy:         {metrics.expectancy:,.2f}",
            f"    Avg Trade Return:   {metrics.avg_trade_return:.4%}",
            f"    Trades/Day:         {metrics.trades_per_day:.2f}",
            f"    Avg Holding Bars:   {metrics.avg_holding_bars}",
            f"    Exposure:           {metrics.exposure_pct:.1%}",
            "",
            "  Performance Ratios:",
            f"    Total Return:       {metrics.total_return:.2%}",
            f"    CAGR:               {metrics.cagr:.2%}",
            f"    Sharpe Ratio:       {metrics.sharpe_ratio:.2f}",
            f"    Sortino Ratio:      {metrics.sortino_ratio:.2f}",
            f"    Calmar Ratio:       {metrics.calmar_ratio:.2f}",
            "",
            "  Drawdown Analysis:",
            f"    Max Drawdown:       {metrics.max_drawdown:.4%}",
            f"    Avg Drawdown:       {metrics.avg_drawdown:.4%}",
            f"    Max DD Duration:    {metrics.max_drawdown_duration} bars",
            f"    Recovery Factor:    {metrics.recovery_factor:.2f}",
            f"    Ulcer Index:        {metrics.ulcer_index:.6f}",
            "",
            "  Risk Metrics:",
            f"    VaR (95%):          {metrics.var_95:.4%}",
            f"    CVaR (95%):         {metrics.cvar_95:.4%}",
            f"    Beta:               {metrics.beta:.4f}",
            f"    Alpha:              {metrics.alpha:.4%}",
            "",
            "  Execution Quality:",
            f"    Total Fees:         {metrics.total_fees:,.2f}",
            f"    Total Slippage:     {metrics.total_slippage:,.2f}",
        ]

        if metrics.total_months > 0:
            lines.extend([
                f"    Best Month:         {metrics.best_month:.2%}",
                f"    Worst Month:        {metrics.worst_month:.2%}",
                f"    Profitable Months:  {metrics.profitable_months}/{metrics.total_months}",
            ])

        lines.extend(["", "=" * 60, ""])
        return "\n".join(lines)

    def print_report(self, metrics: PerformanceMetrics) -> None:
        """Print a formatted metrics report.

        Args:
            metrics: PerformanceMetrics to display.
        """
        print(self.format_report(metrics))
