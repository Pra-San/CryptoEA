"""Tests for backtest metrics module."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import BacktestMetrics


class TestBacktestMetrics:
    """Tests for BacktestMetrics class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.trade_log = pd.DataFrame({
            "entry_time": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "exit_time": pd.date_range("2024-01-01 02:00:00", periods=10, freq="1h"),
            "pnl": [100.0, -50.0, 150.0, -30.0, 200.0, -80.0, 120.0, -20.0, 180.0, -40.0],
            "return_pct": [0.01, -0.005, 0.015, -0.003, 0.02, -0.008, 0.012, -0.002, 0.018, -0.004],
        })

        self.equity_curve = pd.Series(
            [10000.0, 10100.0, 10050.0, 10200.0, 10170.0, 10370.0, 10290.0, 10410.0, 10390.0, 10570.0, 10530.0]
        )

        self.metrics = BacktestMetrics(self.trade_log, self.equity_curve)

    def test_calculate_all_returns_dict(self) -> None:
        """Test that calculate_all returns a dictionary."""
        result = self.metrics.calculate_all()

        assert isinstance(result, dict)

    def test_calculate_returns_total_trades(self) -> None:
        """Test total trades calculation."""
        result = self.metrics.calculate_all()

        assert result["total_trades"] == 10

    def test_calculate_returns_win_rate(self) -> None:
        """Test win rate calculation."""
        result = self.metrics.calculate_all()

        wins = sum(1 for p in self.trade_log["pnl"] if p > 0)
        expected_win_rate = wins / len(self.trade_log)

        assert abs(result["win_rate"] - expected_win_rate) < 0.01

    def test_calculate_returns_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        result = self.metrics.calculate_all()

        assert result["max_drawdown_pct"] >= 0

    def test_calculate_returns_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        result = self.metrics.calculate_all()

        assert isinstance(result["sharpe_ratio"], float)

    def test_calculate_returns_profit_factor(self) -> None:
        """Test profit factor calculation."""
        result = self.metrics.calculate_all()

        assert result["profit_factor"] >= 0

    def test_calculate_returns_calmar_ratio(self) -> None:
        """Test Calmar ratio calculation."""
        result = self.metrics.calculate_all()

        assert isinstance(result["calmar_ratio"], float)

    def test_calculate_returns_empty_trade_log(self) -> None:
        """Test handling of empty trade log."""
        empty_metrics = BacktestMetrics(pd.DataFrame(), pd.Series())
        result = empty_metrics.calculate_all()

        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0
