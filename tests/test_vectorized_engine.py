"""Regression tests for the corrected vectorized backtest engine."""

from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.run_backtest import VectorizedBacktestConfig, VectorizedBacktestEngine


def test_engine_enters_on_direct_signal_flip() -> None:
    """A -1 to +1 signal flip should create a fresh long entry event."""
    index = pd.date_range("2024-01-01", periods=7, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 99.0, 101.0, 103.0, 104.0],
            "high": [101.0, 101.0, 101.0, 102.0, 104.0, 105.0, 106.0],
            "low": [99.0, 99.0, 98.0, 98.0, 100.0, 102.0, 103.0],
            "close": [100.0, 99.0, 99.0, 101.0, 103.0, 104.0, 105.0],
            "signal": [0, -1, -1, 1, 0, 0, 0],
            "stop_loss": [np.nan, 102.0, 102.0, 99.0, np.nan, np.nan, np.nan],
            "take_profit": [np.nan, 95.0, 95.0, 106.0, np.nan, np.nan, np.nan],
        },
        index=index,
    )

    engine = VectorizedBacktestEngine(
        data,
        VectorizedBacktestConfig(
            initial_balance=100000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            risk_per_trade=0.01,
            min_holding_bars=1,
            max_holding_bars=2,
            max_position_pct=1.0,
        ),
        "TEST",
    )

    result = engine.run()

    assert [trade["side"] for trade in result["trades"]] == ["SHORT", "LONG"]
