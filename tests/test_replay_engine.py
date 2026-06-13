from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from deployment.replay_engine import replay_deployment
from scripts.explore_trailing_highwin_edge import simulate_trades


class DummyRuntime:
    symbol = "SOLUSDT"
    balance = 100_000.0
    fee_rate = 0.0
    slippage_rate = 0.0
    risk_per_trade = 0.01
    max_position_pct = 1.0
    params = {
        "break_even_atr": 1.0,
        "exit_on_flat_signal": True,
        "max_holding_bars": 10,
        "min_holding_bars": 1,
        "move_stop_after_partial": False,
        "partial_exit_atr": 1.0,
        "partial_exit_fraction": 0.5,
        "trail_atr_mult": 2.0,
        "use_break_even": False,
        "use_partial_exit": False,
        "use_trailing_stop": False,
    }

    def args(self, balance: float | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            balance=self.balance if balance is None else balance,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            risk_per_trade=self.risk_per_trade,
            max_position_pct=self.max_position_pct,
        )


def test_replay_deployment_matches_research_simulator_on_signal_lifecycle() -> None:
    featured = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0, 104.0, 105.0, 104.0, 102.0],
            "high": [101.0, 102.0, 104.0, 105.0, 105.0, 106.0, 105.0, 103.0],
            "low": [99.0, 99.0, 100.0, 102.0, 103.0, 104.0, 101.0, 100.0],
            "close": [100.0, 101.0, 103.0, 104.0, 104.0, 104.0, 102.0, 101.0],
            "signal": [0, 1, 1, 1, 0, -1, -1, -1],
            "stop_loss": [np.nan, 98.0, 98.0, 98.0, np.nan, 108.0, 108.0, 108.0],
            "take_profit": [np.nan] * 8,
            "atr_14": [2.0] * 8,
        },
        index=pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
    )
    runtime = DummyRuntime()

    expected = simulate_trades(
        featured,
        runtime.symbol,
        runtime.args(),
        runtime.params,
        runtime.slippage_rate,
    )
    actual = replay_deployment(featured, runtime)

    assert len(actual) == len(expected) == 2
    for replayed, simulated in zip(actual, expected):
        assert replayed.entry_time == simulated.entry_time
        assert replayed.exit_time == simulated.exit_time
        assert replayed.side == simulated.side
        assert replayed.exit_reason == simulated.exit_reason
        assert replayed.entry_price == simulated.entry_price
        assert replayed.exit_price == simulated.exit_price
        assert replayed.quantity == simulated.quantity
        assert replayed.pnl == simulated.pnl
        assert replayed.r_multiple == simulated.r_multiple
