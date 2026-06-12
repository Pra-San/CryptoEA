"""Shared strategy runtime for live/demo and shadow backtest processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from scripts.evaluate_prop_firm_constraints import load_candidate
from scripts.explore_high_winrate_edge import trades_per_week
from scripts.explore_trailing_highwin_edge import metrics_from_trades, simulate_trades
from scripts.explore_vwap_volume_profile_edge import apply_strategy, prepare_features
from scripts.run_backtest import vectorized_preprocess


def feature_tail_payload(featured: pd.DataFrame, rows: int = 3) -> dict[str, Any]:
    """Convert recent feature rows to JSON-stable diagnostics."""
    tail = featured.tail(rows)
    return {pd.Timestamp(index).isoformat(): values for index, values in tail.to_dict("index").items()}


@dataclass(frozen=True)
class CandidateRuntime:
    """Loaded research candidate and matching backtest arguments."""

    symbol: str
    timeframe: str
    strategy: str
    name: str
    params: dict[str, Any]
    fee_rate: float
    slippage_rate: float
    risk_per_trade: float
    max_position_pct: float
    balance: float

    @classmethod
    def from_summary(
        cls,
        summary_path: str | Path,
        section: str = "best",
        balance: float = 100000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.001,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 1.0,
    ) -> "CandidateRuntime":
        symbol, timeframe, strategy, name, params = load_candidate(Path(summary_path), section)
        if strategy != "vwap_volume_profile_momentum_search":
            raise ValueError(f"Unsupported deployment candidate strategy: {strategy}")
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            name=name,
            params=params,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            risk_per_trade=risk_per_trade,
            max_position_pct=max_position_pct,
            balance=balance,
        )

    def args(self, balance: float | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            balance=self.balance if balance is None else balance,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            risk_per_trade=self.risk_per_trade,
            max_position_pct=self.max_position_pct,
        )

    def feature_frame(self, raw_klines: pd.DataFrame) -> pd.DataFrame:
        base = vectorized_preprocess(raw_klines[["open", "high", "low", "close", "volume"]], self.timeframe)
        return apply_strategy(prepare_features(base, self.params, {}), self.params)

    def latest_signal(self, raw_klines: pd.DataFrame) -> dict[str, Any] | None:
        featured = self.feature_frame(raw_klines)
        if len(featured) < 3:
            return None
        current = featured.iloc[-1]
        previous = featured.iloc[-2]
        signal = int(current["signal"])
        previous_signal = int(previous["signal"])
        if signal == 0 or signal == previous_signal:
            return {
                "action": "HOLD",
                "signal": signal,
                "signal_bar": featured.index[-1].isoformat(),
                "previous_signal": previous_signal,
                "featured_tail": feature_tail_payload(featured),
            }
        return {
            "action": "ENTER_LONG" if signal > 0 else "ENTER_SHORT",
            "signal": signal,
            "signal_bar": featured.index[-1].isoformat(),
            "previous_signal": previous_signal,
            "stop_loss": float(current["stop_loss"]),
            "take_profit": float(current["take_profit"]),
            "entry_atr": float(current["atr_14"]),
            "close": float(current["close"]),
            "open": float(current["open"]),
            "high": float(current["high"]),
            "low": float(current["low"]),
            "featured_tail": feature_tail_payload(featured),
        }

    def backtest_snapshot(self, raw_klines: pd.DataFrame, balance: float | None = None) -> tuple[list[Any], dict[str, Any]]:
        featured = self.feature_frame(raw_klines)
        args = self.args(balance=balance)
        trades = simulate_trades(featured, self.symbol, args, self.params, self.slippage_rate)
        metrics = metrics_from_trades(trades, featured.index, args.balance)
        metrics["trades_per_week"] = trades_per_week(metrics)
        return trades, metrics
