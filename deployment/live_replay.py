"""Replay the live/demo deployment lifecycle from historical candles."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from deployment.binance_futures import SymbolRules
from deployment.brokers import DemoBroker, RuntimePosition
from deployment.state import SQLiteStateStore
from deployment.strategy_runtime import CandidateRuntime
from scripts.deploy_sol_momentum import latest_signal_from_featured, manage_position
from scripts.explore_high_winrate_edge import trades_per_week
from scripts.explore_trailing_highwin_edge import metrics_from_trades


@dataclass
class LivePathReplayConfig:
    """Configuration for historical replay through the deployment code path."""

    state_db: Path | None = None
    exact_quantity: bool = True
    quantity_step: Decimal = Decimal("0")
    min_quantity: Decimal = Decimal("0")
    price_tick: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")


class CaptureDemoBroker(DemoBroker):
    """Demo broker that captures closed trades with historical bar timestamps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.current_entry_time: pd.Timestamp | None = None
        self.current_exit_time: pd.Timestamp | None = None
        self.trades: list[SimpleNamespace] = []

    def open_position(self, decision: dict[str, Any], price: float, rules: SymbolRules) -> RuntimePosition | None:
        position = super().open_position(decision, price, rules)
        if position is not None and self.current_entry_time is not None:
            position.entry_time = self.current_entry_time.isoformat()
            self._persist_open(position)
        return position

    def close_position(self, position: RuntimePosition, price: float, reason: str) -> RuntimePosition:
        closed = super().close_position(position, price, reason)
        row = self.store.conn.execute("SELECT payload FROM trades WHERE trade_id = ?", (position.trade_id,)).fetchone()
        if row is None:
            return closed
        payload = json.loads(row["payload"])
        self.trades.append(SimpleNamespace(
            entry_time=pd.Timestamp(position.entry_time),
            exit_time=self.current_exit_time,
            side=payload["side"],
            entry_price=float(payload["entry_price"]),
            exit_price=float(payload["exit_price"]),
            quantity=float(payload["quantity"]),
            notional=float(payload["notional"]),
            stop_loss=float(payload["stop_loss"]),
            take_profit=payload.get("take_profit"),
            exit_reason=payload["exit_reason"],
            pnl=float(payload["pnl"]),
            pnl_pct=float(payload["pnl_pct"]),
            risk_amount=float(payload["risk_amount"]),
            r_multiple=float(payload["r_multiple"]),
            fees=float(payload["fees"]),
            slippage=float(payload["slippage"]),
            holding_bars=int(payload["holding_bars"]),
            metadata={"symbol": payload["symbol"], "source": "live_path_replay"},
        ))
        return closed


def exact_symbol_rules(symbol: str) -> SymbolRules:
    return SymbolRules(
        symbol=symbol,
        quantity_step=Decimal("0"),
        min_quantity=Decimal("0"),
        price_tick=Decimal("0"),
        min_notional=Decimal("0"),
    )


def replay_live_deployment_path(
    featured: pd.DataFrame,
    runtime: CandidateRuntime,
    config: LivePathReplayConfig | None = None,
) -> list[SimpleNamespace]:
    """Replay historical bars through the same lifecycle used by deployment demo/live."""

    cfg = config or LivePathReplayConfig()
    if featured.empty or len(featured) < 3:
        return []
    state_db = cfg.state_db
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if state_db is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="cryptoea-live-replay-")
        state_db = Path(temp_dir.name) / "state.sqlite"
    try:
        store = SQLiteStateStore(state_db)
        rules = exact_symbol_rules(runtime.symbol)
        if not cfg.exact_quantity:
            rules = SymbolRules(
                symbol=runtime.symbol,
                quantity_step=cfg.quantity_step,
                min_quantity=cfg.min_quantity,
                price_tick=cfg.price_tick,
                min_notional=cfg.min_notional,
            )
        broker = CaptureDemoBroker(
            store=store,
            balance=runtime.balance,
            risk_per_trade=runtime.risk_per_trade,
            max_position_pct=runtime.max_position_pct,
            fee_rate=runtime.fee_rate,
            slippage_rate=runtime.slippage_rate,
        )
        position: RuntimePosition | None = None

        for i in range(1, len(featured)):
            current_view = featured.iloc[: i + 1]
            if position is not None:
                broker.current_exit_time = pd.Timestamp(featured.index[i])
                position = manage_position(
                    position,
                    current_view,
                    broker,
                    runtime,
                    rules,
                    mark_price=float(featured["close"].iloc[i]),
                )

            if position is None:
                decision = latest_signal_from_featured(current_view)
                if decision and decision["action"].startswith("ENTER") and i + 1 < len(featured):
                    inserted = store.signal(decision["signal_bar"], runtime.symbol, int(decision["signal"]), decision)
                    if inserted:
                        entry_idx = i + 1
                        broker.current_entry_time = pd.Timestamp(featured.index[entry_idx])
                        position = broker.open_position(decision, float(featured["open"].iloc[entry_idx]), rules)
                        if position is not None:
                            position.last_managed_bar = pd.Timestamp(featured.index[entry_idx]).isoformat()
                            broker._persist_open(position)
                elif decision:
                    store.event("signal_hold", {
                        "signal_bar": decision["signal_bar"],
                        "signal": decision["signal"],
                        "action": decision["action"],
                    })

        if position is not None and position.bars_held >= int(runtime.params["min_holding_bars"]):
            broker.current_exit_time = pd.Timestamp(featured.index[-1])
            reason = "partial_then_max_hold" if position.partial_taken else "max_hold"
            broker.close_position(position, float(featured["close"].iloc[-1]), reason)
        return broker.trades
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def live_replay_metrics(
    featured: pd.DataFrame,
    runtime: CandidateRuntime,
    config: LivePathReplayConfig | None = None,
) -> tuple[list[SimpleNamespace], dict[str, Any]]:
    trades = replay_live_deployment_path(featured, runtime, config=config)
    metrics = metrics_from_trades(trades, featured.index, runtime.balance)
    metrics["trades_per_week"] = trades_per_week(metrics)
    return trades, metrics
