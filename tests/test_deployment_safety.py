from decimal import Decimal

import pytest

from deployment.binance_futures import SymbolRules
from deployment.brokers import LIVE_ACK_VALUE, LiveBinanceBroker
from deployment.state import SQLiteStateStore


def test_symbol_rules_round_quantity_and_price() -> None:
    rules = SymbolRules(
        symbol="SOLUSDT",
        quantity_step=Decimal("0.1"),
        min_quantity=Decimal("0.1"),
        price_tick=Decimal("0.01"),
        min_notional=Decimal("5"),
    )

    assert rules.round_quantity(12.349) == Decimal("12.3")
    assert rules.round_price(123.456) == Decimal("123.45")
    assert rules.round_price_up(123.451) == Decimal("123.46")


def test_live_broker_requires_exact_ack(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CRYPTOEA_LIVE_TRADING_ACK", raising=False)
    store = SQLiteStateStore(tmp_path / "state.sqlite")

    with pytest.raises(RuntimeError):
        LiveBinanceBroker(
            client=object(),  # type: ignore[arg-type]
            store=store,
            risk_per_trade=0.01,
            max_position_pct=1.0,
            fee_rate=0.0005,
            slippage_rate=0.001,
            max_notional_usdt=1000,
        )

    monkeypatch.setenv("CRYPTOEA_LIVE_TRADING_ACK", LIVE_ACK_VALUE)


def test_state_store_closed_trade_summary(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite")
    store.upsert_trade(
        "T1",
        {
            "opened_at": "2026-01-01T00:00:00Z",
            "closed_at": "2026-01-01T01:00:00Z",
            "symbol": "SOLUSDT",
            "side": "LONG",
            "quantity": 1,
            "entry_price": 100,
            "exit_price": 105,
            "pnl": 5,
            "pnl_pct": 0.05,
            "r_multiple": 1.2,
            "fees": 0.1,
            "slippage": 0.1,
            "status": "CLOSED",
        },
    )
    store.upsert_trade(
        "T2",
        {
            "opened_at": "2026-01-02T00:00:00Z",
            "closed_at": "2026-01-02T01:00:00Z",
            "symbol": "SOLUSDT",
            "side": "SHORT",
            "quantity": 1,
            "entry_price": 100,
            "exit_price": 98,
            "pnl": -2,
            "pnl_pct": -0.02,
            "r_multiple": -0.5,
            "fees": 0.1,
            "slippage": 0.1,
            "status": "CLOSED",
        },
    )

    summary = store.closed_trade_summary()
    assert summary["closed_trades"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["total_pnl"] == 3
    assert summary["profit_factor"] == 2.5

