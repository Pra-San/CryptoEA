"""SQLite persistence for deployment, demo trading, and shadow comparisons."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStateStore:
    """Small append-friendly state store for resumable deployment."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signals (
                signal_bar TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                signal INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                r_multiple REAL,
                fees REAL NOT NULL DEFAULT 0,
                slippage REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT PRIMARY KEY,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def set_json(self, key: str, value: dict[str, Any] | list[Any] | None) -> None:
        self.conn.execute(
            """
            INSERT INTO kv(key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
            (key, json.dumps(value, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def get_json(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events(kind, payload) VALUES (?, ?)",
            (kind, json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def signal(self, signal_bar: str, symbol: str, signal: int, payload: dict[str, Any]) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO signals(signal_bar, symbol, signal, payload) VALUES (?, ?, ?, ?)",
            (signal_bar, symbol, signal, json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def upsert_trade(self, trade_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO trades(
                trade_id, opened_at, closed_at, symbol, side, quantity, entry_price,
                exit_price, pnl, pnl_pct, r_multiple, fees, slippage, status, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                closed_at=excluded.closed_at,
                exit_price=excluded.exit_price,
                pnl=excluded.pnl,
                pnl_pct=excluded.pnl_pct,
                r_multiple=excluded.r_multiple,
                fees=excluded.fees,
                slippage=excluded.slippage,
                status=excluded.status,
                payload=excluded.payload
            """,
            (
                trade_id,
                payload["opened_at"],
                payload.get("closed_at"),
                payload["symbol"],
                payload["side"],
                float(payload["quantity"]),
                float(payload["entry_price"]),
                payload.get("exit_price"),
                payload.get("pnl"),
                payload.get("pnl_pct"),
                payload.get("r_multiple"),
                float(payload.get("fees", 0.0)),
                float(payload.get("slippage", 0.0)),
                payload["status"],
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def equity(self, ts: str, equity: float, balance: float, unrealized_pnl: float, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO equity(ts, equity, balance, unrealized_pnl, payload) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ts) DO UPDATE SET
                equity=excluded.equity,
                balance=excluded.balance,
                unrealized_pnl=excluded.unrealized_pnl,
                payload=excluded.payload
            """,
            (ts, equity, balance, unrealized_pnl, json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def closed_trade_summary(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT pnl FROM trades WHERE status = 'CLOSED'").fetchall()
        pnl = [float(row["pnl"]) for row in rows if row["pnl"] is not None]
        wins = [value for value in pnl if value > 0]
        losses = [value for value in pnl if value <= 0]
        return {
            "closed_trades": len(pnl),
            "win_rate": len(wins) / len(pnl) if pnl else 0.0,
            "total_pnl": sum(pnl),
            "avg_pnl": sum(pnl) / len(pnl) if pnl else 0.0,
            "gross_profit": sum(wins),
            "gross_loss": abs(sum(losses)),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        }

