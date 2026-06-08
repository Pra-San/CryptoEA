"""State management for live trading.

Handles position tracking, PnL calculation, and state persistence.
Ensures crash recovery and state reconciliation with exchange.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Snapshot of trading state for persistence and recovery."""

    timestamp: str = ""
    balance: float = 0.0
    positions: List[Dict] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    circuit_breaker_active: bool = False
    last_trade_time: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class StateManager:
    """Manages trading state with persistence and crash recovery.

    Usage:
        sm = StateManager(state_dir="/path/to/state")
        sm.save()  # Save current state
        snap = sm.load()  # Load last state for recovery
    """

    def __init__(self, state_dir: str = "data/state") -> None:
        """Initialize state manager.

        Args:
            state_dir: Directory to store state files.
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_file = self.state_dir / "latest_snapshot.json"
        self._state: Optional[StateSnapshot] = None

    def save(
        self,
        balance: float,
        positions: List[Dict],
        open_orders: List[Dict],
        total_realized_pnl: float = 0.0,
        total_unrealized_pnl: float = 0.0,
        circuit_breaker_active: bool = False,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save current trading state to disk.

        Args:
            balance: Current account balance.
            positions: List of open positions.
            open_orders: List of open orders.
            total_realized_pnl: Cumulative realized PnL.
            total_unrealized_pnl: Current unrealized PnL.
            circuit_breaker_active: Whether circuit breaker is active.
            metadata: Additional state metadata.
        """
        snapshot = StateSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            balance=balance,
            positions=positions,
            open_orders=open_orders,
            total_realized_pnl=total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
            circuit_breaker_active=circuit_breaker_active,
            metadata=metadata or {},
        )

        try:
            with open(self._snapshot_file, "w") as f:
                json.dump(asdict(snapshot), f, indent=2)

            logger.debug(f"State saved: balance={balance:,.0f}, "
                        f"positions={len(positions)}, "
                        f"realized_pnl={total_realized_pnl:,.2f}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load(self) -> Optional[StateSnapshot]:
        """Load the last saved state for crash recovery.

        Returns:
            StateSnapshot if file exists, None otherwise.
        """
        if not self._snapshot_file.exists():
            logger.info("No previous state found. Starting fresh.")
            return None

        try:
            with open(self._snapshot_file, "r") as f:
                data = json.load(f)

            snapshot = StateSnapshot(**data)
            logger.info(f"State loaded from {snapshot.timestamp}: "
                       f"balance={snapshot.balance:,.0f}, "
                       f"positions={len(snapshot.positions)}")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def reconcile(
        self, exchange_positions: List[Dict], exchange_orders: List[Dict]
    ) -> List[str]:
        """Reconcile saved state with exchange state.

        Identifies discrepancies between local state and exchange.

        Args:
            exchange_positions: Current positions from exchange.
            exchange_orders: Current orders from exchange.

        Returns:
            List of reconciliation actions taken.
        """
        actions = []
        snapshot = self.load()

        if snapshot is None:
            return ["No previous state to reconcile"]

        # Check for unmatched local positions
        local_symbols = {p["symbol"] for p in snapshot.positions}
        exchange_symbols = {p["symbol"] for p in exchange_positions}

        missing_from_exchange = local_symbols - exchange_symbols
        for symbol in missing_from_exchange:
            actions.append(f"Position {symbol} closed on exchange, "
                          f"removing from local state")

        # Check for unmatched exchange positions
        missing_from_local = exchange_symbols - local_symbols
        for symbol in missing_from_local:
            actions.append(f"Position {symbol} found on exchange, "
                          f"adding to local state")

        # Check for stale open orders
        local_order_ids = {o["orderId"] for o in snapshot.open_orders}
        exchange_order_ids = {o["orderId"] for o in exchange_orders}

        stale_orders = local_order_ids - exchange_order_ids
        for order_id in stale_orders:
            actions.append(f"Stale order {order_id} cancelled on exchange")

        if actions:
            logger.warning(f"Reconciliation found {len(actions)} discrepancies: "
                          f"{actions}")
        else:
            logger.info("State reconciliation: OK - no discrepancies")

        return actions

    def clear(self) -> None:
        """Clear all saved state."""
        if self._snapshot_file.exists():
            self._snapshot_file.unlink()
            logger.info("State cleared.")
