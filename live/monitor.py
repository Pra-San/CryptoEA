"""Live trading monitor with performance tracking and alerts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live.notifications import Notifier

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for live trading monitor."""

    update_interval: int = 60  # seconds
    state_dir: str = "data/state"
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        "max_drawdown_pct": 10.0,
        "consecutive_losses": 5,
        "daily_loss_pct": 3.0,
    })
    use_telegram: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None


@dataclass
class PerformanceSnapshot:
    """Snapshot of trading performance at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    daily_pnl: float = 0.0
    equity: float = 0.0
    initial_equity: float = 0.0


class TradingMonitor:
    """Real-time trading performance monitor with alerting.

    Tracks trading metrics, generates performance snapshots,
    and sends alerts when thresholds are breached.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.notifier = Notifier(
            use_telegram=self.config.use_telegram,
            telegram_token=self.config.telegram_token,
            telegram_chat_id=self.config.telegram_chat_id,
        )
        self.snapshots: list[PerformanceSnapshot] = []
        self.trade_history: list[dict[str, Any]] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self._last_alert_time: dict[str, float] = {}

    def record_trade(
        self,
        trade: dict[str, Any],
    ) -> None:
        """Record a completed trade.

        Args:
            trade: Dictionary containing trade details.
        """
        self.trade_history.append(trade)

        pnl = trade.get("pnl", 0.0)

        if self.equity_history:
            prev_equity = self.equity_history[-1][1]
            new_equity = prev_equity + pnl
        else:
            new_equity = trade.get("initial_capital", 10000.0) + pnl

        self.equity_history.append(
            (datetime.now(timezone.utc), new_equity)
        )

        # Generate snapshot
        snapshot = self._generate_snapshot()
        self.snapshots.append(snapshot)

        # Check alert thresholds
        self._check_alerts(snapshot)

        logger.info(
            "Trade recorded: PnL=%.2f, Equity=%.2f, Win Rate=%.2f%%",
            pnl,
            new_equity,
            snapshot.win_rate * 100,
        )

    def _generate_snapshot(
        self,
    ) -> PerformanceSnapshot:
        """Generate a performance snapshot from trade history.

        Returns:
            PerformanceSnapshot object.
        """
        if not self.trade_history:
            return PerformanceSnapshot()

        pnls = [t.get("pnl", 0.0) for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(pnls)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Calculate drawdown
        if self.equity_history:
            equity_values = [e for _, e in self.equity_history]
            peak = np.maximum.accumulate(equity_values)
            drawdowns = (peak - equity_values) / peak
            max_drawdown_pct = float(np.max(drawdowns)) * 100
            current_drawdown_pct = float(drawdowns[-1]) * 100
            current_equity = equity_values[-1]
        else:
            max_drawdown_pct = 0.0
            current_drawdown_pct = 0.0
            current_equity = 0.0

        initial_equity = (
            self.trade_history[0].get("initial_capital", 10000.0)
            if self.trade_history
            else 10000.0
        )

        # Calculate Sharpe ratio
        if len(pnls) > 1:
            returns = pd.Series(pnls).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = float(
                    (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
                )
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate Calmar ratio
        if max_drawdown_pct > 0:
            total_return = (
                (current_equity - initial_equity) / initial_equity
            )
            calmar_ratio = float(total_return / (max_drawdown_pct / 100))
        else:
            calmar_ratio = 0.0

        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0

        for pnl in pnls:
            if pnl <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(
                    max_consecutive_losses, consecutive_losses
                )
            else:
                consecutive_losses = 0

        # Daily PnL
        if self.trade_history:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = sum(
                t.get("pnl", 0.0)
                for t in self.trade_history
                if t.get("date", "") == today
            )
        else:
            daily_pnl = 0.0

        return PerformanceSnapshot(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            profit_factor=float(profit_factor),
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown_pct=current_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_consecutive_losses=max_consecutive_losses,
            current_consecutive_losses=consecutive_losses,
            daily_pnl=daily_pnl,
            equity=current_equity,
            initial_equity=initial_equity,
        )

    def _check_alerts(
        self,
        snapshot: PerformanceSnapshot,
    ) -> None:
        """Check performance against alert thresholds.

        Args:
            snapshot: Current performance snapshot.
        """
        now = time.time()
        cooldown = 300  # 5 minutes between same alerts

        thresholds = self.config.alert_thresholds

        # Max drawdown alert
        if (
            snapshot.max_drawdown_pct > thresholds.get("max_drawdown_pct", 10.0)
            and now - self._last_alert_time.get("max_drawdown", 0) > cooldown
        ):
            self._last_alert_time["max_drawdown"] = now
            alert_msg = (
                f"🔴 MAX DRAWDOWN ALERT\n"
                f"Symbol: {self.config.alert_thresholds.get('symbol', 'N/A')}\n"
                f"Max DD: {snapshot.max_drawdown_pct:.2f}%\n"
                f"Current Equity: {snapshot.equity:.2f}"
            )
            logger.warning(alert_msg)
            self.notifier.send_alert(alert_msg)

        # Consecutive losses alert
        if (
            snapshot.current_consecutive_losses
            >= thresholds.get("consecutive_losses", 5)
            and now - self._last_alert_time.get("consecutive_losses", 0) > cooldown
        ):
            self._last_alert_time["consecutive_losses"] = now
            alert_msg = (
                f"🔴 CONSECUTIVE LOSSES ALERT\n"
                f"Consecutive Losses: {snapshot.current_consecutive_losses}\n"
                f"Win Rate: {snapshot.win_rate * 100:.2f}%\n"
                f"Total PnL: {snapshot.total_pnl:.2f}"
            )
            logger.warning(alert_msg)
            self.notifier.send_alert(alert_msg)

        # Daily loss alert
        if (
            snapshot.daily_pnl < -thresholds.get("daily_loss_pct", 3.0)
            * snapshot.initial_equity
            / 100
            and now - self._last_alert_time.get("daily_loss", 0) > cooldown
        ):
            self._last_alert_time["daily_loss"] = now
            alert_msg = (
                f"🔴 DAILY LOSS ALERT\n"
                f"Daily PnL: {snapshot.daily_pnl:.2f}\n"
                f"Daily PnL %: {snapshot.daily_pnl / snapshot.initial_equity * 100:.2f}%\n"
                f"Equity: {snapshot.equity:.2f}"
            )
            logger.warning(alert_msg)
            self.notifier.send_alert(alert_msg)

    def get_status_report(
        self,
    ) -> dict[str, Any]:
        """Generate a comprehensive status report.

        Returns:
            Dictionary containing current trading status.
        """
        if not self.snapshots:
            return {"message": "No trading data available"}

        latest = self.snapshots[-1]

        return {
            "timestamp": latest.timestamp.isoformat(),
            "total_trades": latest.total_trades,
            "win_rate": latest.win_rate,
            "total_pnl": latest.total_pnl,
            "profit_factor": latest.profit_factor,
            "max_drawdown_pct": latest.max_drawdown_pct,
            "current_drawdown_pct": latest.current_drawdown_pct,
            "sharpe_ratio": latest.sharpe_ratio,
            "calmar_ratio": latest.calmar_ratio,
            "equity": latest.equity,
            "initial_equity": latest.initial_equity,
        }

    def save_state(
        self,
    ) -> Path:
        """Save monitor state to disk.

        Returns:
            Path to saved state file.
        """
        state_dir = Path(self.config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "total_trades": s.total_trades,
                    "winning_trades": s.winning_trades,
                    "losing_trades": s.losing_trades,
                    "win_rate": s.win_rate,
                    "total_pnl": s.total_pnl,
                    "avg_win": s.avg_win,
                    "avg_loss": s.avg_loss,
                    "profit_factor": s.profit_factor,
                    "max_drawdown_pct": s.max_drawdown_pct,
                    "current_drawdown_pct": s.current_drawdown_pct,
                    "sharpe_ratio": s.sharpe_ratio,
                    "calmar_ratio": s.calmar_ratio,
                    "max_consecutive_losses": s.max_consecutive_losses,
                    "current_consecutive_losses": s.current_consecutive_losses,
                    "daily_pnl": s.daily_pnl,
                    "equity": s.equity,
                    "initial_equity": s.initial_equity,
                }
                for s in self.snapshots
            ],
            "trade_history": self.trade_history,
            "equity_history": [
                (ts.isoformat(), eq) for ts, eq in self.equity_history
            ],
        }

        state_path = state_dir / "monitor_state.json"

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Monitor state saved to %s", state_path)

        return state_path
