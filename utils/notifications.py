"""Notification system for CryptoEA.

Handles Telegram and email alerts for:
- New trade entries/exits
- Circuit breaker triggers
- Error conditions
- Performance milestones
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load env for Telegram config
load_dotenv()


class Notifier:
    """Send alerts via Telegram and/or email."""

    def __init__(self) -> None:
        """Initialize the notifier with environment config."""
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.telegram_bot_token and self.telegram_chat_id)

        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.info(
                "Telegram notifications disabled. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    async def send_alert(self, message: str, level: str = "INFO") -> bool:
        """Send an alert via Telegram.

        Args:
            message: Alert message text.
            level: Alert level (INFO, WARNING, ERROR).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            logger.debug(f"[NOT SENT] [{level}] {message}")
            return False

        try:
            import asyncio

            async def _send() -> None:
                import aiohttp

                url = (
                    f"https://api.telegram.org/bot{self.telegram_bot_token}"
                    f"/sendMessage"
                )
                chat_id = self.telegram_chat_id
                text = f"🤖 *CryptoEA [{level}]*\n\n{message}"

                async with aiohttp.ClientSession() as session:
                    await session.post(
                        url,
                        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    )

            asyncio.run(_send())
            logger.info(f"Telegram alert sent: [{level}]")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def trade_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Send a trade-specific alert.

        Args:
            action: 'ENTRY' or 'EXIT'.
            symbol: Trading symbol.
            price: Trade price.
            quantity: Trade quantity.
            pnl: PnL for exit trades.
            metadata: Additional trade metadata.

        Returns:
            True if sent successfully.
        """
        symbol_display = symbol.replace("USDT", "/USDT")

        if action == "ENTRY":
            message = (
                f"📈 **NEW {symbol_display}**\n"
                f"Price: {price:,.2f}\n"
                f"Qty: {quantity:,.4f}"
            )
            if metadata:
                if "strategy" in metadata:
                    message += f"\nStrategy: {metadata['strategy']}"
                if "signal_strength" in metadata:
                    message += f"\nStrength: {metadata['signal_strength']:.2f}"

        elif action == "EXIT":
            pnl_str = f"\nPnL: {pnl:,.2f} ({pnl / (price * quantity) * 100:.2f}%)" if pnl else ""
            message = (
                f"📉 **CLOSED {symbol_display}**\n"
                f"Price: {price:,.2f}\n"
                f"Qty: {quantity:,.4f}"
                f"{pnl_str}"
            )
            if metadata and "exit_reason" in metadata:
                message += f"\nReason: {metadata['exit_reason']}"

        elif action == "CIRCUIT_BREAKER":
            message = (
                f"🚨 **CIRCUIT BREAKER TRIGGERED**\n"
                f"{message}"
            )
        else:
            message = f"📢 **{symbol_display}**\n{message}"

        return await self.send_alert(message, action)

    async def system_alert(
        self, message: str, level: str = "WARNING"
    ) -> bool:
        """Send a system-level alert.

        Args:
            message: Alert message.
            level: Alert level.

        Returns:
            True if sent successfully.
        """
        return await self.send_alert(message, level)

    def log_only(self, message: str, level: str = "INFO") -> None:
        """Log a message without sending (fallback when notifications disabled).

        Args:
            message: Message to log.
            level: Logging level.
        """
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
