"""WebSocket data feed module for real-time market data."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KlineData:
    """Represents a single kline (OHLCV) candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool


@dataclass
class DataFeedConfig:
    """Configuration for WebSocket data feed."""

    symbols: tuple[str, ...] = ("btcusdt",)
    interval: str = "1m"
    buffer_size: int = 10000
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    state_dir: str = "data/state"


@dataclass
class MarketDepth:
    """Bid/ask depth snapshot."""

    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid_price(self) -> float:
        """Calculate mid price from best bid/ask."""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            return (best_bid + best_ask) / 2.0
        return 0.0

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            return best_ask - best_bid
        return 0.0


class DataFeed:
    """WebSocket-based real-time market data feed.

    Connects to Binance WebSocket streams for real-time kline data,
    depth updates, and trades. Buffers data locally and provides
    callbacks for processing incoming data.
    """

    def __init__(
        self,
        config: DataFeedConfig | None = None,
    ) -> None:
        self.config = config or DataFeedConfig()
        self.kline_buffer: dict[str, deque[KlineData]] = {
            symbol: deque(maxlen=self.config.buffer_size)
            for symbol in self.config.symbols
        }
        self.depth_buffer: dict[str, MarketDepth] = {
            symbol: MarketDepth()
            for symbol in self.config.symbols
        }
        self._running = False
        self._ws_task: asyncio.Task | None = None
        self._on_kline_callbacks: list[Callable] = []
        self._on_depth_callbacks: list[Callable] = []
        self._on_trade_callbacks: list[Callable] = []

    def on_kline(
        self,
        callback: Callable[[str, KlineData], None],
    ) -> None:
        """Register a callback for kline data events.

        Args:
            callback: Function that receives (symbol, kline_data).
        """
        self._on_kline_callbacks.append(callback)

    def on_depth(
        self,
        callback: Callable[[str, MarketDepth], None],
    ) -> None:
        """Register a callback for depth data events.

        Args:
            callback: Function that receives (symbol, depth).
        """
        self._on_depth_callbacks.append(callback)

    def on_trade(
        self,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Register a callback for trade data events.

        Args:
            callback: Function that receives (symbol, trade_info).
        """
        self._on_trade_callbacks.append(callback)

    async def start(
        self,
    ) -> None:
        """Start WebSocket connections for all configured symbols."""
        self._running = True
        logger.info(
            "Starting data feed for symbols: %s",
            ", ".join(self.config.symbols),
        )

        tasks: list[asyncio.Task] = []

        for symbol in self.config.symbols:
            tasks.append(asyncio.create_task(
                self._connect_stream(symbol)
            ))

        self._ws_task = asyncio.gather(*tasks, return_exceptions=True)

        try:
            await self._ws_task
        except asyncio.CancelledError:
            logger.info("Data feed stopped")
        except Exception as e:
            logger.error("Data feed error: %s", e)
        finally:
            self._running = False

    async def stop(
        self,
    ) -> None:
        """Stop all WebSocket connections."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        logger.info("Data feed stopped")

    async def _connect_stream(
        self,
        symbol: str,
    ) -> None:
        """Connect to WebSocket stream for a symbol.

        Args:
            symbol: Trading pair (e.g., 'btcusdt').
        """
        reconnect_attempts = 0

        while self._running:
            try:
                # Using Binance WebSocket API
                kline_stream = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{self.config.interval}"
                depth_stream = f"wss://stream.binance.com:9443/ws/{symbol}@depth20@100ms"
                trade_stream = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

                combined_stream = f"/ws/{kline_stream.split('/')[-1]}/{depth_stream.split('/')[-1]}/{trade_stream.split('/')[-1]}"
                ws_url = f"wss://stream.binance.com:9443{combined_stream}"

                logger.info(
                    "Connecting to WebSocket for %s",
                    symbol,
                )

                import websockets

                async with websockets.connect(ws_url) as ws:
                    reconnect_attempts = 0

                    async for message in ws:
                        if not self._running:
                            break

                        data = json.loads(message)

                        await self._process_message(symbol, data)

            except Exception as e:
                reconnect_attempts += 1
                logger.warning(
                    "WebSocket disconnected for %s (attempt %d/%d): %s",
                    symbol,
                    reconnect_attempts,
                    self.config.max_reconnect_attempts,
                    e,
                )

                if reconnect_attempts > self.config.max_reconnect_attempts:
                    logger.error(
                        "Max reconnection attempts reached for %s",
                        symbol,
                    )
                    break

                await asyncio.sleep(self.config.reconnect_delay)

    async def _process_message(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> None:
        """Process incoming WebSocket message.

        Args:
            symbol: Trading pair.
            data: Parsed JSON message.
        """
        try:
            if "k" in data:
                # Kline/candle data
                kline = data["k"]
                kline_data = KlineData(
                    timestamp=datetime.fromtimestamp(
                        kline.get("t", 0) / 1000, tz=timezone.utc
                    ),
                    open=float(kline.get("o", 0)),
                    high=float(kline.get("h", 0)),
                    low=float(kline.get("l", 0)),
                    close=float(kline.get("c", 0)),
                    volume=float(kline.get("v", 0)),
                    quote_volume=float(kline.get("q", 0)),
                    trades=kline.get("n", 0),
                    is_closed=kline.get("x", False),
                )

                self.kline_buffer[symbol].append(kline_data)

                for callback in self._on_kline_callbacks:
                    try:
                        callback(symbol, kline_data)
                    except Exception as e:
                        logger.error("Kline callback error: %s", e)

            elif "b" in data and "a" in data:
                # Depth data
                bids = [
                    (float(b[0]), float(b[1]))
                    for b in data.get("b", [])
                ]
                asks = [
                    (float(a[0]), float(a[1]))
                    for a in data.get("a", [])
                ]

                depth = MarketDepth(
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now(timezone.utc),
                )

                self.depth_buffer[symbol] = depth

                for callback in self._on_depth_callbacks:
                    try:
                        callback(symbol, depth)
                    except Exception as e:
                        logger.error("Depth callback error: %s", e)

            elif "t" in data and "p" in data:
                # Trade data
                trade_info = {
                    "trade_id": data.get("t"),
                    "price": float(data.get("p", 0)),
                    "quantity": float(data.get("q", 0)),
                    "is_buyer_maker": data.get("m", False),
                    "timestamp": datetime.fromtimestamp(
                        data.get("T", 0) / 1000, tz=timezone.utc
                    ),
                }

                for callback in self._on_trade_callbacks:
                    try:
                        callback(symbol, trade_info)
                    except Exception as e:
                        logger.error("Trade callback error: %s", e)

        except Exception as e:
            logger.error("Error processing message for %s: %s", symbol, e)

    def get_kline_dataframe(
        self,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Get buffered kline data as a DataFrame.

        Args:
            symbol: Symbol to get data for (None for all).

        Returns:
            DataFrame of kline data.
        """
        symbol = symbol or (
            self.config.symbols[0] if self.config.symbols else None
        )

        if symbol is None or symbol not in self.kline_buffer:
            return pd.DataFrame()

        buffer = self.kline_buffer[symbol]

        if not buffer:
            return pd.DataFrame()

        records = [
            {
                "timestamp": k.timestamp,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
                "quote_volume": k.quote_volume,
                "trades": k.trades,
                "is_closed": k.is_closed,
            }
            for k in buffer
        ]

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    def get_depth(
        self,
        symbol: str | None = None,
    ) -> MarketDepth:
        """Get current market depth for a symbol.

        Args:
            symbol: Symbol to get depth for.

        Returns:
            MarketDepth object.
        """
        symbol = symbol or (
            self.config.symbols[0] if self.config.symbols else None
        )

        if symbol is None:
            return MarketDepth()

        return self.depth_buffer.get(symbol, MarketDepth())

    def save_state(
        self,
    ) -> Path:
        """Save current buffer state to disk.

        Returns:
            Path to saved state file.
        """
        state_dir = Path(self.config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        state = {}

        for symbol, buffer in self.kline_buffer.items():
            records = [
                {
                    "timestamp": k.timestamp.isoformat(),
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                }
                for k in buffer
            ]
            state[symbol] = records

        state_path = state_dir / "datafeed_state.json"

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Data feed state saved to %s", state_path)

        return state_path
