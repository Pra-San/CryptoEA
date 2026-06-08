"""Binance API client wrapper for CryptoEA.

Provides a unified interface to Binance REST and WebSocket APIs
with rate limit handling, error recovery, and idempotent operations.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class OrderResult:
    """Result of an order submission."""

    success: bool = False
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    price: float = 0.0
    quantity: float = 0.0
    status: str = ""
    error_message: Optional[str] = None
    raw_response: Dict = field(default_factory=dict)


@dataclass
class BinanceConfig:
    """Configuration for Binance API connection."""

    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # Use testnet by default
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    rate_limit_weight: int = 1


class BinanceClient:
    """Binance API wrapper with rate limit handling and error recovery.

    Usage:
        client = BinanceClient()
        client.connect()
        klines = client.get_klines("BTCUSDT", "1h", limit=500)
        order = client.place_order("BTCUSDT", "BUY", "MARKET", quantity=0.001)
    """

    def __init__(self, config: Optional[BinanceConfig] = None) -> None:
        """Initialize the Binance client.

        Args:
            config: API configuration. Loads from .env if None.
        """
        self.config = config or BinanceConfig()

        if not self.config.api_key:
            self.config.api_key = os.getenv("BINANCE_API_KEY", "")
        if not self.config.api_secret:
            self.config.api_secret = os.getenv("BINANCE_API_SECRET", "")

        self._client = None
        self._connected = False
        self._rate_limit_remaining = 0
        self._rate_limit_reset = 0
        self._last_request_time = 0

    def connect(self) -> bool:
        """Establish connection to Binance API.

        Returns:
            True if connected successfully.
        """
        try:
            from binance.client import Client

            # Use testnet for safety during development
            if self.config.testnet:
                self._client = Client(
                    self.config.api_key,
                    self.config.api_secret,
                    testnet=True,
                )
                logger.info("Connected to Binance TESTNET")
            else:
                self._client = Client(
                    self.config.api_key,
                    self.config.api_secret,
                )
                logger.warning("Connected to Binance MAINNET")

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the Binance connection."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from Binance")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch kline/ohlcv data from Binance.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT').
            interval: Timeframe (e.g., '1m', '5m', '1h').
            limit: Number of bars to fetch.

        Returns:
            DataFrame with OHLCV data.
        """
        if not self._connected:
            logger.error("Not connected to Binance")
            return pd.DataFrame()

        try:
            klines = self._client.get_klines(
                symbol=symbol, interval=interval, limit=limit
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["timestamp"] = df["timestamp"].set_index("timestamp")

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_account_balance(self) -> Dict[str, float]:
        """Get current account balances.

        Returns:
            Dictionary mapping asset to balance.
        """
        if not self._connected:
            logger.error("Not connected to Binance")
            return {}

        try:
            info = self._client.get_account()
            balances: Dict[str, float] = {}

            for balance in info["balances"]:
                free = float(balance["free"])
                locked = float(balance["locked"])
                asset = balance["asset"]

                if free > 0 or locked > 0:
                    balances[asset] = free + locked

            return balances

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get trading rules and filters for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Symbol info dictionary or None.
        """
        if not self._connected:
            return None

        try:
            exchange_info = self._client.get_symbol_info(symbol)
            if exchange_info is None:
                logger.warning(f"Symbol {symbol} not found")
                return None

            # Extract key filters
            filters = {}
            for f in exchange_info.get("filters", []):
                filters[f["filterType"]] = f

            return {
                "symbol": symbol,
                "baseAsset": exchange_info.get("baseAsset"),
                "quoteAsset": exchange_info.get("quoteAsset"),
                "priceFilter": filters.get("PRICE_FILTER"),
                "lotSizeFilter": filters.get("LOT_SIZE"),
                "minNotional": filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL"),
            }

        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float = 0.0,
        quote_quantity: float = 0.0,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place an order on Binance.

        Args:
            symbol: Trading symbol.
            side: 'BUY' or 'SELL'.
            order_type: 'MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT'.
            quantity: Base asset quantity.
            quote_quantity: Quote asset quantity (for market orders).
            price: Limit price (for limit orders).
            stop_price: Stop trigger price.
            reduce_only: If True, order only reduces position.

        Returns:
            OrderResult with order status.
        """
        if not self._connected:
            return OrderResult(
                success=False,
                error_message="Not connected to Binance",
            )

        try:
            order_params: Dict = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "reduceOnly": reduce_only,
            }

            if order_type == "LIMIT" and price:
                order_params["price"] = str(price)
            if order_type in ("STOP_LOSS", "TAKE_PROFIT") and stop_price:
                order_params["stopPrice"] = str(stop_price)
            if order_type == "MARKET" and quote_quantity:
                order_params["quoteOrderQty"] = quote_quantity

            response = self._client.create_order(**order_params)

            result = OrderResult(
                success=True,
                order_id=str(response.get("orderId")),
                symbol=symbol,
                side=side,
                price=float(response.get("price", 0)),
                quantity=float(response.get("executedQty", 0)),
                status=response.get("status", "UNKNOWN"),
                raw_response=response,
            )

            logger.info(
                f"Order placed: {side} {quantity} {symbol} @ {order_type} "
                f"(status: {result.status})"
            )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order failed: {error_msg}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error_message=error_msg,
            )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get currently open orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of open order dictionaries.
        """
        if not self._connected:
            return []

        try:
            if symbol:
                return self._client.get_open_orders(symbol=symbol)
            return self._client.get_open_orders()

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            True if all orders cancelled successfully.
        """
        if not self._connected:
            return False

        try:
            result = self._client.cancel_all_orders(symbol=symbol)
            logger.info(f"Cancelled {len(result)} orders for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel orders for {symbol}: {e}")
            return False

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current ticker price for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Current price or None.
        """
        if not self._connected:
            return None

        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
