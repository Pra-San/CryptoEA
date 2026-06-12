"""Small signed REST client for Binance USD-M Futures.

The client intentionally avoids an exchange SDK wrapper so live order behavior is
visible in this repository. It only implements endpoints used by the deployment
runners.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MAINNET_URL = "https://fapi.binance.com"
DEMO_URL = "https://demo-fapi.binance.com"


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns a non-success response."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Binance API error {status_code}: {payload}")


@dataclass(frozen=True)
class SymbolRules:
    """Exchange filters needed for safe order rounding."""

    symbol: str
    quantity_step: Decimal
    min_quantity: Decimal
    price_tick: Decimal
    min_notional: Decimal

    def round_quantity(self, value: float | Decimal) -> Decimal:
        quantity = Decimal(str(value))
        if self.quantity_step <= 0:
            return quantity
        rounded = (quantity / self.quantity_step).to_integral_value(rounding=ROUND_DOWN) * self.quantity_step
        return rounded.normalize()

    def round_price(self, value: float | Decimal) -> Decimal:
        price = Decimal(str(value))
        if self.price_tick <= 0:
            return price
        rounded = (price / self.price_tick).to_integral_value(rounding=ROUND_DOWN) * self.price_tick
        return rounded.normalize()

    def round_price_up(self, value: float | Decimal) -> Decimal:
        price = Decimal(str(value))
        if self.price_tick <= 0:
            return price
        units = (price / self.price_tick).to_integral_value(rounding=ROUND_DOWN)
        rounded = units * self.price_tick
        if rounded < price:
            rounded += self.price_tick
        return rounded.normalize()


class BinanceFuturesClient:
    """REST client for USD-M Futures market data, account data, and orders."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = MAINNET_URL,
        timeout: float = 10.0,
        recv_window: int = 5000,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret if api_secret is not None else os.getenv("BINANCE_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.recv_window = recv_window
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    @classmethod
    def from_exchange(cls, exchange: str, **kwargs: Any) -> "BinanceFuturesClient":
        base_url = DEMO_URL if exchange == "demo" else MAINNET_URL
        return cls(base_url=base_url, **kwargs)

    def _signed_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_secret:
            raise RuntimeError("BINANCE_API_SECRET is required for signed endpoints")
        signed = dict(params or {})
        signed.setdefault("recvWindow", self.recv_window)
        signed["timestamp"] = int(time.time() * 1000)
        query = urlencode(signed, doseq=True)
        signed["signature"] = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return signed

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        retry_safe: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        request_params = self._signed_params(params) if signed else dict(params or {})
        attempts = 3 if retry_safe else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.request(method, url, params=request_params, timeout=self.timeout)
                if response.status_code in {418, 429}:
                    retry_after = float(response.headers.get("Retry-After", "1"))
                    time.sleep(min(retry_after, 10.0))
                if response.status_code >= 400:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = response.text
                    raise BinanceAPIError(response.status_code, payload)
                return response.json()
            except (requests.Timeout, requests.ConnectionError, BinanceAPIError) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise
                sleep_for = min(0.5 * (2**attempt), 5.0)
                logger.warning("Binance request retry %s %s after %s", method, path, exc)
                time.sleep(sleep_for)
        if last_error:
            raise last_error
        raise RuntimeError("unreachable request state")

    def server_time(self) -> int:
        return int(self._request("GET", "/fapi/v1/time")["serverTime"])

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def symbol_rules(self, symbol: str) -> SymbolRules:
        data = self.exchange_info()
        match = next((item for item in data["symbols"] if item["symbol"] == symbol), None)
        if not match:
            raise ValueError(f"Symbol {symbol} not found in exchangeInfo")
        filters = {item["filterType"]: item for item in match["filters"]}
        lot = filters.get("MARKET_LOT_SIZE") or filters["LOT_SIZE"]
        price = filters["PRICE_FILTER"]
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {"notional": "0"}
        return SymbolRules(
            symbol=symbol,
            quantity_step=Decimal(str(lot["stepSize"])),
            min_quantity=Decimal(str(lot["minQty"])),
            price_tick=Decimal(str(price["tickSize"])),
            min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "0")))),
        )

    def klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        payload = self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(payload, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.set_index("timestamp")

    def closed_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        df = self.klines(symbol, interval, limit=limit)
        now_ms = self.server_time()
        close_ms = (df["close_time"].astype("int64") // 1_000_000).astype("int64")
        return df.loc[close_ms < now_ms].drop(columns=["close_time"])

    def mark_price(self, symbol: str) -> float:
        payload = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(payload["markPrice"])

    def account_balance(self) -> list[dict[str, Any]]:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def usdt_available_balance(self) -> float:
        for row in self.account_balance():
            if row.get("asset") == "USDT":
                return float(row.get("availableBalance", row.get("balance", 0.0)))
        raise RuntimeError("USDT balance not found")

    def position_risk(self, symbol: str) -> list[dict[str, Any]]:
        return self._request("GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True)

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def cancel_all_orders(self, symbol: str) -> Any:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True, retry_safe=False)

    def new_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal | float | None = None,
        reduce_only: bool = False,
        stop_price: Decimal | float | None = None,
        client_order_id: str | None = None,
        response_type: str = "RESULT",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "newOrderRespType": response_type,
        }
        if quantity is not None:
            params["quantity"] = str(quantity)
        if reduce_only:
            params["reduceOnly"] = "true"
        if stop_price is not None:
            params["stopPrice"] = str(stop_price)
            params["workingType"] = "CONTRACT_PRICE"
        if client_order_id:
            params["newClientOrderId"] = client_order_id[:36]
        return self._request("POST", "/fapi/v1/order", params, signed=True, retry_safe=False)
