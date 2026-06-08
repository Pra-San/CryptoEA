"""Helper utilities for CryptoEA."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def load_env(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Optional path to .env file. Defaults to project root .env.
    """
    if env_path:
        load_dotenv(env_path)
    else:
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(subdir: str = "") -> Path:
    """Get the data directory path.

    Args:
        subdir: Optional subdirectory within data folder.

    Returns:
        Path to data directory.
    """
    data_dir = get_project_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return data_dir


def get_cache_path(subdir: str = "") -> Path:
    """Get the cache directory path.

    Args:
        subdir: Optional subdirectory within cache folder.

    Returns:
        Path to cache directory.
    """
    cache_dir = get_data_path("cache") / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_binance_data_path(symbol: str = "", timeframe: str = "") -> Path:
    """Get the Binance data file path.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        timeframe: Timeframe (e.g., '1m').

    Returns:
        Path to the data file.
    """
    binance_data = Path("/Users/speketi/Projects/TEA/data/binance")
    if symbol and timeframe:
        return binance_data / timeframe / f"{symbol.lower()}_{timeframe}_spot.csv"
    return binance_data


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a nested dictionary.

    Args:
        d: Dictionary to search.
        key: Key to look for.
        default: Default value if key not found.

    Returns:
        Value at key, or default if not found.
    """
    keys = key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
            if current is default:
                return default
        else:
            return default
    return current


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string.

    Args:
        value: Float value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a float with appropriate decimal places.

    Args:
        value: Float value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted number string.
    """
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    elif abs(value) >= 1:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:,.{decimals}f}"
