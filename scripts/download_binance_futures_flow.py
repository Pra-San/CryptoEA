#!/usr/bin/env python3
"""Download Binance USD-M futures flow metrics for strategy research.

The public Binance archive provides daily 5-minute metrics files and monthly
funding-rate files. This script combines them into ignored parquet datasets:

data/external/binance_futures_flow/<SYMBOL>_metrics.parquet
data/external/binance_futures_flow/<SYMBOL>_funding.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://data.binance.vision/data/futures/um"
PROJECT_ROOT = Path(__file__).parent.parent
logger = logging.getLogger("binance_futures_flow_download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance futures flow metrics")
    parser.add_argument("--symbols", nargs="+", default=["SOLUSDT"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "external" / "binance_futures_flow")
    return parser.parse_args()


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.DateOffset(months=1)


def fetch_zip_csv(url: str, timeout: float) -> pd.DataFrame | None:
    request = Request(url, headers={"User-Agent": "CryptoEA research downloader"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            return None
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle)


def fetch_metric_day(symbol: str, day: pd.Timestamp, timeout: float) -> pd.DataFrame | None:
    day_text = f"{day:%Y-%m-%d}"
    url = f"{BASE_URL}/daily/metrics/{symbol}/{symbol}-metrics-{day_text}.zip"
    df = fetch_zip_csv(url, timeout)
    if df is None or df.empty:
        return None
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    numeric_cols = [col for col in df.columns if col not in {"create_time", "symbol"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_funding_month(symbol: str, month: pd.Timestamp, timeout: float) -> pd.DataFrame | None:
    month_text = f"{month:%Y-%m}"
    url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month_text}.zip"
    df = fetch_zip_csv(url, timeout)
    if df is None or df.empty:
        return None
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df["funding_interval_hours"] = pd.to_numeric(df["funding_interval_hours"], errors="coerce")
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    df["symbol"] = symbol
    return df


def download_metrics(symbol: str, start: pd.Timestamp, end: pd.Timestamp, workers: int, timeout: float) -> pd.DataFrame:
    days = list(pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC"))
    rows: list[pd.DataFrame] = []
    misses = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_metric_day, symbol, day, timeout): day for day in days}
        for i, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            result = future.result()
            if result is None:
                misses += 1
            else:
                rows.append(result)
            if i % 100 == 0:
                logger.info("%s metrics: checked %d/%d days, files %d, misses %d", symbol, i, len(days), len(rows), misses)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("create_time").drop_duplicates(subset=["create_time", "symbol"])
    return df[(df["create_time"] >= start) & (df["create_time"] <= end)]


def download_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp, workers: int, timeout: float) -> pd.DataFrame:
    months = list(month_starts(start, end))
    rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = {executor.submit(fetch_funding_month, symbol, month, timeout): month for month in months}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("calc_time").drop_duplicates(subset=["calc_time", "symbol"])
    return df[(df["calc_time"] >= start) & (df["calc_time"] <= end)]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        logger.info("Downloading %s from %s to %s", symbol, date.isoformat(start.date()), date.isoformat(end.date()))
        metrics = download_metrics(symbol, start, end, args.workers, args.timeout)
        funding = download_funding(symbol, start, end, args.workers, args.timeout)
        if metrics.empty:
            logger.warning("%s metrics unavailable", symbol)
        else:
            metrics_path = args.output_dir / f"{symbol}_metrics.parquet"
            metrics.to_parquet(metrics_path, index=False)
            logger.info("Wrote %s rows to %s", len(metrics), metrics_path)
        if funding.empty:
            logger.warning("%s funding unavailable", symbol)
        else:
            funding_path = args.output_dir / f"{symbol}_funding.parquet"
            funding.to_parquet(funding_path, index=False)
            logger.info("Wrote %s rows to %s", len(funding), funding_path)


if __name__ == "__main__":
    main()
