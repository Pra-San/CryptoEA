#!/usr/bin/env python3
"""Adaptive walk-forward selector over BTC-regime-gated candidates.

This expands the saved candidate universe by applying causal BTC daily trend
and volatility gates to each candidate's native signals. Each out-of-sample
window ranks variants using only the prior training window.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import DataLoader
from scripts.evaluate_strategy_portfolio import (
    Candidate,
    apply_candidate_strategy,
    load_base_data,
    load_candidate,
    metrics_dict,
    run_candidate,
    scale_trade,
    summarize_windows,
    write_csv,
)
from scripts.run_backtest import vectorized_preprocess


logger = logging.getLogger("regime_gated_wf")


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    mode: str
    sma: int = 100
    roc_days: int = 14
    roc_min: float = 0.0
    vol_lookback: int = 120
    vol_quantile: float = 0.70


@dataclass(frozen=True)
class Variant:
    label: str
    candidate: Candidate
    regime: RegimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate regime-gated candidate variants")
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--candidate-glob", action="append", default=[])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--unique-candidates", action="store_true")
    parser.add_argument("--unique-symbols", action="store_true")
    parser.add_argument("--regime-grid", choices=["compact", "wide"], default="compact")
    parser.add_argument("--min-train-trades", type=int, default=10)
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-pct", type=float, default=1.0)
    parser.add_argument("--sizing-mode", choices=["risk_based", "fixed_notional"], default="risk_based")
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0020)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def walk_forward_windows(start: str, end: str, train_months: int, test_months: int):
    train_start = pd.Timestamp(start, tz="UTC")
    final_end = pd.Timestamp(end, tz="UTC")
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > final_end:
            break
        yield train_start, train_end, test_start, test_end
        train_start = train_start + pd.DateOffset(months=test_months)


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.candidate)
    for pattern in args.candidate_glob:
        paths.extend(PROJECT_ROOT.glob(pattern))
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        unique[str(resolved)] = resolved
    return sorted(unique.values())


def q_name(value: float) -> str:
    return str(int(round(value * 100)))


def regime_configs(grid: str) -> list[RegimeConfig]:
    configs = [RegimeConfig(name="none", mode="none")]
    sma_values = [100, 200] if grid == "compact" else [50, 100, 200]
    roc_values = [14, 30]
    roc_mins = [0.0] if grid == "compact" else [0.0, 0.02]
    vol_quantiles = [0.50, 0.70, 0.85]

    for sma in sma_values:
        for roc_days in roc_values:
            for roc_min in roc_mins:
                suffix = f"sma{sma}_roc{roc_days}_min{int(roc_min * 100)}"
                configs.append(RegimeConfig(f"trend_aligned_{suffix}", "trend_aligned", sma, roc_days, roc_min))
                configs.append(RegimeConfig(f"long_bull_{suffix}", "long_bull", sma, roc_days, roc_min))
                configs.append(RegimeConfig(f"short_bear_{suffix}", "short_bear", sma, roc_days, roc_min))

    for quantile in vol_quantiles:
        configs.append(RegimeConfig(f"low_vol_q{q_name(quantile)}", "low_vol", vol_quantile=quantile))
        configs.append(RegimeConfig(f"high_vol_q{q_name(quantile)}", "high_vol", vol_quantile=quantile))

    for sma in sma_values:
        configs.append(RegimeConfig(f"long_bull_low_vol_sma{sma}", "long_bull_low_vol", sma=sma, roc_days=30))
        configs.append(RegimeConfig(f"short_bear_high_vol_sma{sma}", "short_bear_high_vol", sma=sma))

    return configs


def load_btc_daily_regime(start: str, end: str) -> pd.DataFrame:
    loader = DataLoader(use_cache=True)
    raw = loader.load_symbol("BTCUSDT", "1m")
    if raw.index.tz is None:
        raw = raw.tz_localize("UTC")
    start_ts = pd.Timestamp(start, tz="UTC") - pd.DateOffset(days=420)
    end_ts = pd.Timestamp(end, tz="UTC")
    raw = raw[(raw.index >= start_ts) & (raw.index <= end_ts)]
    daily = vectorized_preprocess(raw, "1d")
    close = daily["close"]
    daily["btc_daily_vol_20"] = daily["log_return"].rolling(20, min_periods=5).std() * np.sqrt(365)
    for sma in [50, 100, 200]:
        daily[f"btc_sma_{sma}"] = close.rolling(sma, min_periods=max(20, sma // 4)).mean()
    for roc_days in [14, 30]:
        daily[f"btc_roc_{roc_days}"] = close.pct_change(roc_days)
    for quantile in [0.50, 0.70, 0.85]:
        daily[f"btc_vol_q_120_{q_name(quantile)}"] = daily["btc_daily_vol_20"].rolling(
            120,
            min_periods=30,
        ).quantile(quantile)

    regime = daily[
        ["close", "btc_daily_vol_20"]
        + [f"btc_sma_{sma}" for sma in [50, 100, 200]]
        + [f"btc_roc_{days}" for days in [14, 30]]
        + [f"btc_vol_q_120_{q_name(q)}" for q in [0.50, 0.70, 0.85]]
    ].rename(columns={"close": "btc_close"})

    # Daily bars are left-labeled after resampling, so shift by one completed day
    # before forward-filling to intraday candidate bars.
    return regime.shift(1).dropna(how="all")


def align_regime(regime_daily: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = regime_daily.reindex(index, method="ffill")
    return aligned.ffill()


def filter_signal(signal: pd.Series, regime: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    if cfg.mode == "none":
        return signal.astype(int)

    sma = regime[f"btc_sma_{cfg.sma}"]
    roc = regime[f"btc_roc_{cfg.roc_days}"]
    vol_threshold = regime[f"btc_vol_q_{cfg.vol_lookback}_{q_name(cfg.vol_quantile)}"]
    bull = (regime["btc_close"] > sma) & (roc > cfg.roc_min)
    bear = (regime["btc_close"] < sma) & (roc < -cfg.roc_min)
    low_vol = regime["btc_daily_vol_20"] <= vol_threshold
    high_vol = regime["btc_daily_vol_20"] >= vol_threshold

    long_allowed = pd.Series(True, index=signal.index)
    short_allowed = pd.Series(True, index=signal.index)
    if cfg.mode == "trend_aligned":
        long_allowed = bull
        short_allowed = bear
    elif cfg.mode == "long_bull":
        long_allowed = bull
        short_allowed = pd.Series(False, index=signal.index)
    elif cfg.mode == "short_bear":
        long_allowed = pd.Series(False, index=signal.index)
        short_allowed = bear
    elif cfg.mode == "low_vol":
        long_allowed = low_vol
        short_allowed = low_vol
    elif cfg.mode == "high_vol":
        long_allowed = high_vol
        short_allowed = high_vol
    elif cfg.mode == "long_bull_low_vol":
        long_allowed = bull & low_vol
        short_allowed = pd.Series(False, index=signal.index)
    elif cfg.mode == "short_bear_high_vol":
        long_allowed = pd.Series(False, index=signal.index)
        short_allowed = bear & high_vol
    else:
        raise ValueError(f"Unknown regime mode: {cfg.mode}")

    filtered = signal.copy()
    filtered.loc[(filtered > 0) & (~long_allowed)] = 0
    filtered.loc[(filtered < 0) & (~short_allowed)] = 0
    return filtered.astype(int)


def precompute(
    candidates: list[Candidate],
    configs: list[RegimeConfig],
    args: argparse.Namespace,
) -> tuple[list[Variant], dict[str, pd.DataFrame], dict[str, pd.Series]]:
    data_cache: dict[tuple[str, str], pd.DataFrame] = {}
    feature_cache: dict[str, pd.DataFrame] = {}
    signal_cache: dict[str, pd.Series] = {}
    variants: list[Variant] = []
    regime_daily = load_btc_daily_regime(args.start, args.end)

    for candidate in candidates:
        base = load_base_data(candidate, args.start, args.end, data_cache)
        featured = apply_candidate_strategy(base, candidate, args.risk_per_trade)
        feature_cache[candidate.label] = featured
        aligned_regime = align_regime(regime_daily, featured.index)
        for cfg in configs:
            label = f"{candidate.label}__{cfg.name}"
            variants.append(Variant(label=label, candidate=candidate, regime=cfg))
            signal_cache[label] = filter_signal(featured["signal"], aligned_regime, cfg)

    return variants, feature_cache, signal_cache


def train_score(metrics: dict[str, Any], min_trades: int) -> float:
    trades = int(metrics.get("total_trades", 0))
    if trades < min_trades:
        return -1_000_000 + trades
    exp_r = float(metrics.get("expectancy_r", 0))
    pf = min(float(metrics.get("profit_factor", 0)), 4.0)
    dd_r = float(metrics.get("max_drawdown_r", 0))
    loss_streak = float(metrics.get("max_consecutive_losses", 0))
    return exp_r * 150 + (pf - 1.0) * 28 - dd_r * 2.4 - loss_streak * 1.8


def candidate_for_variant(variant: Variant) -> Candidate:
    candidate = variant.candidate
    return Candidate(
        label=variant.label,
        path=candidate.path,
        symbol=candidate.symbol,
        timeframe=candidate.timeframe,
        kind=f"{candidate.kind}|regime:{variant.regime.name}",
        params=candidate.params,
    )


def evaluate_variant(
    variant: Variant,
    feature_cache: dict[str, pd.DataFrame],
    signal_cache: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[Any], pd.DatetimeIndex]:
    featured = feature_cache[variant.candidate.label]
    window = featured.loc[(featured.index >= start) & (featured.index <= end)].copy()
    if window.empty:
        return metrics_dict([], pd.DatetimeIndex([]), args.balance), [], pd.DatetimeIndex([])
    window["signal"] = signal_cache[variant.label].loc[window.index]
    candidate = candidate_for_variant(variant)
    result = run_candidate(
        candidate,
        window,
        args.balance,
        args.risk_per_trade,
        args.fee_rate,
        args.slippage_rate,
        args.max_position_pct,
        args.sizing_mode,
    )
    return metrics_dict([scale_trade(trade, candidate, 1.0) for trade in result.get("trades", [])], window.index, args.balance), result.get("trades", []), window.index


def evaluate_selected_portfolio(
    selected: list[Variant],
    feature_cache: dict[str, pd.DataFrame],
    signal_cache: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_trades = []
    timeline = pd.DatetimeIndex([])
    weight = 1.0 / len(selected)
    for variant in selected:
        featured = feature_cache[variant.candidate.label]
        window = featured.loc[(featured.index >= start) & (featured.index <= end)].copy()
        if window.empty:
            continue
        window["signal"] = signal_cache[variant.label].loc[window.index]
        timeline = timeline.union(window.index)
        candidate = candidate_for_variant(variant)
        result = run_candidate(
            candidate,
            window,
            args.balance,
            args.risk_per_trade,
            args.fee_rate,
            args.slippage_rate,
            args.max_position_pct,
            args.sizing_mode,
        )
        all_trades.extend(scale_trade(trade, candidate, weight) for trade in result.get("trades", []))
    return metrics_dict(all_trades, timeline, args.balance)


def select_top(
    scored: list[tuple[float, Variant, dict[str, Any]]],
    top_k: int,
    unique_candidates: bool,
    unique_symbols: bool,
) -> list[Variant]:
    if not unique_candidates and not unique_symbols:
        return [variant for _, variant, _ in scored[:top_k]]

    selected: list[Variant] = []
    seen_candidates: set[str] = set()
    seen_symbols: set[str] = set()
    for _, variant, _ in scored:
        candidate_label = variant.candidate.label
        symbol = variant.candidate.symbol
        if unique_candidates and candidate_label in seen_candidates:
            continue
        if unique_symbols and symbol in seen_symbols:
            continue
        selected.append(variant)
        seen_candidates.add(candidate_label)
        seen_symbols.add(symbol)
        if len(selected) >= top_k:
            return selected
    for _, variant, _ in scored:
        if variant in selected:
            continue
        selected.append(variant)
        if len(selected) >= top_k:
            break
    return selected


def write_rankings(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("backtest_pipeline").setLevel(logging.ERROR)
    logging.getLogger("data.loader").setLevel(logging.WARNING)

    paths = candidate_paths(args)
    if not paths:
        raise SystemExit("No candidates supplied.")
    candidates = [load_candidate(path) for path in paths]
    configs = regime_configs(args.regime_grid)

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    unique_suffix = "_unique" if args.unique_symbols else ""
    run_dir = output_root / f"regime_gated_wf_top{args.top_k}{unique_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Precomputing %d candidates across %d regime gates", len(candidates), len(configs))
    variants, feature_cache, signal_cache = precompute(candidates, configs, args)
    logger.info("Built %d candidate variants", len(variants))

    window_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()

    windows = list(walk_forward_windows(args.start, args.end, args.train_months, args.test_months))
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        scored = []
        for variant in variants:
            train_metrics, _, _ = evaluate_variant(variant, feature_cache, signal_cache, train_start, train_end, args)
            score = train_score(train_metrics, args.min_train_trades)
            scored.append((score, variant, train_metrics))
            ranking_rows.append({
                "window": i,
                "variant": variant.label,
                "candidate": variant.candidate.label,
                "symbol": variant.candidate.symbol,
                "timeframe": variant.candidate.timeframe,
                "kind": variant.candidate.kind,
                "regime": variant.regime.name,
                "regime_mode": variant.regime.mode,
                "train_score": score,
                "train_total_trades": train_metrics.get("total_trades", 0),
                "train_profit_factor": train_metrics.get("profit_factor", 0),
                "train_expectancy_r": train_metrics.get("expectancy_r", 0),
                "train_max_drawdown_r": train_metrics.get("max_drawdown_r", 0),
            })

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = select_top(scored, args.top_k, args.unique_candidates, args.unique_symbols)
        for variant in selected:
            selected_counts[variant.label] += 1
        test_metrics = evaluate_selected_portfolio(selected, feature_cache, signal_cache, test_start, test_end, args)
        window_rows.append({
            "window": i,
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{train_end:%Y-%m-%d}",
            "test_start": f"{test_start:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            "selected": "|".join(variant.label for variant in selected),
            **test_metrics,
        })
        logger.info(
            "  WF %02d selected %s | trades %s PF %.2f expR %.3f ddR %.2f",
            i,
            ", ".join(variant.label for variant in selected),
            test_metrics.get("total_trades", 0),
            test_metrics.get("profit_factor", 0),
            test_metrics.get("expectancy_r", 0),
            test_metrics.get("max_drawdown_r", 0),
        )

    aggregate = summarize_windows(window_rows)
    write_csv(run_dir / "walk_forward.csv", window_rows)
    write_rankings(run_dir / "variant_rankings.csv", ranking_rows)
    summary = {
        "symbol": "REGIME_GATED_PORTFOLIO",
        "timeframe": "mixed",
        "strategy": "regime_gated_adaptive_selector",
        "aggregate": aggregate,
        "selection_counts": dict(selected_counts.most_common()),
        "candidate_count": len(candidates),
        "variant_count": len(variants),
        "regime_grid": args.regime_grid,
        "regimes": [asdict(config) for config in configs],
        "top_k": args.top_k,
        "unique_candidates": args.unique_candidates,
        "unique_symbols": args.unique_symbols,
        "candidates": [
            {
                "label": candidate.label,
                "path": str(candidate.path),
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "kind": candidate.kind,
            }
            for candidate in candidates
        ],
        "costs": {
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "risk_per_trade": args.risk_per_trade,
            "max_position_pct": args.max_position_pct,
            "sizing_mode": args.sizing_mode,
        },
        "period": {
            "start": args.start,
            "end": args.end,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "min_train_trades": args.min_train_trades,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    logger.info(
        "Regime-gated WF saved to %s | %s/%s positive exp, mean expR %.3f, ddR %.2f",
        run_dir,
        aggregate.get("positive_expectancy_windows", 0),
        aggregate.get("windows", 0),
        aggregate.get("mean_expectancy_r", 0),
        aggregate.get("max_drawdown_r", 0),
    )


if __name__ == "__main__":
    main()
