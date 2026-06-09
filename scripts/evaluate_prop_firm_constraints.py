#!/usr/bin/env python3
"""Evaluate strategy candidates against prop-firm style constraints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.explore_trailing_highwin_edge import PARAM_FIELDS as PARTIAL_FIELDS
from scripts.explore_trailing_highwin_edge import apply_strategy as apply_partial_strategy
from scripts.explore_trailing_highwin_edge import load_data, simulate_trades, write_csv
from scripts.explore_vwap_volume_profile_edge import PARAM_FIELDS as VWAP_FIELDS
from scripts.explore_vwap_volume_profile_edge import apply_strategy as apply_vwap_strategy
from scripts.explore_vwap_volume_profile_edge import prepare_features as prepare_vwap_features


BOOL_FIELDS = {
    "allow_long",
    "allow_short",
    "exit_on_flat_signal",
    "move_stop_after_partial",
    "use_break_even",
    "use_mid_target",
    "use_poc_target",
    "use_partial_exit",
    "use_trailing_stop",
}

INT_FIELDS = {
    "bb_lookback",
    "donchian_lookback",
    "ema_fast",
    "ema_slow",
    "max_holding_bars",
    "min_holding_bars",
    "mom_lookback",
    "profile_lookback",
    "slope_bars",
    "volume_lookback",
    "vwap_lookback",
}

STRING_FIELDS = {"family", "target_mode"}


@dataclass(frozen=True)
class PropFirmProfile:
    name: str
    profit_target_pct: float
    max_daily_loss_pct: float
    max_total_loss_pct: float
    min_trading_days: int
    consistency_pct: float | None = None


PROFILES = {
    "ftmo_2step_phase1": PropFirmProfile("ftmo_2step_phase1", 0.10, 0.05, 0.10, 4, 0.50),
    "ftmo_2step_phase2": PropFirmProfile("ftmo_2step_phase2", 0.05, 0.05, 0.10, 4, 0.50),
    "ftmo_1step": PropFirmProfile("ftmo_1step", 0.10, 0.03, 0.10, 0, 0.50),
    "the5ers_high_stakes": PropFirmProfile("the5ers_high_stakes", 0.08, 0.05, 0.10, 3, None),
    "fundednext_stellar_2step_p1": PropFirmProfile("fundednext_stellar_2step_p1", 0.08, 0.05, 0.10, 5, None),
    "fundednext_stellar_2step_p2": PropFirmProfile("fundednext_stellar_2step_p2", 0.05, 0.05, 0.10, 5, None),
    "fundednext_stellar_1step": PropFirmProfile("fundednext_stellar_1step", 0.10, 0.03, 0.06, 2, None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate against prop-firm constraints")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-section", default="best", choices=["best", "best_validation_ranked"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--profile", action="append", choices=sorted(PROFILES), default=[])
    parser.add_argument("--balance", type=float, default=100000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0010)
    parser.add_argument("--risk-grid", default="0.001,0.0015,0.002,0.0025,0.003,0.004,0.005,0.0075,0.01")
    parser.add_argument("--day-boundary-hour-utc", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research" / "optimization")
    return parser.parse_args()


def coerce_params(raw: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in fields:
        if field not in raw:
            continue
        value = raw[field]
        if field in BOOL_FIELDS:
            if isinstance(value, str):
                params[field] = value.strip().lower() == "true"
            else:
                params[field] = bool(value)
        elif field in INT_FIELDS:
            params[field] = int(value)
        elif field in STRING_FIELDS:
            params[field] = str(value)
        else:
            params[field] = float(value)
    return params


def load_candidate(path: Path, section: str) -> tuple[str, str, str, dict[str, Any]]:
    data = json.loads(path.read_text())
    best = data.get(section) or data["best"]
    strategy = str(data.get("strategy", ""))
    fields = VWAP_FIELDS if strategy == "vwap_volume_profile_momentum_search" else PARTIAL_FIELDS
    candidate_name = f"{section}:{best.get('family', best.get('strategy', 'candidate'))}"
    return str(data["symbol"]), str(data["timeframe"]), strategy, candidate_name, coerce_params(best, fields)


def day_key(ts: Any, boundary_hour_utc: int) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    shifted = stamp - pd.Timedelta(hours=boundary_hour_utc)
    return shifted.floor("D")


def evaluate_trades(
    trades: list[Any],
    profile: PropFirmProfile,
    balance: float,
    boundary_hour_utc: int,
) -> dict[str, Any]:
    equity = balance
    start_balance_by_day: dict[pd.Timestamp, float] = {}
    daily_pnl: dict[pd.Timestamp, float] = {}
    trading_days: set[pd.Timestamp] = set()
    min_equity = balance
    max_daily_loss = 0.0
    pass_time = None
    fail_reason = ""
    target = balance * (1.0 + profile.profit_target_pct)
    total_loss_floor = balance * (1.0 - profile.max_total_loss_pct)
    daily_loss_amount = balance * profile.max_daily_loss_pct

    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade.exit_time))
    for trade in ordered:
        day = day_key(trade.exit_time, boundary_hour_utc)
        trading_days.add(day)
        start_balance_by_day.setdefault(day, equity)
        equity += float(trade.pnl)
        daily_pnl[day] = daily_pnl.get(day, 0.0) + float(trade.pnl)
        min_equity = min(min_equity, equity)

        intraday_loss = min(0.0, equity - start_balance_by_day[day])
        max_daily_loss = min(max_daily_loss, intraday_loss)
        daily_floor = start_balance_by_day[day] - daily_loss_amount
        if equity < daily_floor:
            fail_reason = "daily_loss"
            break
        if equity < total_loss_floor:
            fail_reason = "total_loss"
            break

        positive_days = [value for value in daily_pnl.values() if value > 0]
        best_day = max(positive_days) if positive_days else 0.0
        positive_sum = sum(positive_days)
        consistency = best_day / positive_sum if positive_sum > 0 else 0.0
        consistency_ok = profile.consistency_pct is None or consistency < profile.consistency_pct
        if equity >= target and len(trading_days) >= profile.min_trading_days and consistency_ok:
            pass_time = pd.Timestamp(trade.exit_time).isoformat()
            break

    positive_days = [value for value in daily_pnl.values() if value > 0]
    best_day = max(positive_days) if positive_days else 0.0
    positive_sum = sum(positive_days)
    consistency = best_day / positive_sum if positive_sum > 0 else 0.0
    return {
        "profile": profile.name,
        "passed": pass_time is not None,
        "pass_time": pass_time,
        "fail_reason": fail_reason,
        "final_equity": equity,
        "return_pct_until_stop": (equity - balance) / balance * 100.0,
        "min_equity_pct": (min_equity - balance) / balance * 100.0,
        "max_daily_loss_pct": max_daily_loss / balance * 100.0,
        "trading_days": len(trading_days),
        "positive_days": len(positive_days),
        "best_day_profit": best_day,
        "positive_days_profit": positive_sum,
        "best_day_pct_of_positive_profit": consistency,
        "target_pct": profile.profit_target_pct * 100.0,
        "daily_loss_limit_pct": profile.max_daily_loss_pct * 100.0,
        "total_loss_limit_pct": profile.max_total_loss_pct * 100.0,
    }


def main() -> None:
    args = parse_args()
    profiles = [PROFILES[name] for name in args.profile] if args.profile else list(PROFILES.values())
    risk_values = [float(item) for item in args.risk_grid.split(",") if item.strip()]
    symbol, timeframe, strategy, candidate_name, params = load_candidate(args.candidate_summary, args.candidate_section)
    base = load_data(symbol, timeframe, args.start, args.end)
    if strategy == "vwap_volume_profile_momentum_search":
        featured = apply_vwap_strategy(prepare_vwap_features(base, params, {}), params)
    else:
        featured = apply_partial_strategy(base, params)
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    fee_bps = int(round(args.fee_rate * 10_000))
    slip_bps = int(round(args.slippage_rate * 10_000))
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / (
        f"propfirm_eval_{symbol}_{timeframe}_{args.candidate_section}_fee{fee_bps}bps_slip{slip_bps}bps_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for risk in risk_values:
        runtime_args = SimpleNamespace(
            balance=args.balance,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            risk_per_trade=risk,
            max_position_pct=1.0,
        )
        trades = simulate_trades(featured, symbol, runtime_args, params)
        for profile in profiles:
            rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "risk_per_trade": risk,
                "fee_rate": args.fee_rate,
                "slippage_rate": args.slippage_rate,
                "trade_count": len(trades),
                **evaluate_trades(trades, profile, args.balance, args.day_boundary_hour_utc),
            })

    summary = {
        "candidate_summary": str(args.candidate_summary),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "candidate_section": args.candidate_section,
        "candidate_name": candidate_name,
        "start": args.start,
        "end": args.end,
        "profiles": [asdict(profile) for profile in profiles],
        "risk_grid": risk_values,
        "costs": {"fee_rate": args.fee_rate, "slippage_rate": args.slippage_rate},
        "day_boundary_hour_utc": args.day_boundary_hour_utc,
        "best_passes": [
            row
            for row in rows
            if row["passed"]
            and row["risk_per_trade"] == min(r["risk_per_trade"] for r in rows if r["profile"] == row["profile"] and r["passed"])
        ],
    }
    write_csv(run_dir / "propfirm_results.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Saved to {run_dir}")
    for row in rows:
        if row["passed"]:
            print(
                f"{row['profile']} risk {row['risk_per_trade']:.4f} PASS "
                f"return {row['return_pct_until_stop']:.2f}% min_eq {row['min_equity_pct']:.2f}% "
                f"daily {row['max_daily_loss_pct']:.2f}% consistency {row['best_day_pct_of_positive_profit']:.2f}"
            )


if __name__ == "__main__":
    main()
