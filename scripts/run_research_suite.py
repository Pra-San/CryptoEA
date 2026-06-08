#!/usr/bin/env python3
"""Run the reproducible CryptoEA backtest research suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "run_backtest.py"
DASHBOARD_SCRIPT = PROJECT_ROOT / "scripts" / "build_dashboard.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "research" / "backtests"


@dataclass(frozen=True)
class BacktestCase:
    symbol: str
    timeframe: str
    strategy: str
    start: str
    end: str
    run_id: str


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_rate: float
    slippage_rate: float


FULL_PERIOD_CASES = (
    BacktestCase("BTCUSDT", "1h", "v3", "2020-01-01", "2025-06-01", "full_BTCUSDT_1h_v3_20200101_20250601"),
    BacktestCase("ETHUSDT", "1h", "v3", "2020-01-01", "2025-06-01", "full_ETHUSDT_1h_v3_20200101_20250601"),
    BacktestCase("SOLUSDT", "1h", "v3", "2020-01-01", "2025-06-01", "full_SOLUSDT_1h_v3_20200101_20250601"),
    BacktestCase("BTCUSDT", "4h", "v3", "2020-01-01", "2025-06-01", "full_BTCUSDT_4h_v3_20200101_20250601"),
)


STRESS_SCENARIOS = (
    CostScenario("binance_spot_vip0_10bps_fee_3bps_exec", 0.0010, 0.0003),
    CostScenario("vip0_10bps_fee_10bps_exec", 0.0010, 0.0010),
    CostScenario("vip0_10bps_fee_20bps_exec", 0.0010, 0.0020),
)


def run_command(args: list[str], dry_run: bool = False) -> None:
    """Run a command from the project root."""
    print(" ".join(args))
    if dry_run:
        return
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def run_case(
    case: BacktestCase,
    output_dir: Path,
    dry_run: bool = False,
    *,
    run_id: str | None = None,
    validation_mode: str = "full_period",
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
) -> None:
    """Run one deterministic full-period case."""
    command = [
        sys.executable,
        str(BACKTEST_SCRIPT),
        "--symbol",
        case.symbol,
        "--timeframe",
        case.timeframe,
        "--strategy",
        case.strategy,
        "--start",
        case.start,
        "--end",
        case.end,
        "--no-plot",
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id or case.run_id,
        "--validation-mode",
        validation_mode,
    ]
    if fee_rate is not None:
        command.extend(["--fee-rate", str(fee_rate)])
    if slippage_rate is not None:
        command.extend(["--slippage-rate", str(slippage_rate)])
    run_command(command, dry_run=dry_run)


def run_stress_cases(output_dir: Path, dry_run: bool = False) -> None:
    """Run full-period deployment cost stress cases."""
    for scenario in STRESS_SCENARIOS:
        for case in FULL_PERIOD_CASES:
            run_case(
                case,
                output_dir=output_dir,
                dry_run=dry_run,
                run_id=f"stress_{scenario.name}_{case.symbol}_{case.timeframe}_{case.strategy}_{case.start.replace('-', '')}_{case.end.replace('-', '')}",
                validation_mode="cost_stress",
                fee_rate=scenario.fee_rate,
                slippage_rate=scenario.slippage_rate,
            )


def walk_forward_windows(start: str, end: str, train_months: int, test_months: int):
    """Yield rolling train/test windows for fixed-parameter OOS checks."""
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


def run_walk_forward_case(
    case: BacktestCase,
    output_dir: Path,
    train_months: int,
    test_months: int,
    dry_run: bool = False,
    scenario: CostScenario | None = None,
) -> None:
    """Run fixed-parameter walk-forward test slices for one case."""
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_windows(case.start, case.end, train_months, test_months),
        start=1,
    ):
        base_run_id = (
            f"wf{i:02d}_{case.symbol}_{case.timeframe}_{case.strategy}_"
            f"{test_start:%Y%m%d}_{test_end:%Y%m%d}"
        )
        run_id = f"stress_{scenario.name}_{base_run_id}" if scenario else base_run_id
        command = [
            sys.executable,
            str(BACKTEST_SCRIPT),
            "--symbol",
            case.symbol,
            "--timeframe",
            case.timeframe,
            "--strategy",
            case.strategy,
            "--start",
            f"{test_start:%Y-%m-%d}",
            "--end",
            f"{test_end:%Y-%m-%d}",
            "--no-plot",
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--validation-mode",
            "stress_walk_forward" if scenario else "walk_forward",
            "--train-start",
            f"{train_start:%Y-%m-%d}",
            "--train-end",
            f"{train_end:%Y-%m-%d}",
            "--test-start",
            f"{test_start:%Y-%m-%d}",
            "--test-end",
            f"{test_end:%Y-%m-%d}",
        ]
        if scenario:
            command.extend(["--fee-rate", str(scenario.fee_rate), "--slippage-rate", str(scenario.slippage_rate)])
        run_command(command, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible backtest suite")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--stress", action="store_true", help="Run deployment cost stress cases")
    parser.add_argument(
        "--stress-walk-forward",
        action="store_true",
        help="Run deployment cost stress cases on walk-forward test slices",
    )
    parser.add_argument("--walk-symbol", default="BTCUSDT")
    parser.add_argument("--walk-timeframe", default="1h")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    if not args.skip_full:
        for case in FULL_PERIOD_CASES:
            run_case(case, output_dir=output_dir, dry_run=args.dry_run)

    if args.stress:
        run_stress_cases(output_dir=output_dir, dry_run=args.dry_run)

    if args.walk_forward:
        walk_cases = [
            case
            for case in FULL_PERIOD_CASES
            if case.symbol == args.walk_symbol and case.timeframe == args.walk_timeframe
        ]
        if not walk_cases:
            raise ValueError(f"No full-period case for {args.walk_symbol} {args.walk_timeframe}")
        for case in walk_cases:
            run_walk_forward_case(
                case,
                output_dir=output_dir,
                train_months=args.train_months,
                test_months=args.test_months,
                dry_run=args.dry_run,
            )

    if args.stress_walk_forward:
        walk_cases = [
            case
            for case in FULL_PERIOD_CASES
            if case.symbol == args.walk_symbol and case.timeframe == args.walk_timeframe
        ]
        if not walk_cases:
            raise ValueError(f"No full-period case for {args.walk_symbol} {args.walk_timeframe}")
        for case in walk_cases:
            for scenario in STRESS_SCENARIOS:
                run_walk_forward_case(
                    case,
                    output_dir=output_dir,
                    train_months=args.train_months,
                    test_months=args.test_months,
                    dry_run=args.dry_run,
                    scenario=scenario,
                )

    run_command(
        [
            sys.executable,
            str(DASHBOARD_SCRIPT),
            "--root",
            str(output_dir),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
