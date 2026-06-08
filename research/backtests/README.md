# CryptoEA Backtest Research Runs

This directory stores reproducible backtest run bundles and the static dashboard generated from them.

## Reproduce the Verified Full-Period Runs

```bash
.venv/bin/python scripts/run_research_suite.py
```

The suite runs the four v3 cases from the claim:

| Run | Symbol | Timeframe | Strategy | Period |
|---|---|---:|---|---|
| `full_BTCUSDT_1h_v3_20200101_20250601` | BTCUSDT | 1h | v3 | 2020-01-01 to 2025-06-01 |
| `full_ETHUSDT_1h_v3_20200101_20250601` | ETHUSDT | 1h | v3 | 2020-01-01 to 2025-06-01 |
| `full_SOLUSDT_1h_v3_20200101_20250601` | SOLUSDT | 1h | v3 | 2020-01-01 to 2025-06-01 |
| `full_BTCUSDT_4h_v3_20200101_20250601` | BTCUSDT | 4h | v3 | 2020-01-01 to 2025-06-01 |

Each run writes:

- `summary.json`: command, code SHA, data profile, environment, config, and metrics
- `report.txt`: human-readable performance report
- `trades.csv`: every completed trade
- `equity.csv`: realized-PnL equity curve
- `benchmark.csv`: buy-and-hold benchmark curve

## Walk-Forward Slice Check

```bash
.venv/bin/python scripts/run_research_suite.py --skip-full --walk-forward
```

This runs fixed-parameter out-of-sample slices for BTCUSDT 1h v3 using 12-month train labels and 3-month test windows. It is not parameter-optimizing on each training window; it is an OOS stability check for the existing parameter set.

## Dashboard

```bash
.venv/bin/python scripts/build_dashboard.py --root research/backtests
```

Open `research/backtests/dashboard.html` in a browser to inspect all tracked runs.
