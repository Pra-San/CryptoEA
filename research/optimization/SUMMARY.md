# Strategy Exploration Summary

Generated on 2026-06-08.

## Corrections Applied Before Optimization

- Removed lookahead from the vectorized engine by entering on the next bar after a close-derived signal.
- Prevented overlapping trades in the single-position vectorized simulation.
- Added risk-based sizing from initial stop distance, including fees and adverse stop fill.
- Added per-trade R multiples and cumulative R drawdown metrics.
- Made flat-signal exits optional; v3 breakout signals now hold until stop, target, opposite signal, or max holding period by default.
- Added v3 side filters for long-only, short-only, and bidirectional testing.

## Main Finding

The original v3 claim was not valid after removing lookahead and correcting execution. The 1h v3 baseline became negative expectancy under realistic costs.

The strongest corrected v3 BTCUSDT 4h candidate found in the broad random search was:

- Donchian 8, volume lookback 50, volume multiple 1.2561
- ATR stop 0.6828, ATR target 10.8575
- Min hold 3 bars, max hold 24 bars
- Trend filter disabled, both long and short enabled

Holdout validation from 2024-01-01 to 2025-06-01 under 10 bps fee + 20 bps adverse execution:

- Profit factor: 1.27
- Expectancy: 0.249R
- Max drawdown: 8.25R
- Total return: 20.98%

That is not strong enough for live deployment.

## Walk-Forward Result

The best high-holdout candidate from walk-forward-trained search reached:

- Holdout PF: 1.91
- Holdout expectancy: 0.50R
- Holdout max drawdown: 8.87R

But fixed-parameter 12-month train / 3-month test walk-forward at 10 bps fee + 20 bps adverse execution produced:

- Profitable windows: 10 / 17
- Mean expectancy: 0.215R
- Worst expectancy: -0.648R
- Max drawdown: 9.21R

This remains a research candidate, not a deployable edge.

## Deployment Status

No strategy in this exploration should be deployed live yet. The next useful work is to test a new strategy family or add market-regime filters, not to tune the current v3 parameter space harder.
