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

## Second-Pass Family Exploration

Additional work on 2026-06-08 expanded the search beyond v3 into five simple, auditable strategy families:

- Trend pullback
- Channel breakout
- Volatility-squeeze breakout
- Range reversion
- EMA momentum

The search also fixed a vectorized-engine signal handling bug where direct `-1 -> +1` or `+1 -> -1` flips were not counted as fresh entry events.

Best strict BTCUSDT 4h holdout candidate:

- Family: channel breakout
- Stress costs: 10 bps fee + 20 bps adverse execution
- Stress PF: 1.85
- Stress expectancy: 0.366R
- Stress max drawdown: 5.44R
- Stress trades: 33

Fixed 12-month train / 3-month test walk-forward for that candidate failed:

- Profitable windows: 5 / 17
- Mean expectancy: -0.077R
- Trade-weighted expectancy: 0.019R
- Max drawdown: 4.00R

Best SOLUSDT 4h family candidate was more stable but too weak:

- Fixed walk-forward profitable windows: 9 / 17
- Mean expectancy: 0.048R
- Trade-weighted expectancy: 0.044R
- Max drawdown: 2.10R

Walk-forward-trained family search also failed to produce a deployable candidate:

- BTCUSDT 4h best validation PF: 1.15, validation expectancy: 0.057R, train-WF positive windows: 4 / 11
- SOLUSDT 4h best validation PF: 0.88, validation expectancy: -0.030R, train-WF positive windows: 5 / 11

Optimization dashboard:

- `research/optimization/dashboard.html`
- `research/optimization/optimization_index.csv`

Conclusion remains unchanged: no candidate found so far is live-deployment worthy under corrected execution, Binance-style commissions, 20 bps adverse execution stress, and walk-forward validation.

## Third-Pass Portfolio Evaluation

Portfolio-level testing was added after the single-symbol family search. The evaluator scales each saved candidate into an equal-weight sleeve and scales per-trade R by sleeve weight, so portfolio drawdown in R reflects account-level risk contribution.

Representative stressed portfolio results:

| Portfolio | Validation PF | Validation Exp R | Validation DD R | Trades | WF Positive Windows | WF Mean Exp R |
|---|---:|---:|---:|---:|---:|---:|
| BTC/ETH/SOL family candidates | 1.50 | 0.085 | 4.40 | 111 | 7 / 17 | 0.019 |
| BTC/ETH/SOL v3 candidates | 1.26 | 0.051 | 3.49 | 248 | 12 / 17 | 0.038 |
| BTC v3 + SOL family | 1.29 | 0.103 | 4.32 | 125 | 10 / 17 | 0.032 |
| BTC family + SOL family | 1.72 | 0.114 | 3.08 | 70 | 9 / 17 | 0.011 |

Portfolio diversification reduced R drawdowns in some cases, but it did not produce enough expectancy. The best window count was the BTC/ETH/SOL v3 portfolio at 12 / 17 positive-expectancy windows, but mean expectancy was only 0.038R. This is still too weak for live deployment.

## Fourth-Pass Adaptive Walk-Forward Selection

An adaptive selector was added to test whether choosing candidates from the prior 12-month training window improves robustness. Each 3-month out-of-sample window ranks all saved candidates using only the preceding training period, then trades the top K candidates as an equal-weight portfolio.

All runs below use 10 bps commission, 20 bps adverse execution stress, 1% account risk per trade, and report max drawdown in R.

| Adaptive Run | Positive Exp Windows | Mean Exp R | Trade-Weighted Exp R | Max DD R | Trades |
|---|---:|---:|---:|---:|---:|
| Top 1 candidate | 6 / 17 | -0.229 | 0.003 | 9.48 | 113 |
| Top 2 candidates | 8 / 17 | 0.006 | 0.024 | 6.79 | 247 |
| Top 3 candidates | 7 / 17 | -0.012 | 0.000 | 7.95 | 375 |
| Top 2, unique symbols | 6 / 17 | -0.034 | -0.022 | 6.93 | 267 |
| Top 3, unique symbols | 6 / 17 | -0.011 | -0.018 | 5.18 | 410 |

Adaptive selection did not create a deployable edge. The best variant by mean expectancy was top-2 unrestricted selection, but its average edge was effectively zero and only 8 of 17 out-of-sample windows were positive. The unique-symbol constraint reduced concentration risk but did not improve expectancy.

Conclusion remains unchanged: this OHLCV-only candidate universe does not currently contain an edge strong enough for live deployment after realistic costs, spreads/slippage stress, R drawdown accounting, and walk-forward testing.

## Fifth-Pass Regime-Gated Candidate Selection

A BTC daily regime-gating evaluator was added after adaptive candidate selection failed. It expands each saved candidate with causal BTC trend and volatility gates, shifts daily regime features by one completed day to avoid lookahead, ranks variants only on the prior 12-month training window, and evaluates the selected variants on the next 3-month window.

All runs below use 10 bps commission, 20 bps adverse execution stress, 1% account risk per trade, and report drawdown in R.

| Regime-Gated Run | Positive Exp Windows | Mean Exp R | Trade-Weighted Exp R | Max DD R | Trades |
|---|---:|---:|---:|---:|---:|
| Top 2 variants | 7 / 17 | 0.056 | 0.107 | 6.87 | 99 |
| Top 2, unique symbols | 8 / 17 | 0.013 | 0.110 | 2.87 | 102 |
| Top 2, unique base candidates | 7 / 17 | -0.060 | 0.034 | 7.18 | 113 |
| Top 1 variant | 5 / 17 | 0.103 | 0.288 | 5.75 | 50 |
| Top 2, min 20 train trades | 7 / 17 | -0.004 | 0.011 | 6.87 | 219 |

Regime gating improved some individual windows, but the result is not deployable. The positive-window rate stayed weak, top-1 was too sparse, and stricter selection constraints did not improve robustness.

## Sixth-Pass State Trend Following

A state-based trend-following family was added because the prior searches only used trend as an entry event. These strategies stay long or short while a moving-average trend state remains active and use `exit_on_flat_signal=True`, so the backtest exits when the state breaks. This produced the strongest research lead so far, but the honest train-only result remains below deployment quality.

Validation-selected state portfolio, BTCUSDT 12h + ETHUSDT 4h + SOLUSDT 4h:

- Validation PF: 3.07
- Validation expectancy: 0.358R
- Validation max drawdown: 3.38R / 3.41%
- Average win / loss: 2.04R / -0.21R
- Average RR: 9.04
- Trades/day: 0.130
- Average holding time: 46 bars
- Fixed WF: 10 / 17 positive windows, mean 0.568R, max drawdown 4.89R
- Severe 50 bps slippage stress: validation PF 2.52, expectancy 0.275R, max drawdown 3.71R, fixed WF 9 / 17

However, that portfolio is selected using holdout validation, so it is diagnostic only. The honest train-WF-selected version is materially weaker:

- Validation PF: 1.83
- Validation expectancy: 0.104R
- Validation max drawdown: 3.06R / 3.07%
- Average win / loss: 0.88R / -0.16R
- Average RR: 5.41
- Max consecutive losses: 10
- Trades/day: 0.153
- Average holding time: 38 bars
- Fixed WF: 10 / 17 positive windows, mean 0.273R, trade-weighted 0.241R, max drawdown 3.45R
- Severe 50 bps slippage stress: validation PF 1.51, expectancy 0.069R, max drawdown 3.65R, fixed WF 9 / 17

State trend following is the strongest direction found so far, but it is still not ready for live deployment. The honest edge is small after costs, the positive-window rate is only 10 / 17, and the loss streak is too long for immediate capital deployment. The next research step should focus on improving state-trend robustness with predeclared selection rules, not selecting by validation performance.
