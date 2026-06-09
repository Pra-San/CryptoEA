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

## Seventh-Pass High-Winrate Partial/Trailing Search

The original high-winrate target was tested with corrected next-bar execution, no overlapping positions, Binance-style fees, spread/slippage stress, average win/loss, payoff ratio, expectancy, holding time, and drawdown in both percent and R. Plain high-winrate and trailing-only variants did not pass stress gates under the harsh 20 bps normal / 50 bps stress slippage model.

Partial exits were then added: first partial take-profit, stop movement after partial, breakeven logic, and ATR trailing. This is the first variant class that produced a credible realistic-cost lead.

Best single-symbol candidate found so far:

| Candidate | Cost Tier | Trades | Win Rate | Sharpe | PF | Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Max Losses | Trades/Week | Avg Hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOLUSDT 1h partial/trailing trend-state | 10 bps fee + 2 bps slippage | 250 | 65.6% | 3.90 | 3.14 | 0.379 | 0.856 | -0.532 | 1.64 | 5.02 | 4.92% | 4 | 3.38 | 8 bars |
| SOLUSDT 1h partial/trailing trend-state | 10 bps fee + 10 bps stress slippage | 250 | 64.4% | 3.35 | 2.55 | 0.290 | 0.751 | -0.543 | 1.41 | 5.53 | 5.40% | 4 | 3.38 | 8 bars |
| SOLUSDT 1h partial/trailing trend-state | 10 bps fee + 50 bps shock slippage | 252 | 52.0% | 0.00 | 0.99 | 0.000 | 0.495 | -0.536 | 0.92 | 7.73 | 7.75% | 6 | 3.41 | 8 bars |

Walk-forward profile for the same fixed SOLUSDT 1h parameters:

- 17 / 17 three-month windows had positive expectancy.
- 10 / 17 passed a strict three-month gate using at least 10 trades, win rate >= 60%, PF >= 2.5, expectancy >= 0.15R, max drawdown <= 8R, and at least 0.5 trades/week.
- 2024-2025 stressed validation passed the deployment-style gate, but the 50 bps shock test failed completely.

This is a real candidate, not a finished deployment. It is much better than prior OHLCV-only variants, but it is not comparable to the originally claimed PF 9-26 / Sharpe 6-8 table after execution costs and shock testing.

## Eighth-Pass Adaptive Partial Portfolio

An adaptive selector for partial/trailing candidates was added. It ranks candidates using only the preceding training window and then evaluates the selected candidates on the next unseen three-month window. The exact evaluator now supports two risk modes:

- `equal_sleeve`: each selected strategy is scaled to an equal portfolio sleeve; R drawdown is account-level.
- `full_strategy`: each selected strategy keeps its own full 1% risk budget; this matches single-strategy R scale but allows higher portfolio risk.

Best exact top-3 unique-symbol adaptive portfolio at 10 bps fee + 10 bps stress slippage:

| Risk Mode | Positive Windows | Gate Windows | Mean Exp R | Max DD R | Trades | Notes |
|---|---:|---:|---:|---:|---:|---|
| Equal sleeve | 16 / 17 | 0 / 17 | 0.061 | 2.69 | 1,406 | Low drawdown, but account-level R expectancy is diluted by sleeve scaling. |
| Full strategy | 16 / 17 | 7 / 17 | 0.184 | 8.06 | 1,406 | Better table-style R, but drawdown is too high and PF consistency is not deployable. |

Conclusion: the best current direction is the SOLUSDT 1h partial/trailing trend-state candidate plus adaptive portfolio research. The SOL candidate can be tagged as a research candidate for paper trading, but not as a deployable production strategy until it survives forward paper trading and additional predeclared walk-forward/regime tests.

## Ninth-Pass Prop-Firm Constraint Optimization

A prop-firm evaluator was added to convert strategy output into challenge-style pass/fail checks. It tests:

- Profit target
- Daily loss limit
- Maximum total loss
- Minimum trading days
- Best-day consistency where applicable

Implemented profiles:

| Profile | Target | Daily Loss | Max Loss | Min Days | Consistency |
|---|---:|---:|---:|---:|---:|
| FTMO 2-Step Phase 1 | 10% | 5% | 10% | 4 | 50% best day |
| FTMO 2-Step Phase 2 | 5% | 5% | 10% | 4 | 50% best day |
| FTMO 1-Step | 10% | 3% | 10% | 0 | 50% best day |
| The5ers High Stakes | 8% | 5% | 10% | 3 | n/a |
| FundedNext Stellar 2-Step Phase 1 | 8% | 5% | 10% | 5 | n/a |
| FundedNext Stellar 2-Step Phase 2 | 5% | 5% | 10% | 5 | n/a |
| FundedNext Stellar 1-Step | 10% | 3% | 6% | 2 | n/a |

Best SOLUSDT 1h candidate, 2024-01-01 to 2025-06-01:

| Execution Tier | Result |
|---|---|
| 10 bps fee + 10 bps stress slippage | Passed all implemented profiles for risk-per-trade values from 0.10% to 1.00% using closed-PnL daily loss approximation. At 1.00% risk, target-pass runs stayed near -2.56% minimum closed equity and -1.27% worst closed daily loss before target. |
| 10 bps fee + 50 bps shock slippage | No prop-firm profile passed in the tested risk grid; the strategy could not reliably reach targets after execution shock. |

Important limitation: daily-loss checks are currently based on closed trade PnL because the simulator does not yet mark open positions intrabar against prop-firm equity rules. Before using this for a real challenge, add intrabar mark-to-market daily loss checks and broker/session-time calendars.
