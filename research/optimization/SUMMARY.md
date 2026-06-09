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

## Tenth-Pass VWAP, Volume Profile, And Momentum Search

VWAP, rolling volume-profile, value-area, and momentum variants were added after the high-winrate partial/trailing search. The new explorer tests:

- Rolling VWAP reclaim and pullback variants
- UTC daily/session VWAP pullbacks
- VWAP band reversion
- Rolling volume-profile POC/VAL/VAH rejection, breakout, and rotation
- VWAP + profile confluence
- Momentum + volume breakout

The explorer now records average win, average loss, payoff ratio, expectancy, average holding time, trades/day, trades/week, drawdown in R, and drawdown in percent for both validation and stress cost tiers.

Best validation candidates, 2024-01-01 to 2025-06-01:

| Candidate | Cost Tier | Trades | Win Rate | Sharpe | PF | Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Max Losses | Trades/Week | Avg Hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT 1h VWAP/profile confluence | 10 bps fee + 10 bps slippage | 258 | 75.2% | 3.73 | 2.98 | 0.297 | 0.587 | -0.582 | 0.98 | 4.54 | 4.03% | 4 | 3.49 | 8 bars |
| ETHUSDT 1h VWAP/profile confluence | 10 bps fee + 10 bps slippage | 204 | 63.2% | 2.29 | 2.71 | 0.365 | 0.912 | -0.577 | 1.58 | 5.55 | 5.32% | 4 | 2.76 | 8 bars |
| SOLUSDT 1h daily VWAP pullback | 10 bps fee + 10 bps slippage | 578 | 77.5% | 5.45 | 2.99 | 0.189 | 0.358 | -0.390 | 0.87 | 1.55 | 1.54% | 3 | 7.83 | 8 bars |

Stress validation:

| Candidate | Stress Tier | Trades | Win Rate | Sharpe | PF | Exp R | Max DD R | Max DD % | Trades/Week | Fixed WF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT 1h VWAP/profile confluence | 10 bps fee + 10 bps stress slippage | 258 | 73.3% | 2.82 | 2.15 | 0.188 | 5.67 | 5.25% | 3.49 | 11 / 11 positive |
| ETHUSDT 1h VWAP/profile confluence | 10 bps fee + 10 bps stress slippage | 204 | 61.8% | 1.85 | 2.17 | 0.262 | 6.23 | 6.03% | 2.76 | 11 / 11 positive |
| SOLUSDT 1h daily VWAP pullback | 10 bps fee + 50 bps shock slippage | 577 | 64.6% | 1.60 | 1.28 | 0.043 | 5.34 | 5.23% | 7.81 | 11 / 11 positive, 8 / 11 strict gates |

The SOLUSDT daily VWAP pullback is the closest result to the originally requested high-winrate profile: high trade count, high win rate, high Sharpe, low R drawdown, low percent drawdown, and stable walk-forward windows. It is still not a production-live strategy because full-period 50 bps shock slippage reduces PF to 1.28 and expectancy to 0.043R.

Prop-firm constraint evaluation for the SOLUSDT daily VWAP pullback:

| Execution Tier | Result |
|---|---|
| 10 bps fee + 10 bps slippage | Passed 63 / 63 profile-risk combinations tested. It passed all implemented FTMO, The5ers, and FundedNext profiles from 0.10% to 1.00% risk per trade. |
| 10 bps fee + 50 bps shock slippage | Passed 36 / 63 profile-risk combinations tested. It still hit several prop-style targets before stopping, but the full-period strategy statistics were weak under this shock tier. |

Current status: tag the SOLUSDT 1h daily VWAP pullback as the best paper-trade / prop-challenge research candidate so far. Do not mark it live-deployable until the evaluator includes intrabar mark-to-market prop daily-loss checks, exchange-specific spread models, and at least two weeks of forward paper trading.

## Eleventh-Pass Shock-Trained And Filtered Search

After the daily VWAP candidate failed full-period 50 bps shock quality, the search was rerun with the shock model used during training, not only during validation. A separate causal trade-filter optimizer was also added. It filters saved candidate signals by side, UTC hour, day-of-week group, volume ratio, daily volatility, RSI, distance from session VWAP, distance from rolling VWAP, distance from profile POC, and price side of VWAP/POC.

Best new results:

| Branch | Shock Trades | Shock Win Rate | Shock Sharpe | Shock PF | Shock Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Trades/Week | Fixed WF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOLUSDT 1h shock-trained daily VWAP pullback | 153 | 69.9% | 0.91 | 1.32 | 0.061 | 0.350 | -0.610 | 0.57 | 3.75 | 3.69% | 2.07 | 14 / 17 positive |
| SOLUSDT 1h filtered daily VWAP pullback | 60 | 75.0% | 1.39 | 1.95 | 0.105 | 0.287 | -0.441 | 0.65 | 1.39 | 1.39% | 0.81 | 15 / 17 positive |
| SOLUSDT 1h filtered shock-trained VWAP | 46 | 69.6% | 0.62 | 1.38 | 0.073 | 0.401 | -0.702 | 0.57 | 3.05 | 3.01% | 0.93 | 14 / 17 positive |
| SOLUSDT 1h shock-trained state trend | 44 | 52.3% | 0.97 | 1.67 | 0.244 | 1.159 | -0.759 | 1.52 | 2.49 | 2.47% | 0.60 | 7 / 17 positive |
| ETHUSDT 1h shock-trained state trend | 43 | 55.8% | 0.63 | 1.41 | 0.136 | 0.834 | -0.744 | 1.11 | 3.60 | 3.56% | 0.58 | 8 / 17 positive |

Result: the filter optimizer improved the SOL daily VWAP candidate materially under 50 bps shock, but it still does not meet the requested deployment profile. The best filtered version has good win rate and drawdown, but PF is below 2.5 and trade frequency is only 0.81 trades/week. The stress-trained state-trend branch has larger average wins but does not have stable walk-forward coverage.

Current best research candidate remains the SOLUSDT 1h daily VWAP pullback for paper/prop simulation under normal execution. Current best 50 bps shock candidate is the filtered SOLUSDT daily VWAP pullback, but it is too sparse and not live-deployable.

## Twelfth-Pass Multi-Timeframe Confluence Search

The next branch tested shifted 4h and daily context on 1h entries. The explorer uses only completed higher-timeframe candles by shifting 4h/daily features one bar before merging them into the 1h frame. Families tested:

- Multi-timeframe momentum breakout
- Multi-timeframe trend pullback
- Multi-timeframe Bollinger expansion
- Multi-timeframe RSI reclaim
- Multi-timeframe state-follow trend

This pass was guided by multi-timeframe trend/momentum research and practical crypto execution work: higher-timeframe trend confirmation, shifted feature alignment to avoid lookahead, and more patient trade management. It was trained directly under 10 bps Binance-style fee plus 50 bps shock slippage.

Best holdout candidates, 2024-01-01 to 2025-06-01:

| Symbol | Family | Val Trades | Val Win | Val Sharpe | Val PF | Val Exp R | Val DD R | Val TPW | Stress Trades | Stress Win | Stress Sharpe | Stress PF | Stress Exp R | Avg Win R | Avg Loss R | RR | Stress DD R | Stress DD % | Stress TPW | Avg Hold | WF Positive | WF Gates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | mtf_state_follow | 118 | 72.0% | 2.37 | 2.99 | 0.160 | 0.95 | 1.60 | 119 | 57.1% | 0.04 | 1.01 | 0.002 | 0.261 | -0.343 | 0.76 | 6.73 | 6.52% | 1.61 | 12 bars | 5 / 17 | 0 / 17 |
| ETHUSDT | mtf_momentum_breakout | 94 | 75.5% | 1.88 | 4.21 | 0.280 | 1.87 | 1.27 | 94 | 63.8% | 1.06 | 1.79 | 0.112 | 0.396 | -0.389 | 1.01 | 2.94 | 2.91% | 1.27 | 12 bars | 10 / 17 | 0 / 17 |
| SOLUSDT | mtf_bb_expansion | 34 | 79.4% | 1.66 | 3.74 | 0.393 | 1.60 | 0.46 | 34 | 70.6% | 0.45 | 1.30 | 0.065 | 0.391 | -0.718 | 0.54 | 2.70 | 2.67% | 0.46 | 12 bars | 12 / 17 | 0 / 17 |

Result: rejected. The SOLUSDT setup had an attractive validation win rate but only 34 holdout trades, 0.46 trades/week, and stress PF collapsed to 1.30 after the 50 bps shock model. ETH was more balanced but still failed all walk-forward deployment gates. BTC effectively lost its edge after shock costs.

Current conclusion: pure OHLCV strategy mining is flattening out. The best normal-cost candidate is still the SOLUSDT 1h daily VWAP pullback. The best shock-cost candidate is still the filtered SOLUSDT daily VWAP pullback. To look for a material improvement rather than overfitting the same bars harder, the next research branch should ingest non-OHLCV edge data: funding, open interest, liquidation clusters, order book imbalance, realized spread, and exchange-specific maker/taker execution assumptions.

## Thirteenth-Pass Binance Futures-Flow Search

A futures-flow data branch was added to avoid mining the same OHLCV features harder. The downloader builds local parquet datasets from Binance public USD-M archives:

- 5-minute open interest
- Open-interest value
- Top-trader long/short ratios
- Global long/short ratio
- Taker long/short volume ratio
- 8-hour funding rates

The futures-flow explorer shifts hourly flow features before merging them into 1h price bars. It tests flow breakout, flow pullback, flow squeeze, funding contrarian reversion, crowding breakout, and taker absorption variants with the same Binance-style fee and shock-slippage assumptions.

Best futures-flow holdouts:

| Run | Family | Val Trades | Val Win | Val Sharpe | Val PF | Val Exp R | Val DD R | Val TPW | Stress Trades | Stress Win | Stress Sharpe | Stress PF | Stress Exp R | Avg Win R | Avg Loss R | RR | Stress DD R | Stress DD % | Stress TPW | Avg Hold | WF Positive | WF Gates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOLUSDT 1h broad flow | flow_breakout | 114 | 73.7% | 2.72 | 3.00 | 0.216 | 1.47 | 1.54 | 114 | 66.7% | 1.25 | 1.53 | 0.075 | 0.322 | -0.420 | 0.76 | 1.70 | 1.69% | 1.54 | 12 bars | 8 / 17 | 0 / 17 |
| SOLUSDT 1h focused flow | flow_pullback | 99 | 80.8% | 2.52 | 3.42 | 0.217 | 1.12 | 1.34 | 100 | 70.0% | 1.03 | 1.48 | 0.068 | 0.298 | -0.470 | 0.63 | 2.35 | 2.33% | 1.35 | 12 bars | 11 / 12 | 0 / 12 |
| BTCUSDT 1h focused flow | flow_breakout | 80 | 77.5% | 2.40 | 3.35 | 0.312 | 2.17 | 1.08 | 81 | 65.4% | 0.31 | 1.12 | 0.025 | 0.348 | -0.585 | 0.59 | 3.00 | 2.97% | 1.10 | 12 bars | 13 / 17 | 0 / 17 |
| ETHUSDT 1h focused flow | flow_breakout | 64 | 79.7% | 1.40 | 2.48 | 0.262 | 3.26 | 0.87 | 64 | 68.8% | 0.02 | 1.00 | 0.002 | 0.393 | -0.858 | 0.45 | 5.98 | 5.87% | 0.87 | 12 bars | 6 / 12 | 0 / 12 |

Result: rejected for live deployment. This is the first branch that materially improves normal-cost holdout quality with non-OHLCV data, especially SOLUSDT focused flow at 80.8% win rate, PF 3.42, and 1.12R drawdown under normal costs. Under 50 bps shock, all branches lose too much expectancy and fail every strict deployment gate. The winners are still too small relative to stressed execution cost.

Current best research lead: SOLUSDT 1h flow pullback is worth further research only if the execution model can be made more precise. It needs maker/taker split, actual spread sampling, and limit-order fill modeling before deciding whether the 50 bps shock model is too conservative or whether the edge is simply not wide enough.

## Fourteenth-Pass Realistic Binance Futures Cost Model

The previous 50 bps shock model was too punitive as a default deployment gate. Binance's own Futures support page states that futures fees vary by VIP tier and gives regular-user examples of 0.02% maker and 0.05% taker fees for USDS-M contracts. Binance Academy also confirms the maker/taker fee model and the notional-value fee formula. A live Binance futures depth snapshot on 2026-06-09 showed substantially tighter book impact than the old 50 bps-per-side assumption:

| Symbol | Quoted Spread | Est. $1M Buy Impact | Est. $1M Sell Impact |
|---|---:|---:|---:|
| BTCUSDT | 0.016 bps | 0.43 bps | 0.09 bps |
| ETHUSDT | 0.060 bps | 1.47 bps | 1.60 bps |
| SOLUSDT | 1.522 bps | 2.40 bps | 4.82 bps |

Updated cost ladder:

| Tier | Fee / Side | Slippage / Side | Use |
|---|---:|---:|---|
| Normal taker | 5 bps | 2 bps | Baseline liquid Binance futures taker execution. |
| Stressed liquid | 5 bps | 10 bps | Default deployment stress gate. |
| Severe stress | 5 bps | 20-30 bps | Degradation check. |
| Crisis shock | 5 bps | 50 bps | Rejection-only scenario, not default. |

Best realistic-cost search result:

| Candidate | Tier | Trades | Win Rate | Sharpe | PF | Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Max Losses | Trades/Week |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOLUSDT 1h profile breakout | 5 bps fee + 2 bps slippage | 184 | 81.5% | 4.04 | 4.74 | 0.227 | 0.353 | -0.330 | 1.07 | 1.35 | 1.34% | 3 | 2.49 |
| SOLUSDT 1h profile breakout | 5 bps fee + 10 bps stress slippage | 184 | 78.8% | 3.72 | 3.88 | 0.193 | 0.331 | -0.320 | 1.04 | 1.39 | 1.38% | 3 | 2.49 |
| SOLUSDT 1h futures-flow pullback | 5 bps fee + 10 bps stress slippage | 154 | 74.7% | 2.55 | 2.58 | 0.183 | 0.396 | -0.445 | 0.88 | 1.53 | 1.52% | 3 | 2.09 |

Cost degradation for SOLUSDT 1h profile breakout:

| Slippage / Side | Fee / Side | Trades | Win Rate | Sharpe | PF | Exp R | Max DD R | Trades/Week |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 bps | 5 bps | 184 | 81.5% | 4.04 | 4.74 | 0.227 | 1.35 | 2.49 |
| 5 bps | 5 bps | 184 | 81.0% | 3.96 | 4.51 | 0.216 | 1.37 | 2.49 |
| 10 bps | 5 bps | 184 | 78.8% | 3.72 | 3.88 | 0.193 | 1.39 | 2.49 |
| 20 bps | 5 bps | 184 | 76.6% | 3.31 | 3.10 | 0.157 | 1.39 | 2.49 |
| 30 bps | 5 bps | 184 | 73.9% | 2.79 | 2.39 | 0.123 | 1.38 | 2.49 |
| 50 bps | 5 bps | 184 | 67.9% | 1.59 | 1.54 | 0.062 | 1.96 | 2.49 |

Walk-forward result at 5 bps fee + 10 bps slippage: 17 / 17 positive windows, 522 total trades, 79.3% mean win rate, 0.303R trade-weighted expectancy, 1.72R max drawdown, and 2.57 trades/week. The strict per-window deployment gate still reports 0 windows because it requires 120 trades inside every 3-month slice, which is too high for a 2.5 trades/week strategy. Use the full holdout gate plus positive-window consistency here.

Prop-firm evaluation: the profile breakout passed the implemented FTMO, The5ers, and FundedNext profile-risk combinations under both 2 bps and 10 bps slippage tiers, using the closed-PnL daily-loss approximation. This still needs intrabar mark-to-market daily-loss checks before challenge deployment.

Current status: SOLUSDT 1h profile breakout is now the best candidate. It is not guaranteed live-deployable, but it has the first genuinely strong combination of win rate, PF, expectancy, low R drawdown, positive walk-forward coverage, and acceptable weekly trade frequency under a defensible Binance futures cost model.

## Fifteenth-Pass Frequency And Expectancy Focus

The realistic-cost VWAP/profile explorer was rerun with a focused family selector so trials could concentrate on the families that had already survived the corrected execution model: profile breakout, VWAP/profile confluence, momentum/volume breakout, daily VWAP pullback, and VWAP band reversion.

Best new candidate:

- Symbol/timeframe: SOLUSDT 1h
- Family: momentum volume breakout
- Core shape: Donchian 16 breakout, volume lookback 50, momentum lookback 18, EMA 20/200 trend state, 2.5 ATR stop, 12 ATR target, 8-36 bar holding window, trailing stop enabled, both long and short allowed
- Validation: 2024-01-01 to 2025-06-01
- Cost model: 5 bps taker fee per side, with 2 bps normal slippage and 10 bps deployment-stress slippage

Comparison against the previous best realistic-cost SOL profile breakout:

| Candidate | Tier | Trades | Win Rate | Sharpe | PF | Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Max Losses | Trades/Week |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous SOLUSDT 1h profile breakout | 5 bps fee + 10 bps slippage | 184 | 78.8% | 3.72 | 3.88 | 0.193 | 0.331 | -0.320 | 1.04 | 1.39 | 1.38% | 3 | 2.49 |
| New SOLUSDT 1h momentum volume breakout | 5 bps fee + 2 bps slippage | 284 | 82.7% | 4.79 | 5.59 | 0.307 | 0.451 | -0.384 | 1.17 | 1.04 | 1.04% | 2 | 3.85 |
| New SOLUSDT 1h momentum volume breakout | 5 bps fee + 10 bps slippage | 284 | 80.3% | 4.40 | 4.45 | 0.258 | 0.414 | -0.376 | 1.09 | 1.22 | 1.22% | 2 | 3.85 |

Execution-cost degradation for the new candidate:

| Slippage / Side | Fee / Side | Trades | Win Rate | Sharpe | PF | Exp R | Avg Win R | Avg Loss R | RR | Max DD R | Max DD % | Max Losses | Trades/Week |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bps | 5 bps | 283 | 70.3% | 4.88 | 5.85 | 0.321 | 0.550 | -0.220 | 2.47 | 1.04 | 1.04% | 4 | 3.83 |
| 2 bps | 5 bps | 284 | 82.7% | 4.79 | 5.59 | 0.307 | 0.451 | -0.384 | 1.17 | 1.04 | 1.04% | 2 | 3.85 |
| 5 bps | 5 bps | 284 | 82.7% | 4.68 | 5.28 | 0.290 | 0.431 | -0.388 | 1.10 | 1.05 | 1.05% | 2 | 3.85 |
| 10 bps | 5 bps | 284 | 80.3% | 4.40 | 4.45 | 0.258 | 0.414 | -0.376 | 1.09 | 1.22 | 1.22% | 2 | 3.85 |
| 20 bps | 5 bps | 284 | 77.5% | 3.89 | 3.42 | 0.208 | 0.378 | -0.380 | 0.99 | 1.87 | 1.86% | 3 | 3.85 |
| 30 bps | 5 bps | 284 | 74.6% | 3.34 | 2.65 | 0.163 | 0.351 | -0.390 | 0.90 | 3.15 | 3.12% | 4 | 3.85 |
| 50 bps | 5 bps | 284 | 71.8% | 2.52 | 1.93 | 0.107 | 0.310 | -0.411 | 0.76 | 3.83 | 3.80% | 4 | 3.85 |

Walk-forward aggregate from the fixed candidate:

- 17 / 17 positive-expectancy three-month windows.
- 767 total walk-forward trades.
- Trade-weighted expectancy: 0.368R.
- Mean win rate: 80.8%.
- Weighted average win / loss: 0.549R / -0.388R.
- Weighted payoff ratio: 1.39.
- Max walk-forward drawdown: 2.02R / 2.01%.
- Weighted frequency: 3.58 trades/week.

Prop-firm evaluation under 5 bps fee + 10 bps slippage passed all implemented FTMO, The5ers, and FundedNext profiles in the tested risk grid using closed-PnL daily-loss approximation. The lowest passing risks were 0.10% for 5% targets and 0.15% for 8-10% targets; larger risk settings also passed in the current closed-PnL model.

Current status: this supersedes the SOLUSDT 1h profile breakout as the best candidate so far. It improves validation trade count by 54%, deployment-stress expectancy by 34%, trades/week by 54%, win rate, Sharpe, PF, max R drawdown, percent drawdown, and max loss streak. It is the first candidate that looks close to the originally requested table under a defensible Binance futures cost model. It still needs forward paper trading and intrabar mark-to-market daily-loss simulation before real prop-firm or live deployment.
