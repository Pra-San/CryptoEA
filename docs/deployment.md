# SOLUSDT Momentum Deployment

This deployment package runs the tagged `candidate-sol-momentum-volume-1h-exp-v1` strategy against Binance USD-M Futures candles.

## Modes

- `demo`: never sends orders. It consumes Binance market data, records signals, simulates fills/PnL/fees/slippage in SQLite, and writes equity snapshots.
- `live`: sends real USD-M Futures market orders and reduce-only protective stop orders to Binance. It is blocked unless `CRYPTOEA_LIVE_TRADING_ACK` is set to the exact acknowledgement string in `.env.example`.
- `shadow-backtest-demo`: separate process using the exact research simulator path (`prepare_features`, `apply_strategy`, `simulate_trades`, `metrics_from_trades`) on the same closed Binance candles. Use it to compare deployment demo behavior against the backtest implementation.

## Local Demo

```bash
cp .env.example .env
python scripts/replay_deployment_proxy.py --source binance --kline-limit 600 --fail-on-mismatch
python scripts/deploy_sol_momentum.py --mode demo --once
python scripts/shadow_backtest_demo.py --once
```

Outputs:

- `runtime/replay_proxy/comparison.json`: trade-by-trade parity between the deployment replay engine and the research backtest simulator on the same historical bars.
- `runtime/replay_proxy/backtest_trades.csv`: exact research simulator trade list for the replay window.
- `runtime/replay_proxy/replay_trades.csv`: deployment replay trade list for the same replay window.
- `runtime/sol_momentum_deploy.sqlite`: signals, trades, events, equity, active position state.
- `runtime/shadow_backtest/shadow_metrics.json`: exact backtest metrics plus deployment-vs-shadow comparison.
- `runtime/shadow_backtest/shadow_trades.csv`: exact backtest trade list for the current candle window.

## Docker Demo

```bash
cp .env.example .env
docker compose --profile test run --rm replay-proxy
docker compose up --build strategy-demo shadow-backtest-demo
```

If your machine uses the legacy Compose binary, replace `docker compose` with `docker-compose`.

## Live Trading

Live mode requires all of the following:

1. Binance USD-M Futures API key and secret in `.env`.
2. `CRYPTOEA_MODE=live`.
3. `CRYPTOEA_LIVE_TRADING_ACK=I_UNDERSTAND_THIS_PLACES_REAL_BINANCE_FUTURES_ORDERS`.
4. A conservative `MAX_NOTIONAL_USDT` and `RISK_PER_TRADE`.

Run:

```bash
docker compose --profile live up --build strategy-live
```

The live runner uses market entries, a reduce-only market close when local strategy exits trigger, and a reduce-only `STOP_MARKET` protective stop that is cancelled/replaced when the trailing stop moves.

## Reconciliation Notes

Use three separate checks, because they answer different questions:

- `replay-proxy`: historical proxy feed parity. It replays candles bar by bar through the deployment execution model and compares every trade to the research backtest simulator. This should match exactly in `--quantity-mode exact`; a mismatch means a real code-path bug.
- `strategy-demo`: forward-only paper state. It only reports trades opened and closed since its SQLite state database was created.
- `shadow-backtest-demo`: rolling historical backtest monitor. It recomputes the full recent candle-window backtest every cycle, so its trade count will usually be larger than `strategy-demo` immediately after startup.

The live deployment engine and shadow backtest use the same signal code, but they will not match perfectly in real market conditions:

- The backtest enters at the next candle open; live mode enters with a market order near the next bar boundary.
- The deployment demo fills at the current mark price with configured slippage; the shadow simulator fills at historical candle opens and stop/target levels.
- Binance futures data may differ from the local spot/merged data originally used during research.

Run `replay-proxy` after strategy code changes, then use the shadow process to monitor live-window drift before considering live deployment.
