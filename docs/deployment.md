# SOLUSDT Momentum Deployment

This deployment package runs the tagged `candidate-sol-momentum-volume-1h-exp-v1` strategy against Binance USD-M Futures candles.

## Modes

- `demo`: never sends orders. It consumes Binance market data, records signals, simulates fills/PnL/fees/slippage in SQLite, and writes equity snapshots.
- `live`: sends real USD-M Futures market orders and reduce-only protective stop orders to Binance. It is blocked unless `CRYPTOEA_LIVE_TRADING_ACK` is set to the exact acknowledgement string in `.env.example`.
- `shadow-backtest-demo`: separate process using the exact research simulator path (`prepare_features`, `apply_strategy`, `simulate_trades`, `metrics_from_trades`) on the same closed Binance candles. Use it to compare deployment demo behavior against the backtest implementation.

## Local Demo

```bash
cp .env.example .env
python scripts/deploy_sol_momentum.py --mode demo --once
python scripts/shadow_backtest_demo.py --once
```

Outputs:

- `runtime/sol_momentum_deploy.sqlite`: signals, trades, events, equity, active position state.
- `runtime/shadow_backtest/shadow_metrics.json`: exact backtest metrics plus deployment-vs-shadow comparison.
- `runtime/shadow_backtest/shadow_trades.csv`: exact backtest trade list for the current candle window.

## Docker Demo

```bash
cp .env.example .env
docker compose up --build strategy-demo shadow-backtest-demo
```

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

The deployment engine and shadow backtest use the same signal code, but they will not match perfectly in live conditions:

- The backtest enters at the next candle open; live mode enters with a market order near the next bar boundary.
- The deployment demo fills at the current mark price with configured slippage; the shadow simulator fills at historical candle opens and stop/target levels.
- Binance futures data may differ from the local spot/merged data originally used during research.

Use the shadow process to identify these differences before considering live deployment.
