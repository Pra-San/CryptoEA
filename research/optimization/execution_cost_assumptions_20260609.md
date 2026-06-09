# Execution Cost Assumptions - 2026-06-09

## Source Checks

- Binance Futures support states that maker/taker fees are charged on futures transactions and vary by VIP tier. Its USDS-M example uses 0.02% maker and 0.05% taker fees for a regular user. Source: https://www.binance.com/en/support/faq/detail/360033544231
- Binance Academy states that Binance uses a maker/taker model, taker orders execute immediately against the order book, maker orders rest and add liquidity, and fees are calculated as trade value multiplied by fee rate. Source: https://academy.binance.com/lt/articles/how-to-calculate-transaction-fees-on-binance
- Binance public futures depth snapshot on 2026-06-09 showed much tighter instantaneous book impact than the earlier 50 bps shock assumption:
  - BTCUSDT spread: 0.016 bps; estimated $1M market impact: buy 0.43 bps, sell 0.09 bps
  - ETHUSDT spread: 0.060 bps; estimated $1M market impact: buy 1.47 bps, sell 1.60 bps
  - SOLUSDT spread: 1.522 bps; estimated $1M market impact: buy 2.40 bps, sell 4.82 bps

## Backtest Cost Ladder

The new realistic-cost research tier uses:

| Tier | Fee / Side | Slippage / Side | Rationale |
|---|---:|---:|---|
| Normal taker | 5 bps | 2 bps | Regular-user taker-fee approximation plus conservative liquid-book impact. |
| Stressed liquid | 5 bps | 10 bps | Volatile but still liquid Binance futures execution. |
| Severe stress | 5 bps | 20-30 bps | Useful degradation test for fast moves, thin moments, or poor order routing. |
| Crisis shock | 5 bps | 50 bps | Rejection-only scenario; too extreme as the default deployment gate for liquid BTC/ETH/SOL futures. |

Conclusion: 50 bps adverse slippage per side should not be the default pass/fail gate. It is a crisis scenario. A 10 bps adverse-per-side stress tier is stricter than the observed depth snapshot while remaining plausible for volatile execution.
