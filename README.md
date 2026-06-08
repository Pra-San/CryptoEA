# CryptoEA — Professional-Grade Crypto Trading System

**Developer**: Sai Dheeraj Peketi  
**Author**: APEX (Quantitative Trading Engineer)  
**Version**: 0.1.0

## 📊 Project Overview

A systematic, research-driven Expert Advisor for cryptocurrency markets that:
- Discovers statistically significant trading edges using historical Binance data
- Iteratively refines strategies through rigorous walk-forward optimization
- Deploys live on Binance only when edge criteria are met

## 🏗️ Architecture

```
CryptoEA/
├── data/
│   ├── loader.py          # Binance data loader with validation
│   ├── preprocessor.py    # Feature engineering pipeline
│   └── cache/             # Parquet cache for speed
├── strategy/
│   ├── base.py            # Abstract strategy interface
│   ├── v1/
│   │   └── strategy.py    # Volatility-Adjusted Momentum
├── backtest/
│   ├── engine.py          # Vectorized backtesting engine
│   ├── metrics.py         # Comprehensive performance metrics
│   └── reports/           # Backtest result archives
├── risk/
│   ├── position_sizer.py  # Kelly, ATR, volatility sizing
│   └── drawdown_manager.py # Circuit breakers
├── scripts/
│   ├── run_backtest.py    # CLI backtest runner
│   └── run_live.py        # Live trading runner
└── utils/
    ├── logger.py          # Structured logging
    └── helpers.py         # Utility functions
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Binance API keys
```

### 3. Run Backtest
```bash
python scripts/run_backtest.py --symbol BTCUSDT --timeframe 1h --balance 10000
```

## 📈 Current Strategy: Volatility-Adjusted Momentum (v1)

**Hypothesis**: Crypto markets exhibit persistent momentum that is amplified in trending regimes and dampened in ranging regimes.

**Edge Components**:
1. **Volatility-Adjusted Momentum**: Normalizes momentum by volatility to avoid over-leveraging during high-vol periods
2. **Regime Detection**: Hurst exponent identifies trending vs ranging markets
3. **Volume Confirmation**: Requires volume spike on breakouts to filter false signals
4. **ATR-Based Risk**: Dynamic stops and targets based on current market volatility

**Signal Logic**:
- LONG: Strong positive momentum + trending regime + volume confirmation
- SHORT: Strong negative momentum + trending regime + volume confirmation
- FLAT: Ranging regime, low volatility, or no volume confirmation

## 📊 Data Available

| Symbol | Timeframe | Period | Bars | Quality |
|--------|-----------|--------|------|---------|
| BTCUSDT | 1m | 2017-08 to 2026-05 | 4.6M | Excellent |
| ETHUSDT | 1m | 2017-08 to 2026-05 | 4.6M | Excellent |
| SOLUSDT | 1m | 2020-08 to 2026-05 | 3.0M | Excellent |

Data source: Binance spot klines, merged from monthly archives + REST API for recent data.

## 🎯 Edge Qualification Criteria

A strategy is READY for live deployment when ALL minimum criteria are met:

| Metric | Minimum | Target |
|--------|---------|--------|
| OOS Sharpe Ratio | > 1.2 | > 2.0 |
| OOS Calmar Ratio | > 1.0 | > 2.5 |
| OOS Win Rate | > 40% | > 50% |
| OOS Max Drawdown | < 15% | < 10% |
| Walk-Forward Efficiency | > 70% | > 85% |
| Trade Count (OOS) | > 100 | > 300 |
| Profit Factor | > 1.5 | > 2.0 |

## ⚠️ Risk Management

- **Position Sizing**: ATR-based fixed fractional (default 1% risk per trade)
- **Circuit Breakers**: Daily/weekly/monthly loss limits with automatic halts
- **Max Drawdown**: 10% peak-to-trough triggers full trading halt
- **Consecutive Losses**: 5 losses triggers 50% position reduction

## 📝 Git Commit Protocol

```bash
[PHASE] Short description (≤ 72 chars)

- Key metrics (Sharpe, DD, Win Rate)
- What changed from previous iteration
- What was learned / hypothesis tested
- Next steps
```

## 📜 License

Proprietary — Sai Dheeraj Peketi. All rights reserved.
