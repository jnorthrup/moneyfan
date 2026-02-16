# Product Guide: moneyfan-trader

## Vision
A polyglot crypto trading platform that empowers developers and retail traders to build, backtest, simulate, and deploy automated trading strategies against live exchanges (Coinbase, Kraken) — with a focus on transparency, extensibility, and fee-aware profitability.

## Target Users
- **Retail traders & hobbyists** who want automated strategies without manual execution
- **Developers & quants** who build and backtest custom algorithms with real market data
- **Researchers** experimenting with ML-driven market signals (HRM/PyTorch models)

## Core Goals
1. **Simulate before you risk** — paper-trade with real price feeds before going live
2. **Backtest rigorously** — replay historical candle data (699K+ hourly candles) to validate strategies
3. **Go live safely** — deploy authenticated bots to Coinbase Advanced Trade or Kraken with risk controls
4. **ML-enhanced signals** — integrate heart-rate-model (HRM) PyTorch signals into trading decisions
5. **Fee-aware** — all P&L calculations account for maker/taker fees, spread, and network costs

## Key Features
- Multi-strategy engine: mean reversion, momentum, grid trading, DCA
- Real-time price feeds via Coinbase public WebSocket API
- SQLite & DuckDB backtesting with historical OHLCV data
- Playwright E2E + unit test suite
- Web dashboard (Vite + Express) for monitoring bots and trade history
- Kotlin/XChange JVM bot as an alternative execution layer
- Python ML pipeline (`hrm/`) for continuous model training and A/B testing of signals

## Success Metrics
- Backtest Sharpe ratio > 1.0 on 2-year BTC/ETH hourly data
- Live bot P&L positive net of all fees over 30-day windows
- Simulation-to-live parity: <5% strategy drift between sim and live modes
- Test coverage: >80% on core trading logic
