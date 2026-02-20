# moneyfan — High-Entropy Goals

## High-Entropy Performance Targets
**20%+ net annualized return, Sharpe ≥ 1.8, MaxDD ≤ 15%**  
**100% turnover cap, 2023-2026 out-of-sample testing**  
**30-day live paper trading + 10K Monte-Carlo slippage simulations**

## High-Entropy Architecture Choices
**Data Store**: DuckDB (hrm/data/market.duckdb, schema ready, awaiting data)  
**Vector Store Replacement**: Replace hyperbolic memory with 64-dim numpy vectors  
**3-Predictor MVP**: 5m Transformer + 15m XGBoost + 1h LightGBM  
**4-Stage HRM Rollout**: EarnHFT-inspired, 80% benefit, debuggable in weekend  
**Predictor/Live Split**: Pure numpy+MLX inference vs pandas training (50ms latency)

## High-Entropy Risk Controls
**Veto Layer**: HRM high-level rejects trades when regime_confidence < 0.75  
**Portfolio Limits**: 20% max per symbol, 3-5 uncorrelated positions  
**Hard Stops**: 2% loss max, 5% daily drawdown freeze, 3 losing trades → 1hr pause  
**Kelly/Fixed-Fraction**: 1-2% risk per trade, ATR-based stops/targets

## High-Entropy Validation Protocol
**Walk-Forward**: 12-month train, 3-month test, 4 cycles (3 years total)  
**Ablation Test**: Merge vs hierarchy must show ≥0.5 Sharpe improvement  
**Statistical Significance**: 10K random shuffles, p < 0.05 alpha validation  
**Live Paper**: 30 days minimum, real market data with 0.1-0.3% slippage

## High-Entropy Trading Recipe
**Regime Detection**: 15-60min updates, veto threshold 0.75 confidence  
**Entry Threshold**: |aggregated_signal| > 0.3, 1-2% risk per trade  
**Take Profit**: 2-3× ATR or trailing stop, stop loss = 1× ATR or 2% hard stop

---

## Cross-Exchange Training Strategy (Locked)
**Binance = Training Fuel | Coinbase = Execution Venue**

### Data Flow
```
Binance public REST (historical klines + depth, no auth, unlimited)
     ↓
BinanceDataLoader + StochasticBagGenerator (GBM/OU bagging)
     ↓
Train 3 predictors (5m Transformer / 15m XGBoost / 1h LightGBM) + HRM workers
     ↓ (frozen models)
Live Coinbase WS candles (real-time, harmonized features)
     ↓
Test-time predictor (MLX, pure numpy buffer) + vector store skew
     ↓
HRM router (4-stage) → live_executor.py → Coinbase Advanced Trade paper orders
```

### Feature Harmonization
Same TA-lib indicators, same imbalance calc → models transfer cleanly across exchanges.

### Implementation Files
- `data/binance_data_loader.py` — public REST klines, orderbook, no auth
- `train/binance_stochastic_bag_trainer.py` — 10k+ synthetic paths, retrains all predictors
- `live_executor.py` + `coinbase_websocket.py` — unchanged, Coinbase execution

### Validation Pipeline
[✓] Cross-exchange strategy locked (Binance train → Coinbase execute)
[✓] Push binance_data_loader + binance_stochastic_bag_trainer
[ ] Retrain 3 predictors + HRM on Binance bags
[ ] Run 4h Coinbase paper validation
[ ] Scale 3 → 8 short-horizon (once PF > 1.5 on paper)
[ ] Vector cache ablation (4h each)
[ ] Full 24h paper run after retraining

### Pandas Policy (Locked)
**ALL live agents including predictors = pandas enabled**
- Predictors, HRM, training path all use pandas DataFrames
- 48-column schema enforced for all live agents
- No numpy-only constraint anywhere in live path

---

## Instrument Kernels (Locked Names)

### 24 Agents/Codecs (hrm/codecs.py)
Each outputs `[confidence, direction, regime_fit]`:
```
volatility_breakout    momentum_trend         mean_reversion
trend_following        pairs_trading          grid_trading
volume_profile         order_flow             correlation_trading
liquidity_making       sector_rotation        composite_alpha
rsi_reversal           bollinger_bands        macd_cross
atr_breakout           tick_momentum          dca_baseline
technical_ml           hrm_mean_reversion     volatility_x_momentum
mean_reversion_v2      sector_rotation_v2     composite_trend
```

### Computational Kernels (hrm/kernels.py)
```
rolling_mean      rolling_std       rolling_max       rolling_min
rolling_zscore    rolling_quantile  volatility_breakout
momentum_trend    mean_reversion    cross_sectional_rank
cross_sectional_zscore  sector_neutralize
```

### Lazy Instruments (hrm/instruments.py)
```
market_data       features          sectors            signals
```

### Feature Columns (hrm/features.py)
```
price_cols:     open, high, low, close, volume
derived_cols:   returns, volatility, momentum, rsi, ma_ratio
time_cols:      hour, day_of_week, session, month, month_end
```

### DuckDB Schema (hrm/data/market.duckdb)
```
candles(symbol, time, timestamp, open, high, low, close, volume)
```
