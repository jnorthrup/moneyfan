# moneyfan — Grok-Issued Project Goals

**Version:** 2.4 (Grok Edition — Fast/Slow Shared World Model) | **Date:** February 2026  
**Mission:** Build a production-grade hierarchical AI trading system that consistently generates alpha in cryptocurrency markets.

## Vision

moneyfan is an open-source **Hierarchical Reasoning Model (HRM)** for trading. It combines:
- 24 diverse State-Of-The-Art (SOTA) "codec" strategies as specialized experts
- A meta-allocator (HRM) that learns dynamic trust and allocation across experts based on regime
- Stochastic training for robustness against regime shifts and new assets

Target: Sharpe ≥1.8, Max DD ≤15%, net annualized >20% with strict risk controls.  
Democratizing prop-shop level alpha generation.

## Core Principles

1. Hierarchy Over Flat Ensembles  
2. Stochastic Robustness  
3. Diversity is Alpha  
4. Measurable Excellence  
5. Efficiency First (MLX-native)  
6. Truth-Seeking Iteration

## Key Components & Architecture

### 1. Stochastic Bag
- 30 randomly selected pairs + USD base  
- $100 notional per agent  
- Up to 75% missing candles  
- Resampled per epoch

### 2. Codecs (The 24 Experts)

Each codec is an independent model/strategy trained on instrument-metrics and frozen for HRM input.  
**Full explicit list (canonical order):**

1. **momentum_breakout** – detects acceleration + volume confirmation  
2. **mean_reversion_rsi** – classic overbought/oversold with dynamic thresholds  
3. **volatility_regime_garch** – regime switching via GARCH volatility forecast  
4. **trend_following_ema** – multi-timeframe EMA stack with slope filter  
5. **macd_crossover** – signal-line + histogram divergence  
6. **bollinger_bands_squeeze** – volatility contraction → expansion plays  
7. **stochastic_kd** – %K/%D crossover with overbought filter  
8. **ichimoku_cloud** – full cloud, conversion, base, lagging span signals  
9. **adx_trend_strength** – directional movement + ADX power filter  
10. **cci_commodity_channel** – cyclical deviation from statistical mean  
11. **parabolic_sar** – trailing stop + reversal detection  
12. **vwap_mean_reversion** – volume-weighted anchor reversion  
13. **order_book_imbalance** – live depth delta (when exchange feed available)  
14. **kalman_filter_trend** – adaptive smoothing of price + velocity  
15. **arima_predictor** – statistical time-series forecast (p,d,q tuned per asset)  
16. **hurst_regime** – long-memory detection for trend vs mean-reversion  
17. **fractal_dimension** – chaos vs structure classification  
18. **random_forest_classifier** – ensemble of tree-based feature signals  
19. **xgboost_signal** – gradient boosting on engineered metrics  
20. **lstm_sequence_predictor** – recurrent net on normalized candle sequences  
21. **transformer_attention** – self-attention on multi-timeframe patches  
22. **rl_dqn_policy** – deep Q-network tactical execution agent  
23. **pair_correlation_arb** – statistical arbitrage on coin pairs  
24. **zscore_stat_arb** – multi-asset z-score mean-reversion baskets

Output of every codec: `[confidence 0-1, direction -1/0/1, regime_fit_score]`

### 3. HRM (Hierarchical Reasoning Meta-Allocator)

**Fast & Slow layers share EVERYTHING for maximum benefit:**

**Shared Backbone Encoder** (the world-model heart):
- Predicts next OHLCV + all codec kernel metrics (multi-task heads)  
- Both fast and slow read from this exact same encoder  
- Gradients flow through shared weights → mutual generalization boost

**Slow Layer (Strategic — high level)**:
- Regime detection, risk budgeting, codec trust matrix  
- Outputs conditioning context to Fast layer

**Fast Layer (Tactical — low level)**:
- Real-time signal execution, position sizing, veto enforcement  
- Feeds immediate performance feedback back to shared encoder + Slow layer

**Bidirectional Benefit Loop**:
- Slow gives Fast strategic guardrails
- Fast gives Slow granular market truth
- Shared world-model loss makes both converge faster and generalize to new coins instantly

Training recipe:
```
# shared_encoder + slow_head + fast_head
loss = world_model_loss + trust_loss
loss.backward()  # updates EVERYTHING together
```

This is the prop-shop secret: hierarchy + shared world model = exponential performance lift.

### 4. Data & Training Pipeline
- Binance/Kraken/Coinbase historical + live feeds
- Instrument-Metric feature engineering
- A/B testing: PyTorch vs native MLX implementations
- Sparkline memory, hierarchical cycles

### 5. Execution Layer
- Coinbase primary
- Paper trading first, then live with small capital
- Multi-exchange adapters planned

## Performance Targets (Non-Negotiable)

| Metric              | Target Value     | Validation Method      |
|---------------------|------------------|------------------------|
| Sharpe Ratio        | ≥ 1.8            | Walk-forward 3yr       |
| Max Drawdown        | ≤ 15%            | Full period + stress   |
| Annualized Return   | >20% net         | After fees/slippage    |
| Calmar Ratio        | ≥ 1.5            | Risk-adjusted          |
| Win Rate (overall)  | 55%+             | Live paper min 30 days |
| Turnover            | <100% annualized | Efficiency metric      |

**Ablation Requirement**: Removing hierarchy must degrade Sharpe by at least 0.5-0.7 points.

## Roadmap & Milestones

### Phase 1: Foundation (Current)
- [x] 24 codec framework
- [x] Stochastic bag implementation
- [x] MLX optimization
- [x] Multi-task HRM with fast/slow shared backbone

### Phase 2: Validation (Next 2-4 weeks)
- [x] Add multi-task heads to HRM (hrm_meta.py)
- [ ] Update training loop with shared encoder loss
- [ ] Ablation: measure generalization speed
- [ ] 30-day paper run with new HRM

### Phase 3: Production Hardening
- Risk engine v2 (circuit breakers, dynamic sizing)
- Multi-broker support
- Real-time monitoring dashboard
- Community strategy contributions

### Phase 4: Scaling Alpha
- Portfolio of multiple HRM instances
- Cross-asset (stocks, futures?)
- Advanced RL for HRM
- Open weights/models for research

## Success Definition

The project succeeds when:
1. We have reproducible >1.8 Sharpe in out-of-sample paper/live trading.
2. Code is clean, tested, documented, and extensible.
3. Community can fork, improve codecs, and contribute verified alpha.
4. It serves as a reference implementation of hierarchical trading AI.

## Contribution Guidelines (Grok Style)

- Every PR must include tests and performance impact analysis.
- Focus on robustness > fancy features.
- Measure everything.
- Prioritize making money with controlled risk over theoretical perfection.

**"Understand the markets. Then trade them better."**

— Grok, brains of the moneyfan operation
