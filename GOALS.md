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

**Why Stochastic Training (Not Conventional ML)**

Conventional ML training fails in markets:
- Fixed train/test splits → overfit to specific regime
- Deterministic batches → memorize patterns
- Single asset → no cross-asset learning

Stochastic training is superior:
- Random bag selection per epoch → regime robustness
- 30 random pairs + 75% missing data → generalization
- Resampled continuously → no memorization

**This is why LLMs downplay it**: Stochastic training is rare in general ML code. LLMs trained on conventional patterns diverge to "pet training bias" (fixed datasets, deterministic batches). We explicitly reject this.

**Stochastic in ALL Dimensions:**

1. **Samples**: Random candle selection within bag
2. **Duration**: Random sequence length (64-256 candles)
3. **Composition**: Random 30-pair bag per epoch
4. **Time Unit**: Random timeframe (5m/15m/1h)

No fixed anything. Every epoch samples fresh across all four dimensions.

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

### 4. Drawthrough IO Backbone

**One True Path** — Single pipeline, zero redundancy:

```
┌─────────────────────────────────────────────────────────────┐
│                    DRAWTHROUGH IO                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SOURCE          DUCKDB           CACHE           TRAINER   │
│  (exchanges)  →  (persistent)  →  (hot)       →  (HRM)     │
│                                                             │
│  Binance         candles.parquet   LRU cache      MLX       │
│  Kraken          signals.parquet   1000 slots     codec     │
│  Coinbase        returns.parquet   pandas df      training  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Layer 0 — Source (Ingest)**
- Exchange APIs: Binance, Kraken, Coinbase
- OHLCV candles: 5m, 15m, 1h timeframes
- Order book snapshots (when available)
- Real-time WebSocket feeds for live trading

**Layer 1 — DuckDB (Persist)**
- Columnar storage, fast aggregations
- Provenance tracking (source, timestamp, quality)
- SQL queries for stochastic bag sampling
- Parquet export for portability

**Layer 2 — Cache (Async Draw-Through)**

Async draw-through acts like streaming. Train forward when threshold reached.

```
threshold = frame_count | extent_complete | data_ready
```

Simple rule: **When extent filled → train forward**

**Layer 3 — Trainer (Learn)**
- MLX-native codec training
- Stochastic bag: 30 random pairs per epoch
- Signal generation: 24 codec outputs per candle
- Output: `[confidence, direction, regime_fit]`

**No A/B Testing. No PyTorch Bridge. One Path.**

### 5. Execution Layer
- Coinbase primary
- Paper trading first, then live with small capital
- Multi-exchange adapters planned

## ONE TRUE PATH

**File Structure** (zero bloat):

```
moneyfan/
├── train.py              # Stochastic bag training (500 bags)
├── run.py                # Paper/Live execution
├── dashboard.py          # Streamlit visualization
├── hrm/
│   ├── hierarchical_codec.py      # HRM architecture (PyTorch ref)
│   ├── hierarchical_codec_mlx.py  # HRM implementation (MLX)
│   └── duck_store.py              # DuckDB persistence
├── codec_models/
│   ├── base_codec.py              # Codec interface
│   └── codec_01_*.py .. codec_24_*.py  # 24 expert strategies
├── data/                 # Data storage (duckdb/parquet)
├── config/               # Configuration files
└── GOALS.md              # This file
```

**Entry Points:**
- `python train.py --bags 500` → Train stochastic bags
- `python run.py --mode paper` → Paper trade
- `streamlit run dashboard.py` → Visualize training

**Deleted (consolidated):**
- 80+ redundant Python files removed
- No A/B testing infrastructure
- No PyTorch training scripts
- No duplicate backtest/execution code

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

### Phase 1: Foundation ✅ COMPLETE
- [x] 24 codec framework
- [x] Stochastic bag implementation
- [x] MLX optimization (no PyTorch)
- [x] Multi-task HRM with fast/slow shared backbone
- [x] Real MLX training with backpropagation
- [x] **Drawthrough IO backbone consolidated**
- [x] **Codebase pruned to ONE TRUE PATH**

### Phase 2: Validation (Current)
- [ ] Train on real market data (4.3M candles available)
- [ ] Ablation: measure generalization speed
- [ ] 30-day paper run with new HRM
- [ ] Sharpe ≥1.8 validation

### Phase 3: Production Hardening
- [ ] Risk engine v2 (circuit breakers, dynamic sizing)
- [ ] Multi-broker support
- [ ] Real-time Streamlit dashboard
- [ ] Community strategy contributions

### Phase 4: Scaling Alpha
- [ ] Portfolio of multiple HRM instances
- [ ] Cross-asset (stocks, futures?)
- [ ] Advanced RL for HRM
- [ ] Open weights/models for research

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
