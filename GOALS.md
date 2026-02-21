# moneyfan — Project Goals

**Version:** 2.5 (Crypto-Technical Edition) | **Date:** February 2026
**Mission:** Build a production-grade Hierarchical Reasoning Model (HRM) trading system that consistently generates alpha in cryptocurrency markets.

## Vision

moneyfan is an open-source **HRM** (Hierarchical Reasoning Model — sapientinc/HRM) for trading. It combines:
- 24 diverse SOTA codec experts as specialised signal generators
- A meta-allocator (HRM) that learns dynamic `allocation_confidence` and notional allocation across experts based on regime
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

### The Draw-Thru Architecture
It is important to emphasize that this system operates on a direct "draw-thru" architecture. Data flows linearly and efficiently:
**`data -> {duck->cache,API} -> pandas`**
This eliminates unnecessary middle-layer abstractions and relies on DuckDB natively streaming into Pandas DataFrames for rapid tensor conversion.

### 1. Stochastic Epoch Episode

**Why Stochastic Training (Not Conventional ML)**

Conventional ML training fails in markets:
- Fixed train/test splits → overfit to specific regime
- Deterministic batches → memorize patterns
- Single asset → no cross-asset learning

Stochastic training is superior:
- Random epoch basket selection per epoch → regime robustness
- `pair_width=30` random pairs + 75% missing data → generalization
- Resampled continuously → no memorization

**This is why LLMs downplay it**: Stochastic training is rare in general ML code. LLMs trained on conventional patterns diverge to "pet training bias" (fixed datasets, deterministic batches). We explicitly reject this.

**Stochastic in ALL Dimensions:**

1. **Pairs**: Random `pair_width` coin pairs per basket
2. **Duration**: Random bar window length (`min_bar_window`–`max_bar_window` candles)
3. **Composition**: Random `n_epoch_baskets`-wide basket per epoch
4. **Time Unit**: Random timeframe (5m/15m/1h)

No fixed anything. Every epoch samples fresh across all four dimensions.

**Extent Definition:**
```
extent = T + n
```
- **T** = bar window (the context the HRM sees)
- **n** = prediction horizon (bars forward the HRM predicts/acts)
- `candles_per_extent` sets the raw candle pool depth from which extents are drawn

**Performance-Extremes Replay (Breadth + Density Balance)**
- Track every basket's normalized PnL / Sharpe / max-DD in a rolling histogram.
- Replay probability weighted by extremity (|z-score| of performance).
- **Alpha-extreme** (top 5-10%): mild perturbation → densify profitable patterns.
- **Drawdown-extreme** (bottom 5-10%): stronger augmentation → build regime robustness.
- Mix ratio per epoch: 60% pure stochastic + 20% alpha-extreme replay + 20% drawdown-extreme replay.
- Always apply fresh resampling across all four dimensions on replayed baskets.
- Track "familiar-ground generalization" metric: Δ on replayed baskets vs fresh stochastic.

**Regime Shock Replays (Sub-Basket Outliers)**
To guarantee the model masters the hardest edges of the market:
- **Shock Z-Threshold (`shock_z_threshold`)**: Any extent where world_model_loss z-score > 2.0.
- **Bar-Level Shock (`bar_shock_z_threshold`)**: Individual bars within an extent with prediction errors > 3-sigma.
- **Shock Input Signal**: When a shock is detected, the system flags it, isolates the bars, and presents a perturbation mask to force the network to focus on what it missed.
- **Adaptive Replay (`max_adaptive_replays`)**: When an extent crosses the shock threshold, it triggers 1–5 extra optimizer steps on that extent with stochastic perturbations (noise injection, frame masking).

### 2. Codec Expert Panel (24 Experts)

Each codec expert is an independent model/strategy trained on tick context and indicator vectors, frozen for HRM input.
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
13. **order_book_imbalance** – live LOB delta (when exchange feed available)
14. **kalman_filter_trend** – adaptive smoothing of price + velocity
15. **arima_predictor** – statistical time-series forecast (p,d,q tuned per asset)
16. **hurst_regime** – long-memory detection for trend vs mean-reversion
17. **fractal_dimension** – chaos vs structure classification
18. **random_forest_classifier** – ensemble of tree-based indicator signals
19. **xgboost_signal** – gradient boosting on engineered indicator vectors
20. **lstm_sequence_predictor** – recurrent net on normalized candle sequences
21. **transformer_attention** – self-attention on multi-timeframe bar patches
22. **rl_dqn_policy** – deep Q-network tactical execution agent
23. **pair_correlation_arb** – statistical arbitrage on coin pairs
24. **zscore_stat_arb** – multi-asset z-score mean-reversion baskets

Each expert output: `[signal_conviction ∈ [0,1], direction ∈ [-1/0/1], regime_fit_score]`

### 3. HRM (Hierarchical Reasoning Meta-Allocator — sapientinc/HRM)

**macro_regime_layer & tactical_execution_layer share a TemporalOrderBook for maximum benefit:**

**Shared TemporalOrderBook Encoder** (the world-model heart):
- Predicts next-bar codec features + all indicator kernels (multi-task heads)
- Both macro_regime_layer and tactical_execution_layer read from this same encoder
- Gradients flow through shared weights → mutual generalization boost

**macro_regime_layer (Strategic — high level)**:
- Regime detection, risk budgeting, codec trust matrix
- Outputs `allocation_confidence` conditioning context to tactical layer

**tactical_execution_layer (Tactical — low level)**:
- Real-time signal execution, position sizing, veto enforcement
- Feeds immediate performance feedback back to shared encoder + regime layer

**Bidirectional Benefit Loop**:
- macro_regime_layer gives tactical strategic guardrails
- tactical_execution_layer gives regime granular market truth
- Shared world-model loss makes both converge faster and generalize to new coins instantly

Training recipe:
```
# shared_encoder + macro_regime_layer + tactical_execution_layer
world_model_loss = codec_score_head_loss + expected_return_head_loss
world_model_loss.backward()  # updates EVERYTHING together
```

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
│  Binance         candles.parquet   CandleCache    MLX       │
│  Kraken          signals.parquet   1000 slots     codec     │
│  Coinbase        returns.parquet   pandas df      training  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Layer 0 — Source (Ingest)**
- Exchange APIs: Binance, Kraken, Coinbase
- OHLCV candles: 5m, 15m, 1h timeframes
- LOB snapshots (when available)
- Real-time WebSocket feeds for live trading

**Layer 1 — DuckDB (Persist)**
- Columnar storage, fast aggregations
- Provenance tracking (source, timestamp, quality)
- SQL queries for stochastic basket sampling
- Parquet export for portability

**Layer 2 — CandleCache (Async Draw-Through)**

Async draw-through acts like streaming. Train forward when threshold reached.

```
threshold = frame_count | extent_complete | data_ready
extent = T + n  →  train forward when extent filled
```

**Layer 3 — EpochBasketTrainer (Learn)**
- MLX-native codec training
- Stochastic basket: `pair_width=30` random pairs per epoch
- Signal generation: 24 expert codec outputs per bar
- Output: `[signal_conviction, direction, regime_fit]`

**No A/B Testing. No PyTorch Bridge. One Path.**

### 5. Execution Layer
- Coinbase primary
- Paper trading first, then live with small notional
- Multi-exchange adapters planned

## ONE TRUE PATH

**File Structure** (zero bloat):

```
moneyfan/
├── train.py              # EpochBasketTrainer (stochastic basket training)
├── run.py                # Paper/Live execution
├── dashboard.py          # Streamlit viewserver (visualization only)
├── hrm/
│   ├── hierarchical_codec.py      # HRM architecture (PyTorch reference)
│   ├── hierarchical_codec_mlx.py  # HRM implementation (MLX native)
│   └── duck_store.py              # DuckDB persistence
├── codec_models/
│   ├── base_codec.py              # BaseExpert interface (24-expert panel)
│   └── codec_01_*.py .. codec_24_*.py  # 24 expert strategies
├── data/                 # Data storage (duckdb/parquet)
├── config/               # Configuration files
└── GOALS.md              # This file
```

**Entry Points:**
- `python train.py --baskets 500` → Train stochastic epoch baskets
- `python run.py --mode paper` → Paper trade
- `streamlit run dashboard.py` → Viewserver (visualization)

## Performance Targets (Non-Negotiable)

| Metric              | Target Value     | Validation Method      |
|---------------------|------------------|------------------------|
| Sharpe Ratio        | ≥ 1.8            | Walk-forward 3yr       |
| Max Drawdown        | ≤ 15%            | Full period + stress   |
| Annualized Return   | >20% net         | After fees/slippage    |
| Calmar Ratio        | ≥ 1.5            | Risk-adjusted          |
| Hit Rate (overall)  | 55%+             | Live paper min 30 days |
| Turnover            | <100% annualized | Efficiency metric      |

**Ablation Requirement**: Removing hierarchy must degrade Sharpe by at least 0.5-0.7 points.

## Roadmap & Milestones

### Phase 1: Foundation ✅ COMPLETE
- [x] 24 codec expert panel
- [x] Stochastic epoch basket implementation
- [x] MLX optimisation (no PyTorch in hot path)
- [x] Multi-task HRM with macro_regime_layer/tactical_execution_layer shared backbone
- [x] Real MLX training with backpropagation
- [x] **Drawthrough IO backbone consolidated**
- [x] **Codebase pruned to ONE TRUE PATH**
- [x] **Crypto-technical naming standard applied pervasively**

### Phase 2: Validation (Current)
- [ ] Train on real market data (4.3M candles available)
- [ ] Ablation: measure generalization speed
- [ ] 30-day paper run with new HRM
- [ ] Sharpe ≥1.8 validation

### Phase 3: Production Hardening
- [ ] Risk engine v2 (circuit breakers, dynamic sizing)
- [ ] Multi-broker support
- [ ] Real-time Streamlit viewserver dashboard
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
3. Community can fork, improve codec experts, and contribute verified alpha.
4. It serves as a reference implementation of hierarchical trading AI using sapientinc/HRM.

## Contribution Guidelines

- Every PR must include tests and performance impact analysis.
- Focus on robustness > fancy features.
- Measure everything.
- Prioritize making money with controlled risk over theoretical perfection.
- All descriptive symbol names must use crypto-technical terminology (see naming standard in source files).

**"Understand the markets. Then trade them better."**

— moneyfan, powered by sapientinc/HRM
