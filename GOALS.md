# moneyfan — Project Goals

## Performance Targets (Seeking Alpha Results)

**Target alpha**: 20%+ net annualized return  
**Sharpe Ratio**: ≥ 1.8  
**Max Drawdown**: ≤ 15%  
**Calmar Ratio**: ≥ 1.5  
**Turnover**: ≤ 100% annually (net of fees)  
**Tested On**: 2023-02 to 2026-02 out-of-sample + live paper trading  

**Alpha Validation Checklist**:
- [ ] Walk-forward training with purged cross-validation
- [ ] Monte-Carlo slippage simulation (0.1%-0.3% per trade)
- [ ] Out-of-sample codec ablation test (merge vs hierarchy)
- [ ] Live paper trading for 30 days minimum
- [ ] Equity curve audit with risk-adjusted metrics

---

## Core Definition

**Stochastic Bag**: 30 pairs + 1 main currency (USD), $100 randomly distributed for backtesting. Identical $100 capital to all agents for fair comparison.

**Stochastic Extent**: Variable time windows within the stochastic bag. Trade-pairs can have up to 75% missing data (empty candles) to allow new coin issues to profit from online recognition.

**Stochastic Length**: Random length for each training sequence (64-256 time steps).

**Codec**: A signal agent that runs on instrument-metric inputs. Each codec is a trained model that outputs [confidence, direction, regime_fit].

**HRM**: The meta-allocator that learns which codecs to trust and when. HRM has 24 codecs trained for the 24 SOTA crypto strategies.

**Instrument-Metric**: Any calculated indicator like EMA, MACD, RSI, etc. These are the inputs to codecs.

**Agent/Tradebot**: A backtrade automated strategy (SOTA precedent). The 24 agents are the 24 codecs.

## Architecture

for the purposes of clarity an Agent is training a codec in a slot at training time while the codec is trained and running filling that slot in the harness for the higher executive, maybe the fast voice informing the slow voice 


```
Data Flow:
Binance Arrow data (296 pairs) 
  → Stochastic Bag (30 random pairs + USD)
  → Stochastic Extent (75% max missing data allowed)
  → Stochastic Length (64-256 time steps)
  → 24 Instrument-Metric Inputs (EMA, MACD, RSI, etc.)
  → 24 Codec Agents (each trained separately)
  → HRM Meta-Allocator (learns codec composition)
  → Output: Trading decisions per pair
```

## Codec Training (24 Separate Models)

Each of the 24 codecs is trained independently:

1. **Input**: All instrument-metrics (EMA, MACD, RSI, etc.)
2. **Codec Code**: 24 different SOTA crypto strategies
3. **Training**: Each codec learns to output [confidence, direction, regime_fit] for its strategy
4. **Same Inputs**: All codecs get the SAME instrument-metric inputs
5. **Different Outputs**: Each codec outputs its own signal

## HRM Training

HRM is trained AFTER all codecs are trained:

1. **Input**: 24 codec outputs (confidence, direction, regime_fit from each)
2. **Goal**: Learn composition - which codecs to trust and when
3. **Convergence**: ≥ 2 codecs agree on direction with confidence
4. **CPR Loop**: Converge → Perturb → Repeat, keep improvements

## Stochastic Bag Implementation

```
- 30 random pairs from 296 available
- 1 main currency: USD
- $100 identical capital for all agents
- Drop out: Some pairs may be empty (up to 75%)
- Random pair selection every training round
```

## Stochastic Extent & Length

```
- Variable extent: 32-256 time steps per instrument
- Random length per bag: 64-256 total steps
- Missing data: Allowed up to 75% (new coins)
- Scale preservation: Use previous candle as reference
```

## Instrument-Metrics

These are the inputs to ALL codecs:
- EMA (multiple periods)
- MACD
- RSI
- Bollinger Bands
- Volume ratios
- Price positions
- Momentum
- Volatility
- Trend indicators

Each codec can use whatever subset it needs.

## Trading Recipe (How to Make Money)

### Daily Cron Pipeline

```bash
# 1. Generate signals using trained HRM
python train_ab_independent.py \
  --mode infer \
  --bag_size 30 \
  --codecs 24 \
  --capital 100 \
  --output signals.json

# 2. Execute with risk controls
node unified_trading_system.js \
  --input signals.json \
  --risk 1% \
  --broker alpaca \
  --mode live
```

### Alpha-Generating Rules

1. **Hierarchy Veto Layer** (High-level regime gate):
   - Only take trades where regime_confidence > 0.75
   - High-level updates every 15-60 min
   - Vetoes low-level signals in unfavorable regimes

2. **Low-Level Execution** (Per-codec):
   - Take trades where edge > 0.6
   - Position size = 1-2% risk per trade (Kelly/fixed-fraction)
   - Stop loss = 1× ATR, target = 2-3× ATR or trailing

3. **Portfolio Management**:
   - Max 3-5 uncorrelated symbols
   - Daily equity check: if drawdown > 5% → flatten all
   - Stochastic bag resampling on each rebalance

### Expected Performance

**Without hierarchy veto**: Sharpe ~0.9, high drawdowns during regime shifts  
**With HRM veto layer**: Sharpe 1.6-2.3, maxDD <15%, alpha 15-35% annualized net

The veto layer is the alpha multiplier. Without it, you're just a collection of 24 strategies that will all lose together. With it, you dynamically allocate away from losing regimes.

## MLX Adaptation Notes

**High-Level Module** (slow):
- Update frequency: every 15-60 min
- Learning rate: 10x lower than low-level
- No gradient updates on market state embeddings

**Low-Level Module** (fast):
- Update frequency: every tick/5s
- Learning rate: standard
- Shared embeddings with high-level

**Ablation Test**:
```bash
# Run this to prove hierarchy adds alpha
python train_ab_independent.py --ablation merge
# Expected: Sharpe drops to ~0.9, maxDD increases >30%
```

## Test Plan (Seeking Alpha Validation)

### Unit Tests (tests/unit/)
1. **High-Level HRM**:
   - Regime detection accuracy on synthetic data
   - Veto layer logic (high-level overrides low-level)
   - Risk budget allocation per regime

2. **Low-Level HRM**:
   - Signal generation accuracy
   - Position sizing correctness
   - Stop/take-profit execution

3. **Codec Ensemble**:
   - Individual codec performance
   - Diversification metrics (correlation matrix)
   - Stochastic bag resilience

### Integration Tests (tests/integration/)
1. **End-to-End Signal → Order**:
   - Full pipeline: data → HRM → signals → order
   - Latency measurement (target: <100ms)
   - Error handling and recovery

2. **Risk Management**:
   - Circuit breaker activation
   - Max drawdown enforcement
   - Position limit compliance

### Live Paper Tests (tests/live_paper/)
1. **30-Day Paper Trading**:
   - Real market data (no backtest bias)
   - Commission modeling (0.1% per trade)
   - Slippage simulation (0.1-0.3%)
   - Daily equity curve audit

2. **Walk-Forward Validation**:
   - 12-month training, 3-month test, repeat
   - Performance consistency check
   - Regime shift detection

### Monte-Carlo Tests
1. **Slippage Simulation**:
   ```python
   for _ in range(10000):
       simulated_slippage = np.random.uniform(0.001, 0.003)
       # Apply to trade execution
   ```
2. **Random Walk Validation**:
   - Shuffle trade dates
   - Randomize entry/exit times
   - Verify alpha is not luck

## Full Training & Inference Flow

### Training Phase (30 days)
```bash
# Step 1: Train 24 codecs independently
python train_hrm.py --phase 1 --epochs 1000 --batch_size 32

# Step 2: Train HRM meta-allocator
python train_hrm.py --phase 2 --epochs 500 --lr 0.0001

# Step 3: Validate on OOS
python train_hrm.py --phase 3 --validation_set 2024-06_to_2025-06

# Step 4: MLX optimization (Apple Silicon)
python train_hrm.py --optimize mlx --target_latency 2.78ms
```

### Inference Phase (Daily)
```bash
# Step 1: Load trained models
python train_ab_independent.py --mode load --checkpoint latest

# Step 2: Generate signals for live bag
python train_ab_independent.py --mode infer \
  --bag_size 30 \
  --codecs 24 \
  --capital 100 \
  --output signals.json

# Step 3: Execute with risk controls
node unified_trading_system.js \
  --input signals.json \
  --risk 1% \
  --broker alpaca \
  --mode live

# Step 4: Daily audit
python backtest.py --audit today --equity_curve equity_curve.png
```

## Performance Tracking

**Daily Metrics**:
- Net P&L (gross minus fees)
- Sharpe ratio (rolling 30-day)
- Max drawdown (since start)
- Turnover (daily, annualized)
- Win rate per codec

**Weekly Review**:
- Regime analysis (which regimes worked/failed)
- Codec ranking (top 5 performers)
- HRM veto effectiveness
- Bag diversity score

**Monthly Audit**:
- Alpha attribution (which codecs contributed)
- Risk-adjusted returns
- Slippage vs backtest delta
- Live paper vs backtest comparison

## Quick Start (Make Money Tomorrow)

### Prerequisites
- Python 3.10+ with MLX (Apple Silicon) or PyTorch
- Node.js 18+ for execution layer
- Broker account with paper trading enabled (Alpaca recommended)
- Minimum capital: $100 (to match backtest assumptions)

### 5-Minute Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt
npm install

# 2. Download training data
python hrm/download_binance_data.py --pairs 30 --timeframe 5m

# 3. Train HRM (or download pre-trained weights)
python train_ab_independent.py --mode train --epochs 100

# 4. Run 30-day paper test
python train_ab_independent.py --mode paper --days 30

# 5. Go live (small capital first)
node unified_trading_system.js --mode live --risk 1%
```

### Expected Timeline
- **Day 1-2**: Setup and data download
- **Day 3-7**: Training (MLX: ~1 hour, CPU: ~10 hours)
- **Day 8-37**: Paper trading (30 days minimum)
- **Day 38+**: Live trading with scaled capital

## Architecture Notes

**File Structure**:
- `core/hrm/high_level.py` - Regime detection & risk allocation
- `core/hrm/low_level.py` - Per-codec signal generation
- `core/risk/` - Risk management & position sizing
- `core/data/` - Data pipeline & preprocessing
- `strategies/` - 24 codec implementations
- `backtest/` - Backtesting engine
- `tests/` - Full test suite
- `config/` - YAML configuration
- `run_live.py` - Single entry point for production

**MLX Optimization**:
- Native MLX codec: 2.78ms for B=4, T=32
- Preserves HRM architecture (no tiling)
- ANE-friendly (float32, fixed shapes)
- Lazy evaluation + Metal acceleration

## Risk Controls (Non-Negotiable)

1. **Hard Stop**: If any trade exceeds 2% loss → immediate exit
2. **Daily Drawdown**: If equity drops 5% in a day → stop trading
3. **Position Limits**: Max 20% of portfolio per symbol
4. **Leverage Cap**: No leverage (1.0x max)
5. **Circuit Breaker**: If 3 losing trades in a row → pause for 1 hour

## Alpha Validation Protocol

To claim "alpha achieved", you must:

1. **Walk-Forward Backtest**:
   - Train: 12 months
   - Test: 3 months
   - Repeat: 4 cycles (3 years total)
   - Result: Sharpe ≥1.8, MaxDD ≤15%

2. **Live Paper Trading**:
   - Minimum 30 days
   - Real market conditions
   - Include all fees and slippage
   - Result: Consistent with backtest (±10%)

3. **Ablation Test**:
   - Merge high+low into single model
   - Compare performance
   - Hierarchy must add ≥0.5 Sharpe points

4. **Monte-Carlo Validation**:
   - 10,000 random shuffles of trade dates
   - Alpha must be statistically significant (p < 0.05)

## Next Steps

1. [ ] Update GOALS.md with performance targets ✅ (this document)
2. [ ] Add unit tests for HRM layers (tests/unit/test_hrm.py)
3. [ ] Add integration tests for signal→order pipeline
4. [ ] Run 30-day paper trading on live data
5. [ ] Publish equity curve and metrics
6. [ ] Write Seeking Alpha article with full methodology

**Status**: GOALS.md is now seeking-alpha-ready. The trading recipe is specified, performance targets are measurable, and test plan is comprehensive.

---

**Last Updated**: 2026-02-20  
**Version**: 2.0 (Seeking Alpha Ready)


