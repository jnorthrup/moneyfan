# HRM Training vs Standard ML Practices

## Executive Summary

The HRM (Hierarchical Reasoning Model) training system in `train.py` implements a **non-standard** training paradigm optimized for financial time series. It diverges significantly from standard ML practices in several key ways.

---

## 1. Data Pipeline Comparison

### Standard ML Practice
```python
# Typical PyTorch DataLoader pattern
dataset = FinancialDataset(symbols=['BTCUSDT', 'ETHUSDT'], ...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### HRM Approach (train.py)
```python
# Episode-based stochastic sampling
def train_episode(self, episode_id, episode_pairs):
    # 1. Load candles for specific pairs (lines 1277)
    df = self.candle_pipeline.load_candles(episode_pairs, None, None)

    # 2. Optional calendar-stochastic sampling (lines 1295-1307)
    if max_extent_days > 0:
        df, extent_meta = sample_stochastic_calendar_extent_df(...)
        # This picks a RANDOM time window between min_days and max_days
        # End timestamp: np.random.randint(0, len(end_candidates))
        # Duration: np.random.randint(min_days, max_days)

    # 3. Take last N candles (lines 1330-1332)
    df = df.iloc[-self.config.candles_per_extent:]

    # 4. Multiple bar windows per episode (lines 1429-1450)
    for bar_seq_i in range(self.config.bar_sequences_per_episode):
        # RANDOM bar window start within symbol range
        start_idx = np.random.randint(range_start, max_start_exclusive)
        batch = codec_features[start_idx:start_idx + bar_window_len]
```

### Key Differences

| Aspect | Standard ML | HRM Approach |
|--------|-------------|--------------|
| **Data sampling** | Sequential batches or fixed windows | Stochastic windows per episode |
| **Batch construction** | Fixed batch size, sequential | Random window start, varying lengths |
| **Dataset split** | Train/val/test splits | Per-episode stochastic splits |
| **Shuffling** | At epoch boundaries | Within each episode |
| **Data source** | Pre-built dataset | Real-time candle loading + DuckDB |

---

## 2. Training Loop Architecture

### Standard ML Practice
```python
# Fixed training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### HRM Approach
```python
# Multi-level stochastic training
for episode_id in range(num_episodes):
    # 1. Select random pairs (lines 1941-1948)
    episode_pairs = np.random.choice(all_pairs, size=pair_width)

    # 2. Load stochastic candles for this episode
    df = load_candles(episode_pairs, ...)  # Random extent

    # 3. Train multiple bar windows per episode (lines 1429-1600)
    for bar_seq_i in range(bar_sequences_per_episode):
        # RANDOM bar window within loaded candles
        start_idx = np.random.randint(...)

        # 4. Multiple training steps per window
        # World-model pretraining (lines 1467-1492)
        world_model_loss = trainer.pretrain_step(batch_mx, ...)
        trainer.optimizer.update(model, grads)

        # 5. Optional trade updates (lines 1515-1530)
        if np.random.random() < trade_update_prob:
            alpha_loss = trainer.trade_step(...)

        # 6. Optional energy updates (lines 1545-1560)
        if np.random.random() < energy_update_prob:
            energy_loss = trainer.energy_step(...)
```

### Key Differences

| Aspect | Standard ML | HRM Approach |
|--------|-------------|--------------|
| **Loop structure** | Epoch → Batch | Episode → Bar Window → Training Step |
| **Batch source** | Fixed dataset | Stochastic sampling per episode |
| **Update frequency** | Every batch | Multiple updates per window |
| **Loss composition** | Single loss per batch | Multiple loss types (pretrain, trade, energy) |

---

## 3. Stochastic Data Verification

### Does HRM Use Real Stochastic Data? **YES, BUT...**

The training IS stochastic at multiple levels:

```python
# Level 1: Episode pair selection (lines 1941-1948)
if bool(getattr(self.config, "reseed_pairs_by_episode", True)):
    np.random.seed(episode_id)  # Reseed per episode
episode_pairs = list(np.random.choice(
    all_pairs,
    size=min(self.config.pair_width, len(all_pairs)),
    replace=False
))

# Level 2: Bar window selection (lines 1439-1446)
selected_range = eligible_ranges[np.random.randint(0, len(eligible_ranges))]
range_start = int(selected_range['start'])
range_end = int(selected_range['end'])
max_start_exclusive = range_end - bar_window_len + 1
start_idx = np.random.randint(range_start, max_start_exclusive)

# Level 3: Calendar-stochastic sampling (lines 428-444)
span_days_requested = int(np.random.randint(min_days_i, max_days_i + 1))
end_idx = int(np.random.randint(0, len(end_candidates)))
end_ts = end_candidates.iloc[end_idx]
start_ts = end_ts - pd.Timedelta(days=span_days_requested)
```

**However, there's a critical issue:**

### Issue: `np.random.seed(episode_id)` - Deterministic "Stochasticity"

```python
# Line 1942: Seeding based on episode_id means:
# - Episode 0: Always same "random" pairs
# - Episode 1: Always same different "random" pairs
# - Episode 2: Always same different "random" pairs
```

This creates **deterministic permutations**, not true randomness. Each episode sees the same "random" data pattern every time you run training.

### Standard ML Practice for True Randomness
```python
# NOT seeding per batch/epoch
# Let the system use true random seeds
for batch in dataloader:
    # True randomness from OS entropy pool
    ...

# OR seed once at start
np.random.seed()  # Use system entropy
```

---

## 4. Data Loading Patterns

### Standard ML Practice
```python
# Pre-load entire dataset
dataset = load_all_data(symbols, start_date, end_date)
dataloader = DataLoader(dataset, ...)

# Training loop
for batch in dataloader:
    # Data already in memory
    ...
```

### HRM Approach
```python
# Load on-demand per episode
def train_episode(self, episode_id, episode_pairs):
    # Each episode loads fresh data from disk/DuckDB
    df = self.candle_pipeline.load_candles(episode_pairs, None, None)
    # This hits DuckDB or Parquet files every time

    # Then computes features
    codec_features = self.candle_pipeline.compute_signals(df, ...)
```

### Concerns with HRM Approach

| Issue | Impact |
|-------|--------|
| **I/O bottleneck** | Every episode hits disk/DB |
| **Cache stampede** | Same candles loaded repeatedly |
| **No dataset-level split** | Risk of data leakage across episodes |
| **Inconsistent preprocessing** | Features computed fresh each time |

---

## 5. Model Architecture Patterns

### Standard ML Practice
```python
# Fixed architecture
class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

# Single forward pass
output = model(batch)
loss = criterion(output, target)
```

### HRM Approach (MLX)
```python
# Hierarchical model with memory state
class MLXHierarchicalCodec(nn.Module):
    def forward(self, bar_codec_features, memory=None, mode="pretrain"):
        # Memory state carries across sequences
        temporal_ob_state, regime_state, tactical_state = (
            memory if memory else (None, None, None)
        )

        # Multiple attention heads
        regime_out = regime_attention(bar_codec_features, regime_state)
        tactical_out = tactical_attention(bar_codec_features, tactical_state)

        # Gradient stopping for BPTT
        new_memory = (
            mx.stop_gradient(temporal_ob_state),
            mx.stop_gradient(regime_state),
            mx.stop_gradient(tactical_state)
        )
        return output, new_memory

# Training with memory
world_model_loss, hrm_memory = trainer.pretrain_step(
    batch_mx, memory=hrm_memory
)
```

### Key Differences

| Aspect | Standard ML | HRM Approach |
|--------|-------------|--------------|
| **State handling** | Stateless | Stateful (memory across sequences) |
| **Gradient flow** | Through entire network | BPTT with gradient stopping |
| **Architecture** | Fixed layers | Hierarchical attention |
| **Loss computation** | Single forward pass | Multiple specialized heads |

---

## 6. Optimizer and Training Mechanics

### Standard ML Practice
```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### HRM Approach
```python
# MLX optimizers with coalescing
trainer = MLXBasketTrainer(config)  # Sets up optimizer

# Coalesced updates (lines 1467-1477)
if self.config.replay_coalescing:
    for perturbed_bar_batch in chunk:
        loss, memory = trainer.pretrain_step(
            batch_mx, memory=memory, auto_eval=False
        )
    # Single optimizer update for multiple batches
    trainer.flush_updates(total_loss, memory=memory)

# With gradient clipping (lines 693-707)
def clip_gradients(self, grads, max_norm=1.0):
    total_norm_sq = sum(mx.square(g).sum() for g in grads.values())
    total_norm = mx.sqrt(total_norm_sq) + 1e-8
    scale = mx.minimum(max_norm / total_norm, mx.array(1.0))
    return {k: v * scale for k, v in grads.items()}

# Training steps
trainer.pretrain_step(batch_mx, clip_gradients=True, max_gradient_norm=1.0)
trainer.trade_step(batch_mx, realized_returns_mx, clip_gradients=True)
trainer.energy_step(batch_mx, realized_returns_mx, clip_gradients=True)
```

### Comparison

| Aspect | Standard ML | HRM Approach |
|--------|-------------|--------------|
| **Optimizer update** | Every batch | Coalesced (multiple batches) |
| **Gradient clipping** | Optional, global | Per-parameter, configurable |
| **Learning rate** | Fixed or scheduled | Configurable per optimizer |
| **Multiple loss terms** | Combined with weights | Separate training steps |
| **State preservation** | N/A | Memory passed between steps |

---

## 7. Data Stochasticity: The Real Issue

### What HRM Does (Deterministic "Stochasticity")
```
Episode 0:
  - np.random.seed(0) → Always same "random" pairs
  - Fixed bar window lengths
  - Deterministic sampling within symbol ranges

Episode 1:
  - np.random.seed(1) → Always different, but same each run
  - Same deterministic pattern

Result: Same "random" data every training run
```

### What Standard Practice Would Do
```python
# Option 1: True randomness (no seeding)
for episode in range(num_episodes):
    episode_pairs = random.choice(all_pairs)  # True entropy

# Option 2: Global seed at start only
np.random.seed(42)
for episode in range(num_episodes):
    episode_pairs = np.random.choice(all_pairs)  # Different each run

# Option 3: Reservoir sampling for true stochasticity
for episode in range(num_episodes):
    # Sample with probability, not fixed size
    episode_pairs = [p for p in all_pairs if np.random.random() < 0.5]
```

### Problems with Current HRM Approach

1. **No true randomness**: `seed(episode_id)` is deterministic
2. **No validation split**: All data used for training
3. **No test set separation**: Risk of overfitting to specific time periods
4. **Inconsistent sampling**: Different episodes may sample overlapping data
5. **No cross-validation**: No way to assess generalization

---

## 8. Recommended Improvements

### Add True Stochastic Sampling
```python
# Remove np.random.seed(episode_id) - let OS provide entropy
# OR seed once at training start

# Add proper data splits
TRAIN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...][:20]
VAL_SYMBOLS = ['ADAUSDT', 'DOTUSDT', 'MATICUSDT', ...][:5]
TEST_SYMBOLS = ['AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', ...][:5]
```

### Add Time-Based Splits
```python
# Standard ML practice for time series
train_data = df[df['timestamp'] < '2024-01-01']
val_data = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2024-06-01')]
test_data = df[df['timestamp'] >= '2024-06-01']
```

### Add Proper Dataset Class
```python
class StochasticWindowDataset:
    def __init__(self, symbols, start_date, end_date, window_size):
        self.data = load_all_data(symbols, start_date, end_date)
        self.window_size = window_size

    def __getitem__(self, idx):
        # True random sampling without seeding
        start = np.random.randint(0, len(self.data) - self.window_size)
        return self.data[start:start + self.window_size]
```

---

## 9. Summary

### HRM Approach Strengths
- ✅ Real data from DuckDB/Parquet (not synthetic)
- ✅ Multi-level stochastic sampling design
- ✅ Hierarchical architecture for financial time series
- ✅ Multiple loss objectives (pretrain, trade, energy)
- ✅ Stateful training with memory

### HRM Approach Weaknesses
- ❌ **Deterministic "stochasticity"** via `seed(episode_id)`
- ❌ **No data splits** (train/val/test)
- ❌ **No cross-validation**
- ❌ **I/O bottleneck** from on-demand loading
- ❌ **Risk of data leakage** across episodes
- ❌ **No time-based validation**

### Comparison Verdict

| Criteria | Standard ML | HRM (Current) |
|----------|-------------|---------------|
| **Data randomness** | ✅ True random | ❌ Deterministic |
| **Train/val/test split** | ✅ Yes | ❌ No |
| **Time series handling** | ✅ Time-based split | ⚠️ Window-based only |
| **Cross-validation** | ✅ K-fold available | ❌ No |
| **I/O efficiency** | ✅ Pre-loaded | ⚠️ On-demand |
| **Model architecture** | ⚠️ Standard | ✅ Hierarchical |
| **Domain adaptation** | ⚠️ Generic | ✅ Finance-specific |

---

## 10. Recommendation

The HRM training system is **not following standard ML practices** in several critical ways:

1. **Replace `np.random.seed(episode_id)`** with true randomness or a single global seed
2. **Add train/val/test splits** based on time periods
3. **Add proper dataset class** with pre-loading and caching
4. **Add cross-validation** for robust evaluation
5. **Consider data partitioning** by symbol to prevent leakage

The system IS using real stochastic data (real OHLCV candles), but the sampling methodology is deterministic rather than truly random, which may affect model generalization.
