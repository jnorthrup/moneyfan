# HRM Data Split Guide

## Overview

The training system now follows **standard ML practices** for data splitting and stochastic sampling.

## Quick Start

### Default Behavior (True Randomness + Symbol Splits)
```bash
python train.py --episodes 1000
```
- Uses **true randomness** (system entropy)
- Splits by **symbols**: 70% train, 15% val, 15% test
- Progress shows: `[TRAIN]`, `[VAL]`, or `[TEST]`

### Reproducible Training (Fixed Seed)
```bash
python train.py --episodes 1000 --random-seed 42
```
- Same "random" data every run
- Useful for debugging and comparison

### Legacy Deterministic Behavior (NOT RECOMMENDED)
```bash
python train.py --episodes 1000 --no-true-randomness
```
- Uses `np.random.seed(episode_id)` (deterministic)
- Every run sees same data pattern
- ⚠️ Deprecated, only for backward compatibility

## Data Split Modes

### Split by Symbols (Recommended for Most Cases)
```bash
python train.py --episodes 1000 --split-mode symbols --train-split 0.70 --val-split 0.15
```

**How it works:**
1. All 30 symbols shuffled randomly
2. Split into train (70%), val (15%), test (15%)
3. Episodes assigned round-robin: train → val → test → train...

**Output:**
```
[DATA_SPLIT] Split mode: symbols
[DATA_SPLIT] Train symbols (21): BTCUSDT, ETHUSDT, SOLUSDT, ...
[DATA_SPLIT] Val symbols (4): AVAXUSDT, LINKUSDT, ...
[DATA_SPLIT] Test symbols (5): DOTUSDT, MATICUSDT, ...

Episode 1/1000 [TRAIN] pnl=$9.69 ...
Episode 2/1000 [VAL] pnl=$10.23 ...
Episode 3/1000 [TEST] pnl=$8.45 ...
Episode 4/1000 [TRAIN] ...
```

### Split by Time (For Time Series Validation)
```bash
python train.py --episodes 1000 --split-mode time --time-split-fraction 0.70
```

**How it works:**
1. Loads historical data
2. Splits time periods: train (70%), val (15%), test (15%)
3. Each episode samples from appropriate time range

⚠️ **Note**: Time-based splits require time range determination in data loading

## Randomness Modes

### True Randomness (Default)
```bash
python train.py --episodes 1000 --use-true-randomness
```
- Uses OS entropy pool
- Different "random" pairs every run
- ✅ Recommended for production

### Fixed Seed (Reproducible)
```bash
python train.py --episodes 1000 --random-seed 42
```
- Same pairs every run
- Useful for debugging
- ⚠️ Still uses real market data, just deterministic sampling

### Deterministic (Legacy)
```bash
python train.py --episodes 1000 --no-true-randomness
```
- Uses `np.random.seed(episode_id)`
- ❌ Deprecated, not recommended

## Checkpoint Resume with Splits

When resuming, the checkpoint preserves:
- ✅ Train/val/test symbol assignments
- ✅ Randomness configuration
- ✅ Data split mode

```bash
# Start training
python train.py --episodes 1000 --split-mode symbols --train-split 0.70

# ... training runs for 100 episodes, you hit Ctrl-C ...

# Resume from checkpoint
python train.py --episodes 1000 --resume-checkpoint
```

The resumed training will:
1. Load saved symbol assignments
2. Continue with same split configuration
3. Maintain train/val/test progression

## Expected Output

### Before (Deterministic - Problematic)
```
Episode 1/1000 pnl=$9.69 ...
Episode 1/1000 pnl=$9.69 ...  # Same output every run!
```

### After (True Randomness - Correct)
```
Episode 1/1000 [TRAIN] pnl=$9.69 ...
Episode 2/1000 [VAL] pnl=$10.23 ...
Episode 3/1000 [TEST] pnl=$8.45 ...
Episode 4/1000 [TRAIN] pnl=$11.50 ...  # Different each run!
```

## Best Practices

### For Production Training
```bash
python train.py \
  --episodes 100000 \
  --split-mode symbols \
  --train-split 0.70 \
  --val-split 0.15 \
  --use-true-randomness
```

### For Reproducible Experiments
```bash
python train.py \
  --episodes 10000 \
  --split-mode symbols \
  --train-split 0.70 \
  --random-seed 42
```

### For Cross-Validation (Future)
```bash
# Run 5 folds
for fold in {0..4}; do
  python train.py \
    --episodes 20000 \
    --random-seed $fold \
    --fold $fold \
    --total-folds 5
done
```

## Migration Guide

### Old Command
```bash
python train.py --episodes 1000 --fully-stochastic-pair-sampling
```

### New Command
```bash
python train.py --episodes 1000 --use-true-randomness
```

### Old vs New

| Aspect | Old (Before) | New (After) |
|--------|--------------|-------------|
| **Randomness** | Deterministic (seed(episode_id)) | True random (system entropy) |
| **Data splits** | None - all training | Train/Val/Test splits |
| **Progress output** | Generic | Shows [TRAIN]/[VAL]/[TEST] |
| **Reproducibility** | Every run same | Use --random-seed 42 |
| **Checkpoint saves** | Basic | Includes split config |

## Command Reference

```
--split-mode {symbols,time}    How to split data (default: symbols)
--train-split FLOAT            Training fraction (default: 0.70)
--val-split FLOAT              Validation fraction (default: 0.15)
--test-split FLOAT             Test fraction (default: 0.15, auto-calc)
--time-split-fraction FLOAT    Time fraction for train (default: 0.70)
--use-true-randomness          Use system entropy (default: True)
--no-true-randomness           Use episode_id seeding (DEPRECATED)
--random-seed INT              Fixed seed for reproducibility
```

## Examples

### Example 1: Balanced 70/15/15 split with true randomness
```bash
python train.py \
  --episodes 100000 \
  --split-mode symbols \
  --train-split 0.70 \
  --val-split 0.15 \
  --use-true-randomness
```

### Example 2: Strict training only (no val/test)
```bash
python train.py \
  --episodes 100000 \
  --split-mode symbols \
  --train-split 1.00 \
  --val-split 0.00 \
  --test-split 0.00
```

### Example 3: Reproducible with fixed seed
```bash
python train.py \
  --episodes 100000 \
  --random-seed 42
```
