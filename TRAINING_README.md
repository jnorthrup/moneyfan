# Unified Training System

## Overview

Consolidated training system with a single data pipeline and Streamlit dashboard.

**Data Flow:** Source → DuckDB → Pandas → Cache → Trainer

## Usage

### Command Line Training

```bash
# Train 500 bags
python3 train.py --bags 500 --capital 100

# Train with custom settings
python3 train.py --bags 100 --capital 1000 --bag-size 20
```

### Streamlit Dashboard

```bash
# Launch interactive dashboard
streamlit run dashboard.py
```

The dashboard provides real-time visualization:
- Cumulative PnL chart
- Win rate distribution
- Capital growth tracking
- Per-bag PnL analysis
- Summary statistics

## Architecture

### Components

1. **DataCache**: LRU cache for DataFrame storage
2. **DataPipeline**: Unified data loading (DuckDB → Pandas → Cache)
3. **UnifiedTrainer**: Single trainer class with MLX backend
4. **TrainingConfig**: Centralized configuration

### Data Flow

```
Source (Binance/Kraken/etc)
    ↓
DuckDB (persistent storage)
    ↓
Pandas (in-memory processing)
    ↓
Cache (LRU, avoid recomputation)
    ↓
Trainer (MLX codec training)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bags` | 500 | Number of stochastic bags to train |
| `capital` | 100.0 | Starting capital per bag |
| `bag_size` | 30 | Number of symbols per bag |
| `sequences_per_bag` | 100 | Training sequences per bag |
| `min_seq_len` | 64 | Minimum sequence length |
| `max_seq_len` | 256 | Maximum sequence length |
| `epochs` | 3 | Training epochs per bag |
| `learning_rate` | 1e-4 | Learning rate |
| `cache_size` | 1000 | Maximum cached DataFrames |

## Output

### Checkpoints

Saved every 10 bags to `training_checkpoint.json`:
- Completed bag count
- Current results
- Timestamp

### Final Results

Saved to `training_results.json`:
- All bag results
- Summary statistics
- Total PnL and win rate

## Deleted Files

The following redundant training scripts have been consolidated and removed:

- `train_500_bags_duckdb.py`
- `train_ab_independent.py`
- `train_ab_test.py`
- `train_all.py`
- `train_hierarchical_codec_viz.py`
- `train_hrm_noveto.py`
- `train_hrm.py`
- `train_mlx.py`
- `train_real_data.py`
- `backbone_duck_trainer.py`
- `backbone_trainer.py`
- `stochastic_train.py`
- `bag_trading_marathon.py`
- `binance_stochastic_bag_trainer.py`
- `demo_emulated_training.py`
- `run_historical_training.py`
- `simple_training_session.py`
- `weekly_coinbase_bags.py`

All functionality has been consolidated into:
- `train.py` - Unified training system
- `dashboard.py` - Streamlit visualization

## No A/B Testing

This unified system focuses on single-path training with MLX. A/B testing between PyTorch and MLX has been removed for simplicity.
