# Train 500 Stochastic Bags

Train 500 stochastic bags with live progress tracking, real-time P&L monitoring, and GOALS.md validation.

## Features

- **500 Bags Training**: Train 500 stochastic bags with full progress tracking
- **Live Dashboard**: Real-time curses-based progress dashboard
- **Real-time Metrics**: Live P&L, win rate, drawdown, and bag statistics
- **Stochastic Bag Training**: Random 30 pairs per bag, 100 sequences per bag
- **PyTorch/MLX**: Supports both PyTorch and MLX training frameworks
- **GOALS.md Validation**: Validates results against GOALS.md targets
- **Risk Controls**: Per-trade risk management and portfolio limits

## Usage

### Quick Start

```bash
# Train 500 bags with defaults ($100 per bag, 5s updates)
./train_500_bags.sh

# Custom configuration
./train_500_bags.sh 250 200 10  # 250 bags, $200 per bag, 10s updates
```

### Direct Python Execution

```bash
# Basic training
python3 train_500_bags.py --bags 500 --capital 100 --update-interval 5

# Without dashboard (for background execution)
python3 train_500_bags.py --bags 500 --capital 100 --no-dashboard

# Custom configuration
python3 train_500_bags.py --bags 250 --capital 200 --bag-size 20 --sequences-per-bag 50 --output my_results
```

### Command Line Options

- `--bags`: Number of bags to train (default: 500)
- `--capital`: Initial capital per bag in USD (default: 100)
- `--update-interval`: Seconds between dashboard updates (default: 5)
- `--bag-size`: Size of stochastic bag (default: 30)
- `--sequences-per-bag`: Sequences per bag (default: 100)
- `--epochs`: Number of epochs (default: 5)
- `--output`: Output directory (default: "bag_training_results")
- `--no-dashboard`: Disable progress dashboard
- `--no-mlx`: Disable MLX (use PyTorch)

## Dashboard Controls

- **View**: Real-time progress, P&L, win rate, drawdown, bag statistics
- **Controls**: Press `q` to stop training gracefully
- **Metrics**: Overall progress, current bag progress, recent bag results

## Training Process

1. **Data Loading**: Load all .feather files from `hrm/data/arrow/`
2. **Bag Creation**: For each bag, select 30 random trading pairs
3. **Sequence Training**: Train 100 sequences per bag
4. **Model Updates**: Update model parameters after each sequence
5. **Progress Tracking**: Real-time dashboard updates every 5 seconds
6. **Results Save**: Save all metrics, equity curves, and models

## Output Files

All results are saved to the output directory (default: `bag_training_results/`):

1. **bag_results.csv**: Complete bag-by-bag results
2. **equity_curve.csv**: Equity progression over all bags
3. **bag_history.csv**: Recent bag history (last 500 bags)
4. **metrics.json**: Final performance metrics
5. **config.json**: Training configuration
6. **trained_model.pt**: PyTorch model weights (if PyTorch available)

## Performance Targets (from GOALS.md)

The training validates against GOALS.md targets:

- **Sharpe Ratio**: ≥ 1.8
- **Max Drawdown**: ≥ -15%
- **Annualized Return**: 20%+ net
- **Win Rate**: Target 60%+
- **Risk per Trade**: 1-2% (configurable)
- **Portfolio Limits**: 20% max per symbol, 3-5 positions
- **Hard Stops**: 2% loss max, 5% daily drawdown freeze

## Architecture

The training uses the same architecture as GOALS.md:

1. **Stochastic Bag**: 30 random pairs + USD per bag
2. **Codec Agents**: 24 SOTA crypto strategies (MLX/PyTorch)
3. **HRM Meta-Allocator**: Hierarchical Risk Management
4. **3-Predictor MVP**: 5m Transformer + 15m XGBoost + 1h LightGBM
5. **Vector Store**: 64-dim numpy vectors
6. **Veto Layer**: HRM high-level rejects trades when regime_confidence < 0.75

## Real-time Monitoring

### Dashboard Metrics
- Overall progress: `500/500 bags (100%)`
- Current bag progress: `Bag #123: 45% complete`
- Total P&L: `+$1,234.56`
- Win rate: `62.3% (123/197)`
- Max drawdown: `-8.5%`
- Current bag equity: `$102.34`

### Bag Results
Each bag shows:
- Bag ID
- P&L
- Number of trades
- Win rate
- Final equity
- Sequences trained

## Performance Optimization

### For Large Training Sessions
```bash
# Run in screen/tmux for long sessions
screen -S train_bags
./train_500_bags.sh 1000 500 30  # 1000 bags, $500, 30s updates

# Detach with Ctrl+A D
# Reattach: screen -r train_bags
```

### Background Execution
```bash
# Run in background with logging
nohup ./train_500_bags.sh 500 100 10 > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Parallel Training
```bash
# Train multiple bag sets in parallel
./train_500_bags.sh 250 100 5  # Terminal 1
./train_500_bags.sh 250 200 5  # Terminal 2
```

## Example Training Scenarios

### 1. Quick Test (50 bags)
```bash
python3 train_500_bags.py --bags 50 --capital 100 --update-interval 1
```

### 2. Extended Training (1000 bags)
```bash
python3 train_500_bags.py --bags 1000 --capital 500 --update-interval 30
```

### 3. Small Bag Size
```bash
python3 train_500_bags.py --bags 500 --capital 100 --bag-size 15 --sequences-per-bag 50
```

### 4. High Capital Training
```bash
python3 train_500_bags.py --bags 200 --capital 10000 --update-interval 10
```

## Troubleshooting

### No Data Available
Ensure the `hrm/data/arrow/` directory exists with `.feather` files:
```bash
ls hrm/data/arrow/ | wc -l  # Should show > 100 files
```

### No PyTorch/MLX
The script supports both PyTorch and MLX. If neither is available, the script will use placeholder models.

### Dashboard Issues
If the curses dashboard has issues, run without it:
```bash
python3 train_500_bags.py --no-dashboard
```

### Memory Issues
For large training sessions, reduce batch size and bag size:
```bash
python3 train_500_bags.py --bags 500 --bag-size 15 --sequences-per-bag 50
```

## Notes

- Training 500 bags with 100 sequences each = 50,000 total sequences
- Each sequence trains the model with gradient updates
- Results are saved after each bag completion
- The system respects all GOALS.md risk controls
- Training can be stopped anytime with Ctrl+C or 'q' in dashboard
- Model weights are saved at the end of training