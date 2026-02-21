# Bag Trading Marathon

A long-running trading session with real-time progress monitoring for the HRM system.

## Features

- **Continuous Trading**: Run for configurable hours (default: 24 hours)
- **Live Progress Dashboard**: Real-time tracking of P&L, positions, and performance
- **Risk Controls**: Maximum positions, daily drawdown limits, hard stop losses
- **Stochastic Bag Trading**: 30 random pairs + USD, resampled hourly
- **HRM Meta-Allocator**: 24 codec agents with hierarchical veto layer
- **No Timeout**: Configurable update interval (default: 5 seconds)
- **Real-time Metrics**: Live tracking of equity, drawdown, win rate, etc.

## Usage

### Quick Start

```bash
# Start marathon with defaults (24 hours, $1000 capital, 5s updates)
./start_marathon.sh

# Custom configuration
./start_marathon.sh 12 500 10  # 12 hours, $500 capital, 10s updates
```

### Direct Python Execution

```bash
# Basic marathon
python bag_trading_marathon.py --hours 24 --capital 1000 --update-interval 5

# Without dashboard (for background execution)
python bag_trading_marathon.py --hours 24 --capital 1000 --no-dashboard

# Custom output directory
python bag_trading_marathon.py --hours 6 --capital 2000 --output my_marathon_results
```

### Command Line Options

- `--hours`: Number of hours to run (default: 24)
- `--capital`: Initial capital in USD (default: 100)
- `--update-interval`: Seconds between dashboard updates (default: 5)
- `--codecs`: Number of codec agents (default: 24)
- `--bag-size`: Size of stochastic bag (default: 30)
- `--output`: Output directory (default: "marathon_results")
- `--no-dashboard`: Disable progress dashboard
- `--no-mlx`: Disable MLX inference (use PyTorch/placeholder)

## Dashboard Controls

- **View**: Real-time P&L, positions, bag contents, regime confidence
- **Controls**: Press `q` to stop the marathon gracefully
- **Metrics**: Total returns, Sharpe ratio, max drawdown, win rate

## Risk Controls

- **Max Positions**: 5 concurrent positions (configurable)
- **Daily Drawdown**: 5% limit (configurable)
- **Hard Stop Loss**: 2% per trade (configurable)
- **Veto Layer**: HRM regime confidence < 0.75 rejects trades

## Output Files

All results are saved to the output directory (default: `marathon_results/`):

1. **trades.csv**: Complete trade history
2. **equity_curve.csv**: Equity over time
3. **metrics.json**: Performance metrics and statistics
4. **marathon.log**: Detailed execution log

## Performance Targets (from GOALS.md)

- **Sharpe Ratio**: ≥ 1.8
- **Max Drawdown**: ≥ -15%
- **Annualized Return**: 20%+ net
- **Win Rate**: Target 60%+

## Architecture

The marathon uses the same components as the paper trading system:

1. **StochasticBag**: Manages 30 random trading pairs + USD
2. **CodecAgent**: 24 SOTA crypto strategies (MLX-based)
3. **HRMMetaAllocator**: Hierarchical Risk Management with veto layer
4. **RiskManager**: Position sizing and risk controls
5. **ProgressDashboard**: Real-time curses-based UI

## Monitoring

### Live Metrics
- Equity and P&L
- Current drawdown
- Win rate
- Position count
- Regime confidence
- Veto count

### After Session
- Review `metrics.json` for final performance
- Analyze `equity_curve.csv` for drawdown patterns
- Examine `trades.csv` for trade quality

## Troubleshooting

### No MLX Support
If MLX is not available (non-Mac systems), use `--no-mlx`:
```bash
python bag_trading_marathon.py --no-mlx
```

### No Data Available
Ensure the `hrm/data/arrow/` directory exists with `.feather` files for trading pairs.

### Dashboard Issues
If the curses dashboard has issues, run without it:
```bash
python bag_trading_marathon.py --no-dashboard
```

### Background Execution
For long marathons, run in screen/tmux:
```bash
screen -S marathon
python bag_trading_marathon.py --hours 48 --capital 10000 --update-interval 30
# Detach with Ctrl+A D
# Reattach: screen -r marathon
```

## Example Marathon Scenarios

### 1. Quick Test (1 hour)
```bash
python bag_trading_marathon.py --hours 1 --capital 1000 --update-interval 1
```

### 2. Extended Session (48 hours)
```bash
python bag_trading_marathon.py --hours 48 --capital 10000 --update-interval 30
```

### 3. Background Execution
```bash
nohup python bag_trading_marathon.py --hours 24 --capital 5000 --no-dashboard > marathon.log 2>&1 &
```

### 4. Multiple Parallel Sessions
```bash
# Terminal 1
python bag_trading_marathon.py --hours 24 --capital 1000 --output marathon1

# Terminal 2  
python bag_trading_marathon.py --hours 24 --capital 2000 --output marathon2
```

## Notes

- Marathon runs continuously without the 10-minute timeout issue
- Update interval controls how often metrics are calculated and displayed
- For production, consider running with `--no-dashboard` and monitoring logs
- The system will respect all GOALS.md targets and risk controls
- Results are saved even if the marathon is interrupted (Ctrl+C)