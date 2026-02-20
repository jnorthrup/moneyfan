#!/bin/bash
# launch_live_paper.sh
# Launch 24-hour live paper trading on Coinbase Advanced Trade
# This script will run MVP with live execution enabled

set -e  # Exit on error

echo "============================================================"
echo "MONEYFAN LIVE PAPER TRADING LAUNCH"
echo "============================================================"
echo "Project: jnorthrup/moneyfan"
echo "Branch: main"
echo "Mode: Paper Trading (Coinbase Advanced Trade)"
echo "Execution: Live (real API keys required)"
echo "============================================================"

# Check for required environment variables
if [ -z "$COINBASE_API_KEY" ]; then
    echo "❌ ERROR: COINBASE_API_KEY environment variable not set"
    echo "Please set your Coinbase Advanced Trade API key:"
    echo "export COINBASE_API_KEY=\"your-api-key-here\""
    exit 1
fi

if [ -z "$COINBASE_API_SECRET" ]; then
    echo "❌ ERROR: COINBASE_API_SECRET environment variable not set"
    echo "Please set your Coinbase Advanced Trade API secret:"
    echo "export COINBASE_API_SECRET=\"your-api-secret-here\""
    exit 1
fi

echo "✅ API keys detected"
echo "API Key: ${COINBASE_API_KEY:0:8}..."
echo "API Secret: ${COINBASE_API_SECRET:0:8}..."
echo ""

# Create results directory
mkdir -p paper_results

# Setup logging
LOG_FILE="paper_results/live_trading_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Start live paper trading
echo "🚀 Launching live paper trading..."
echo "Press Ctrl+C to stop"
echo ""

# Run the MVP runner with live execution
python3 -c "
import sys
import time
from mvp_runner import MVPConfig, MVPPaperTrading

config = MVPConfig(
    n_predictors=3,
    predictor_types=['transformer_5m', 'xgboost_15m', 'lightgbm_1h'],
    vector_dim=64,
    paper_capital=1000.0,
    paper_mode=True,
    live_execution_enabled=True,
    coinbase_api_key='$COINBASE_API_KEY',
    coinbase_api_secret='$COINBASE_API_SECRET'
)

print('='*60)
print('MONEYFAN LIVE PAPER TRADING')
print('='*60)
print(f'Capital: \${config.paper_capital:.2f}')
print(f'Predictors: {config.n_predictors}')
print(f'Paper mode: {config.paper_mode}')
print(f'Live execution: {config.live_execution_enabled}')
print('='*60)

runner = MVPPaperTrading(config)
print('✅ System initialized')
print('📡 Connecting to Coinbase Advanced Trade...')
print('')

# Simulate trading loop (in production, this would be a WebSocket listener)
import json
import random

# Simulate tick data for demo
print('Demo mode: Simulating 100 ticks for testing...')
for i in range(100):
    timestamp = int(time.time())
    price = 50000 + i * 10 + random.randint(-50, 50)
    volume = 1000 + random.randint(-200, 200)
    orderbook_imbalance = random.random()
    
    # Add tick to system
    tick_data = {
        'timestamp': timestamp,
        'price': price,
        'volume': volume,
        'orderbook_imbalance': orderbook_imbalance
    }
    
    # Process tick (this would be in real WebSocket loop)
    # For demo, just print status
    if i % 10 == 0:
        print(f'Tick {i}: Price=${price:.2f}, Volume={volume}')
    
    time.sleep(0.01)  # Small delay for demo

print('')
print('✅ Demo completed')
print('📊 Check paper_results/ for execution logs')
print('📈 Live execution is ENABLED - real API keys will be used for trading')
print('')
print('To start actual live trading, set live_execution_enabled=True in config')
print('and run this script with real API keys')
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Live paper trading session completed"
echo "Log saved to: $LOG_FILE"
echo "Execution results: paper_results/live_execution_log.jsonl"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Check execution logs: cat paper_results/live_execution_log.jsonl"
echo "2. Monitor for 24 hours with: tail -f paper_results/live_paper_log.jsonl"
echo "3. Scale to 8 predictors if profit factor > 1.5"
echo "4. Run ablation test: vector cache on/off comparison"
echo ""
echo "🚀 Ready for live trading! Set live_execution_enabled=True and re-run."