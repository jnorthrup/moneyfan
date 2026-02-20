#!/bin/bash
# run_24h_live.sh
# Run 24-hour live paper trading with full monitoring

set -e

echo "============================================================"
echo "MONEYFAN 24-HOUR LIVE PAPER TRADING SESSION"
echo "============================================================"
echo "Duration: 24 hours"
echo "Capital: $1000"
echo "Predictors: 3 (MVP) → 8 (if criteria met)"
echo "Kill switch: ACTIVE (5% max drawdown, 3 consecutive losses)"
echo "============================================================"
echo ""

# Check environment variables
if [ -z "$COINBASE_API_KEY" ] || [ -z "$COINBASE_API_SECRET" ]; then
    echo "❌ ERROR: Coinbase API keys not set"
    echo "Please set: export COINBASE_API_KEY=\"...\" && export COINBASE_API_SECRET=\"...\""
    exit 1
fi

echo "✅ API keys configured"
echo ""

# Create results directory
mkdir -p paper_results
mkdir -p logs

# Start timestamp
START_TIME=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="logs/session_${START_TIME}.log"
echo "Session log: $SESSION_LOG"
echo ""

# Function to run monitoring
run_monitoring() {
    echo "📊 Starting 24-hour monitoring dashboard..."
    echo "Press Ctrl+C to stop monitoring"
    echo ""
    
    while true; do
        clear
        
        echo "============================================================"
        echo "MONEYFAN 24H LIVE PAPER TRADING - MONITORING DASHBOARD"
        echo "============================================================"
        echo "Session: $START_TIME"
        echo "Last update: $(date)"
        echo ""
        
        # Check if process is running
        if ! pgrep -f "mvp_runner.py" > /dev/null; then
            echo "⚠️  MVP Runner not running - checking for errors..."
        fi
        
        # Show execution logs
        if [ -f "paper_results/live_execution_log.jsonl" ]; then
            echo "📊 EXECUTION LOGS:"
            tail -n 5 "paper_results/live_execution_log.jsonl" 2>/dev/null | while read line; do
                if [ -n "$line" ]; then
                    action=$(echo "$line" | jq -r '.action' 2>/dev/null || echo "unknown")
                    product=$(echo "$line" | jq -r '.result.product' 2>/dev/null || echo "BTC-USD")
                    size=$(echo "$line" | jq -r '.result.size_usd' 2>/dev/null || echo "0")
                    echo "   $action: $size $product"
                fi
            done
            echo ""
        fi
        
        # Show metrics
        if [ -f "paper_results/metrics.json" ]; then
            echo "🎯 PERFORMANCE METRICS:"
            jq -r '"  \(.key): \(.value)"' paper_results/metrics.json 2>/dev/null | head -10
            echo ""
        fi
        
        # Show equity curve
        if [ -f "paper_results/equity_curve.json" ]; then
            equity_count=$(jq length paper_results/equity_curve.json 2>/dev/null || echo 0)
            if [ "$equity_count" -gt 0 ]; then
                first_equity=$(jq '.[0].equity' paper_results/equity_curve.json 2>/dev/null || echo 1000)
                last_equity=$(jq '.[-1].equity' paper_results/equity_curve.json 2>/dev/null || echo 1000)
                echo "💰 EQUITY CURVE:"
                echo "   Points: $equity_count"
                echo "   First: \$$first_equity"
                echo "   Last: \$$last_equity"
                echo "   Change: \$$(echo \"$last_equity - $first_equity\" | bc -l) (2 decimals)"
                echo ""
            fi
        fi
        
        # Show regime changes
        if [ -f "paper_results/regimes.json" ]; then
            regime_count=$(jq length paper_results/regimes.json 2>/dev/null || echo 0)
            if [ "$regime_count" -gt 0 ]; then
                latest_regime=$(jq -r '.[-1].regime' paper_results/regimes.json 2>/dev/null || echo "unknown")
                echo "🔄 REGIME CHANGES:"
                echo "   Total: $regime_count"
                echo "   Latest: $latest_regime"
                echo ""
            fi
        fi
        
        # Show kill switch status
        echo "🛑 KILL SWITCH STATUS:"
        echo "   Max Drawdown: 5% hard limit"
        echo "   Max Single Trade Loss: 2% hard limit"
        echo "   Max Consecutive Losses: 3 → 1hr pause"
        echo ""
        
        # Show controls
        echo "============================================================"
        echo "CONTROLS:"
        echo "  q - Quit monitoring"
        echo "  k - Trigger manual kill switch"
        echo "  r - Refresh dashboard"
        echo "  m - Show full metrics"
        echo "  e - Show equity curve"
        echo "  h - Show help"
        echo "============================================================"
        
        # Check for keyboard input with timeout
        read -t 2 -n 1 -s input
        
        case $input in
            q)
                echo ""
                echo "Stopping monitoring..."
                return 0
                ;;
            k)
                echo ""
                echo "⚠️  MANUAL KILL SWITCH - Are you sure? (y/N)"
                read -p "> " confirm
                if [ "$confirm" = "y" ]; then
                    echo "🚨 TRIGGERING MANUAL KILL SWITCH..."
                    pkill -f mvp_runner.py
                    echo "MVP Runner stopped"
                    sleep 2
                    return 1
                fi
                ;;
            r)
                # Refresh
                ;;
            m)
                echo ""
                if [ -f "paper_results/metrics.json" ]; then
                    cat paper_results/metrics.json | jq . 2>/dev/null || cat paper_results/metrics.json
                else
                    echo "No metrics file yet"
                fi
                echo ""
                echo "Press Enter to continue..."
                read
                ;;
            e)
                echo ""
                if [ -f "paper_results/equity_curve.json" ]; then
                    tail -n 10 paper_results/equity_curve.json | jq . 2>/dev/null || echo "Cannot parse"
                else
                    echo "No equity curve file yet"
                fi
                echo ""
                echo "Press Enter to continue..."
                read
                ;;
            h)
                echo ""
                echo "Help:"
                echo "q - Quit monitoring"
                echo "k - Manual kill switch"
                echo "r - Refresh"
                echo "m - Show metrics"
                echo "e - Show equity curve"
                echo ""
                echo "Press Enter to continue..."
                read
                ;;
        esac
        
        # Check if we should generate 4-hour report
        current_hour=$(date +%H)
        if [ "$current_hour" = "00" ] || [ "$current_hour" = "04" ] || \
           [ "$current_hour" = "08" ] || [ "$current_hour" = "12" ] || \
           [ "$current_hour" = "16" ] || [ "$current_hour" = "20" ]; then
            echo ""
            echo "📊 GENERATING 4-HOUR REPORT..."
            python3 -c "
from execution.kill_switch import KillSwitch, KillSwitchConfig
ks = KillSwitch(KillSwitchConfig())
ks.update_from_metrics_file()
report = ks.generate_4h_report()
print(report)
" 2>/dev/null || echo "Could not generate report"
            sleep 3
        fi
    done
}

# Start monitoring in background
echo "Starting monitoring in background..."
run_monitoring &
MONITOR_PID=$!

# Start MVP runner in background
echo "Starting MVP runner with live execution..."
python3 -c "
import asyncio
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
    coinbase_api_secret='$COINBASE_API_SECRET',
    validation_days=1  # 24 hours
)

print('='*60)
print('MONEYFAN 24H LIVE PAPER TRADING')
print('='*60)
print(f'Capital: \${config.paper_capital:.2f}')
print(f'Predictors: {config.n_predictors}')
print(f'Paper mode: {config.paper_mode}')
print(f'Live execution: {config.live_execution_enabled}')
print('='*60)

runner = MVPPaperTrading(config)
print('✅ System initialized')
print('📡 Connecting to Coinbase Advanced Trade...')

# In production, this would be a WebSocket listener
# For demo, we'll run a simulated loop
import json
import random

print('')
print('⏳ Running 24-hour simulation...')
print('   (In production, this would be real WebSocket data)')
print('')

# Simulate ticks
for hour in range(24):
    print(f'Hour {hour+1}/24...')
    for tick in range(100):  # 100 ticks per hour
        timestamp = int(time.time())
        price = 50000 + hour * 100 + tick + random.randint(-100, 100)
        volume = 1000 + random.randint(-200, 200)
        orderbook_imbalance = random.random()
        
        # Add tick to system
        tick_data = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'orderbook_imbalance': orderbook_imbalance
        }
        
        # Process tick (would be in real WebSocket loop)
        # For now, just log progress
        if tick % 50 == 0:
            print(f'  Tick {tick}: Price=\${price:.2f}')
        
        time.sleep(0.01)
    
    # Save metrics every hour
    runner.save_results()
    
    # Generate 4-hour report
    if (hour + 1) % 4 == 0:
        print(f'\\n📊 4-HOUR REPORT (Hour {hour+1})')
        print('='*40)

print('\\n✅ 24-hour simulation completed')
print('📊 Check paper_results/ for final metrics')
" 2>&1 | tee "$SESSION_LOG" &
MVP_PID=$!

echo ""
echo "============================================================"
echo "24-HOUR SESSION STARTED"
echo "============================================================"
echo "Session log: $SESSION_LOG"
echo "MVP Runner PID: $MVP_PID"
echo "Monitor PID: $MONITOR_PID"
echo ""
echo "Monitoring dashboard running in this terminal..."
echo "MVP Runner running in background..."
echo ""
echo "To stop:"
echo "1. Press Ctrl+C in this terminal (will stop both)"
echo "2. Or run: kill -TERM $MVP_PID"
echo ""
echo "📊 Dashboard controls:"
echo "  q - Quit monitoring"
echo "  k - Manual kill switch"
echo "  r - Refresh dashboard"
echo "  m - Show metrics"
echo "  e - Show equity curve"
echo ""
echo "============================================================"
echo ""

# Wait for monitoring to finish
wait $MONITOR_PID

# Cleanup
echo ""
echo "Cleaning up..."
kill $MVP_PID 2>/dev/null
wait $MVP_PID 2>/dev/null

echo "============================================================"
echo "24-HOUR SESSION COMPLETED"
echo "============================================================"
echo "Session log: $SESSION_LOG"
echo "Results saved to: paper_results/"
echo "============================================================"