#!/bin/bash
# monitor_paper_trading.sh
# Monitor live paper trading session for 24 hours
# Run this in a separate terminal after launching live trading

set -e

echo "============================================================"
echo "MONITORING LIVE PAPER TRADING"
echo "============================================================"
echo "Monitoring directory: paper_results/"
echo "Log file: live_paper_log.jsonl"
echo "Execution log: live_execution_log.jsonl"
echo "============================================================"
echo ""

# Check if monitoring directory exists
if [ ! -d "paper_results" ]; then
    echo "❌ ERROR: paper_results directory not found"
    echo "Please launch live trading first"
    exit 1
fi

# Check if log files exist
LOG_FILE="paper_results/live_paper_log.jsonl"
EXEC_LOG="paper_results/live_execution_log.jsonl"

if [ ! -f "$LOG_FILE" ] && [ ! -f "$EXEC_LOG" ]; then
    echo "⚠️  WARNING: No log files found yet"
    echo "Live trading may not have started"
    echo ""
    echo "Starting monitoring in 5 seconds..."
    sleep 5
fi

# Setup monitoring
echo "📊 Starting 24-hour monitoring..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Create monitoring dashboard file
DASHBOARD_FILE="paper_results/monitoring_dashboard.txt"
echo "Live Paper Trading Dashboard - $(date)" > "$DASHBOARD_FILE"
echo "============================================================" >> "$DASHBOARD_FILE"
echo "" >> "$DASHBOARD_FILE"

# Function to update dashboard
update_dashboard() {
    clear
    echo "============================================================"
    echo "MONEYFAN LIVE PAPER TRADING - MONITORING DASHBOARD"
    echo "============================================================"
    echo "Last update: $(date)"
    echo ""
    
    # Check execution logs
    if [ -f "$EXEC_LOG" ]; then
        echo "📊 EXECUTION LOGS:"
        echo "   Total executions: $(wc -l < "$EXEC_LOG" 2>/dev/null || echo 0)"
        echo "   Last execution:"
        tail -n 1 "$EXEC_LOG" 2>/dev/null | jq -r '.timestamp' 2>/dev/null || echo "   No data yet"
        echo ""
    fi
    
    # Check paper trading logs
    if [ -f "$LOG_FILE" ]; then
        echo "📈 PAPER TRADING LOGS:"
        echo "   Total entries: $(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)"
        echo ""
    fi
    
    # Check metrics file
    METRICS_FILE="paper_results/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "🎯 PERFORMANCE METRICS:"
        cat "$METRICS_FILE" | jq -r 'to_entries[] | "   \(.key): \(.value)"' 2>/dev/null || echo "   Metrics file exists but cannot parse"
        echo ""
    fi
    
    # Check equity curve
    EQUITY_FILE="paper_results/equity_curve.json"
    if [ -f "$EQUITY_FILE" ]; then
        echo "💰 EQUITY CURVE:"
        echo "   Points: $(jq length "$EQUITY_FILE" 2>/dev/null || echo 0)"
        echo "   First equity: $(jq '.[0].equity' "$EQUITY_FILE" 2>/dev/null || echo "N/A")"
        echo "   Last equity: $(jq '.[-1].equity' "$EQUITY_FILE" 2>/dev/null || echo "N/A")"
        echo ""
    fi
    
    # Check regime changes
    REGIME_FILE="paper_results/regimes.json"
    if [ -f "$REGIME_FILE" ]; then
        echo "🔄 REGIME CHANGES:"
        echo "   Total: $(jq length "$REGIME_FILE" 2>/dev/null || echo 0)"
        echo "   Latest:"
        tail -n 1 "$REGIME_FILE" 2>/dev/null | jq -r '.regime' 2>/dev/null || echo "   No data yet"
        echo ""
    fi
    
    echo "============================================================"
    echo "MONITORING CONTROLS:"
    echo "1. Press 'q' to quit"
    echo "2. Press 'l' to show full logs"
    echo "3. Press 'r' to reset dashboard"
    echo "============================================================"
}

# Function to show full logs
show_logs() {
    echo "============================================================"
    echo "FULL LOG OUTPUT"
    echo "============================================================"
    echo ""
    
    if [ -f "$EXEC_LOG" ]; then
        echo "📊 EXECUTION LOGS:"
        tail -n 20 "$EXEC_LOG" 2>/dev/null | jq -r 'select(.action != null) | "\(.action): \(.result.product)//\(.result.size_usd//0)"' 2>/dev/null || echo "No execution data"
        echo ""
    fi
    
    if [ -f "$LOG_FILE" ]; then
        echo "📈 TRADING LOGS:"
        tail -n 20 "$LOG_FILE" 2>/dev/null | jq -r '.signal.direction' 2>/dev/null | sort | uniq -c | while read count dir; do
            echo "   $dir: $count signals"
        done
        echo ""
    fi
    
    echo "Press Enter to return to dashboard..."
    read
}

# Main monitoring loop
while true; do
    update_dashboard
    
    # Check for keyboard input with timeout
    read -t 2 -n 1 -s input
    
    if [ "$input" = "q" ]; then
        echo ""
        echo "Stopping monitoring..."
        break
    elif [ "$input" = "l" ]; then
        show_logs
    elif [ "$input" = "r" ]; then
        echo "Dashboard reset..."
        sleep 1
    fi
done

echo "============================================================"
echo "Monitoring stopped at $(date)"
echo "Summary saved to: $DASHBOARD_FILE"
echo "============================================================"