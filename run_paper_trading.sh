#!/bin/bash
# Run 30-day paper trading for HRM system

echo "================================================"
echo "Running 30-day Paper Trading for HRM System"
echo "================================================"

# Check if data exists
if [ ! -d "hrm/data/arrow" ]; then
    echo "❌ Error: hrm/data/arrow directory not found"
    echo "   Please download Binance data first"
    echo "   Run: python hrm/download_binance_data.py --pairs 30 --timeframe 5m"
    exit 1
fi

# Count feather files
FEATHER_COUNT=$(ls hrm/data/arrow/*.feather 2>/dev/null | wc -l)
if [ $FEATHER_COUNT -eq 0 ]; then
    echo "❌ Error: No feather files found in hrm/data/arrow"
    echo "   Please download Binance data first"
    exit 1
fi

echo "✅ Found $FEATHER_COUNT feather files in hrm/data/arrow"

# Run paper trading
python paper_trading.py \
    --days 30 \
    --capital 100 \
    --codecs 24 \
    --bag-size 30 \
    --output paper_trading_results \
    --save-equity-curve

# Check exit code
if [ $? -eq 0 ]; then
    echo "================================================"
    echo "✅ Paper trading completed successfully!"
    echo "   Results saved to: paper_trading_results/"
    echo "   Equity curve: paper_trading_results/equity_curve.png"
    echo "   Trades: paper_trading_results/trades.csv"
    echo "   Metrics: paper_trading_results/metrics.json"
    echo "================================================"
else
    echo "================================================"
    echo "❌ Paper trading failed"
    echo "================================================"
    exit 1
fi