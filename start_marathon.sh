#!/bin/bash
# Start Bag Trading Marathon
# Usage: ./start_marathon.sh [hours] [capital] [update_interval]

HOURS=${1:-24}
CAPITAL=${2:-1000}
INTERVAL=${3:-5}

echo "Starting Bag Trading Marathon..."
echo "Hours: $HOURS, Capital: \$$CAPITAL, Update Interval: ${INTERVAL}s"
echo ""

# Run the marathon
python bag_trading_marathon.py --hours $HOURS --capital $CAPITAL --update-interval $INTERVAL

echo ""
echo "Marathon completed. Check the results in marathon_results/"