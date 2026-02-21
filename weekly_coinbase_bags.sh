#!/bin/bash
# Weekly Coinbase Bags - GOALS.md Compliant Training
# Usage: ./weekly_coinbase_bags.sh [weeks] [capital] [bags_per_week]

WEEKS=${1:-4}
CAPITAL=${2:-1000}
BAGS_PER_WEEK=${3:-100}

echo "Starting weekly Coinbase bags training..."
echo "Weeks: $WEEKS"
echo "Capital: \$$CAPITAL"
echo "Bags per week: $BAGS_PER_WEEK"
echo ""

# Run the training
python3 weekly_coinbase_bags.py --weeks $WEEKS --capital $CAPITAL --bags-per-week $BAGS_PER_WEEK

echo ""
echo "Training completed. Check results in weekly_coinbase_bags/"