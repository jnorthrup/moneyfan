#!/bin/bash
# Train 500 Stochastic Bags with DuckDB Backend
# Usage: ./train_500_bags_duckdb.sh [upload_binance] [bags] [capital]

UPLOAD=${1:-false}
BAGS=${2:-500}
CAPITAL=${3:-100}

echo "Starting training of $BAGS stochastic bags with DuckDB..."
echo "Upload Binance data: $UPLOAD"
echo "Capital per bag: \$$CAPITAL"
echo ""

# Run the training
if [ "$UPLOAD" = "true" ]; then
    python3 train_500_bags_duckdb.py --upload-binance --bags $BAGS --capital $CAPITAL
else
    python3 train_500_bags_duckdb.py --bags $BAGS --capital $CAPITAL
fi

echo ""
echo "Training completed. Check results in bag_training_results_duckdb/"