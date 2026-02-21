#!/bin/bash
# Train 500 Stochastic Bags
# Usage: ./train_500_bags.sh [bags] [capital] [update_interval]

BAGS=${1:-500}
CAPITAL=${2:-100}
INTERVAL=${3:-5}

echo "Starting training of $BAGS stochastic bags..."
echo "Capital per bag: \$$CAPITAL"
echo "Update interval: ${INTERVAL}s"
echo ""

# Run the training
python3 train_500_bags.py --bags $BAGS --capital $CAPITAL --update-interval $INTERVAL

echo ""
echo "Training completed. Check results in bag_training_results/"