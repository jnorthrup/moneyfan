#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$ROOT_DIR/train_pretrain_stochastic_continue.log"
GATE_FILE="$ROOT_DIR/synthetic_gate_reference.json"

echo "Starting pretraining regime..."
# Use hidden-dim 96 as recommended by CONTINUOUS_OPTIMIZER_QUICKSTART.md for testing
cd "$ROOT_DIR"
python3 museum/train.py --episodes 20 --notional 100 --pretrain-only --fully-stochastic-pair-sampling --hidden-dim 96 --regime-layers 3 --tactical-layers 3 --attention-heads 6 > "$LOG_FILE" 2>&1

echo "Pretraining complete. Running QA checks..."
python3 museum/quick_variability_check.py --log-file "$LOG_FILE"

echo "Building synthetic gate reference baselines..."
python3 museum/synthetic_gate_evaluator.py --json > "$GATE_FILE"
echo "Synthetic gate reference written to $GATE_FILE"
