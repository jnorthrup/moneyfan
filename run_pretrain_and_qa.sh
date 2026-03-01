#!/bin/bash
set -e

echo "Starting pretraining regime..."
# Use hidden-dim 96 as recommended by CONTINUOUS_OPTIMIZER_QUICKSTART.md for testing
python train.py --episodes 20 --notional 100 --pretrain-only --fully-stochastic-pair-sampling --hidden-dim 96 --regime-layers 3 --tactical-layers 3 --attention-heads 6 > train_pretrain_stochastic_continue.log 2>&1

echo "Pretraining complete. Running QA checks..."
python quick_variability_check.py
