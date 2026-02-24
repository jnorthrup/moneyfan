#!/bin/bash
python train.py \
  --pretrain-only \
  --timer-based \
  --max-training-seconds 10800 \
  --episodes 100000 \
  --pair-width 30 \
  --bar-sequences-per-episode 100 \
  --min-bar-window 64 \
  --max-bar-window 256 \
  --learning-rate 1e-4
