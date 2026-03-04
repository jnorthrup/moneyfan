from pathlib import Path
import sys
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    EpisodeTrainingConfig,221
    should_run_trade_step,
)

def test_should_run_trade_step_probabilistic():
    config = EpisodeTrainingConfig(
        trade_step_schedule_mode="probabilistic",
        trade_update_prob=1.0,  # Force 100%
        trade_update_min_abs_return=0.0
    )
    # Should always run if prob is 1.0 and return is >= 0
    assert should_run_trade_step(0, 0.01, config) is True
    
    config.trade_update_prob = 0.0
    assert should_run_trade_step(0, 0.01, config) is False

def test_should_run_trade_step_deterministic():
    config = EpisodeTrainingConfig(
        trade_step_schedule_mode="deterministic",
        trade_step_schedule_interval=2,
        trade_update_min_abs_return=0.0
    )
    # Should run on 0, 2, 4...
    assert should_run_trade_step(0, 0.01, config) is True
    assert should_run_trade_step(1, 0.01, config) is False
    assert should_run_trade_step(2, 0.01, config) is True

def test_should_run_trade_step_density_gated():
    config = EpisodeTrainingConfig(
        trade_step_schedule_mode="density_gated",
        trade_step_min_density=0.5,
        trade_update_prob=1.0,
        trade_update_min_abs_return=0.0
    )
    # Density 0.6 >= 0.5 -> pass
    assert should_run_trade_step(0, 0.01, config, sample_density=0.6) is True
    # Density 0.4 < 0.5 -> fail
    assert should_run_trade_step(0, 0.01, config, sample_density=0.4) is False

def test_should_run_trade_step_return_gating():
    config = EpisodeTrainingConfig(
        trade_step_schedule_mode="probabilistic",
        trade_update_prob=1.0,
        trade_update_min_abs_return=0.05
    )
    # Return 0.06 >= 0.05 -> pass
    assert should_run_trade_step(0, 0.06, config) is True
    # Return 0.04 < 0.05 -> fail
    assert should_run_trade_step(0, 0.04, config) is False
    # Negative return -0.06 (abs is 0.06) >= 0.05 -> pass
    assert should_run_trade_step(0, -0.06, config) is True
