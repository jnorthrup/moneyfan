import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hrm"))

from convergence import (  # noqa: E402
    ModelSignalConvergenceTracker,
    convergence_from_snapshot,
    rolling_convergence,
)


def test_convergence_from_snapshot_rewards_directional_agreement():
    signals = np.array([0.8, 0.7, 0.6, -0.1], dtype=np.float32)
    confidences = np.array([0.9, 0.8, 0.7, 0.2], dtype=np.float32)
    score = convergence_from_snapshot(signals, confidences, min_agree=2, min_confidence_sum=0.25)
    assert score > 0.6


def test_rolling_convergence_detects_regime_shift():
    signals_hist = np.array(
        [
            [0.8, 0.7, 0.6],
            [0.7, 0.6, 0.5],
            [0.9, 0.7, 0.4],
            [-0.8, 0.7, 0.3],
            [-0.9, 0.8, 0.4],
        ],
        dtype=np.float32,
    )
    conf_hist = np.full_like(signals_hist, 0.8, dtype=np.float32)

    scores = rolling_convergence(
        signals_hist,
        conf_hist,
        window=3,
        min_agree=2,
        min_confidence_sum=0.25,
    )

    # Pre-shift agreement should be high; after shift should drop.
    assert scores[2] > scores[4]


def test_tracker_emits_per_instrument_convergence_and_agreement():
    tracker = ModelSignalConvergenceTracker(
        n_instruments=2,
        n_models=3,
        window=4,
        min_agree=2,
        min_confidence_sum=0.2,
    )

    # Shape [N, M, 5] where [..,0]=signal and [..,1]=confidence
    snapshot = np.zeros((2, 3, 5), dtype=np.float32)
    snapshot[0, :, 0] = [0.7, 0.6, 0.5]
    snapshot[0, :, 1] = [0.9, 0.8, 0.7]
    snapshot[1, :, 0] = [0.8, -0.8, 0.1]
    snapshot[1, :, 1] = [0.9, 0.9, 0.2]

    out = tracker.update(snapshot)

    assert out["convergence"].shape == (2,)
    assert out["agreement_ratio"].shape == (2,)
    assert out["confidence_support"].shape == (2,)
    assert out["convergence"][0] > out["convergence"][1]

