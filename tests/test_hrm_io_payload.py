import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hrm"))

from hrm_io import HRMIO, HRMOutput  # noqa: E402


class _DecisionStub:
    def __init__(self, model_id: str):
        self.model_id = model_id


def test_hrm_output_to_dict_preserves_legacy_keys_and_adds_convergence():
    out = HRMOutput(
        decision=_DecisionStub("momentum_trend"),
        active_bot="momentum_trend",
        signal=0.42,
        confidence=0.77,
        regime="trending",
        timestamp="2026-02-17T00:00:00Z",
        convergence=0.61,
        agreement_ratio=0.75,
        confidence_support=2.1,
    )

    payload = out.to_dict()

    # Legacy keys
    assert payload["decision"] == "momentum_trend"
    assert payload["active_bot"] == "momentum_trend"
    assert payload["signal"] == 0.42
    assert payload["confidence"] == 0.77
    assert payload["regime"] == "trending"
    assert payload["timestamp"] == "2026-02-17T00:00:00Z"

    # New convergence schema keys
    assert payload["convergence"] == 0.61
    assert payload["agreement_ratio"] == 0.75
    assert payload["confidence_support"] == 2.1


def test_compute_convergence_payload_handles_missing_columns():
    io = object.__new__(HRMIO)
    payload = io._compute_convergence_payload(pd.DataFrame())
    assert payload["convergence"] == 0.0
    assert payload["agreement_ratio"] == 0.0
    assert payload["confidence_support"] == 0.0


def test_compute_convergence_payload_from_bot_states():
    io = object.__new__(HRMIO)
    states = pd.DataFrame(
        {
            "last_signal": [0.8, 0.7, -0.1, 0.6],
            "confidence": [0.9, 0.8, 0.2, 0.7],
        }
    )

    payload = io._compute_convergence_payload(states)

    assert 0.0 <= payload["convergence"] <= 1.0
    assert payload["agreement_ratio"] > 0.5
    assert payload["confidence_support"] > 0.0

