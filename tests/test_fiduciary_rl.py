import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hrm"))

from fiduciary_rl import FiduciaryRLConfig, fiduciary_overlay, structured_fiduciary_loss  # noqa: E402


def test_fiduciary_overlay_enforces_limits():
    cfg = FiduciaryRLConfig(
        min_confidence=0.6,
        max_position_delta=0.5,
        max_gross_exposure=0.7,
        max_concentration=0.6,
    )

    control = torch.tensor(
        [[[1.0, 0.95, 1.0], [1.0, 0.20, -1.0], [1.0, 0.95, 1.0]]],
        dtype=torch.float32,
    )
    holdings = torch.zeros((1, 3), dtype=torch.float32)

    overlay = fiduciary_overlay(control, holdings, cfg)
    target = overlay.target_positions

    # low-confidence lane should stay at hold
    assert torch.allclose(target[0, 1], torch.tensor(0.0), atol=1e-6)
    # gross exposure capped
    assert target.abs().sum().item() <= cfg.max_gross_exposure + 1e-6


def test_structured_loss_surfaces_concentration_breach():
    cfg = FiduciaryRLConfig(
        min_confidence=0.1,
        max_position_delta=1.0,
        max_gross_exposure=1.0,
        max_concentration=0.2,
        transaction_cost_bps=0.0,
    )

    # Strong single-asset conviction should trigger concentration breach.
    control = torch.tensor([[[1.0, 1.0, 1.0], [0.05, 0.05, 0.0], [0.05, 0.05, 0.0]]], dtype=torch.float32)
    holdings = torch.zeros((1, 3), dtype=torch.float32)
    returns = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float32)

    loss, metrics, _ = structured_fiduciary_loss(
        control=control,
        returns=returns,
        holdings=holdings,
        config=cfg,
    )

    assert torch.isfinite(loss)
    assert metrics["concentration_breach"] > 0.0
