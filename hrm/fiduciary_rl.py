"""
Structured fiduciary signal processing for HRM meta-allocation.

This module keeps candle/model signals as the hot path while adding a
fiduciary governance overlay:
  - bound actions by confidence and exposure policy
  - shape reward with execution costs and risk penalties
  - emit oversight diagnostics from the same tensors used for execution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class FiduciaryRLConfig:
    """Policy and reward configuration for fiduciary-constrained RL."""

    min_confidence: float = 0.25
    max_position_delta: float = 0.20
    max_gross_exposure: float = 1.00
    max_turnover: float = 0.35
    max_concentration: float = 0.30

    transaction_cost_bps: float = 6.0

    pnl_reward_weight: float = 1.00
    downside_penalty_weight: float = 0.60
    turnover_penalty_weight: float = 0.25
    concentration_penalty_weight: float = 0.20
    confidence_penalty_weight: float = 0.10
    stability_penalty_weight: float = 0.05
    fiduciary_penalty_weight: float = 0.40

    alert_threshold: float = 0.30


@dataclass
class FiduciaryOverlay:
    """Constrained execution intent plus oversight diagnostics."""

    target_positions: torch.Tensor
    action_delta: torch.Tensor
    oversight_score: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _scalar(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().mean().item())


def fiduciary_overlay(
    control: torch.Tensor,
    holdings: torch.Tensor,
    config: FiduciaryRLConfig,
) -> FiduciaryOverlay:
    """
    Convert controller outputs into policy-compliant target positions.

    Args:
        control: [B, N, 3] -> (velocity, confidence, direction)
        holdings: [B, N]
    """

    velocity = control[..., 0].clamp(0.0, 1.0)
    confidence = control[..., 1].clamp(0.0, 1.0)
    direction = control[..., 2].clamp(-1.0, 1.0)

    confidence_mask = (confidence >= config.min_confidence).to(control.dtype)

    desired = direction * velocity * confidence
    desired = holdings + (desired - holdings) * confidence_mask

    delta = (desired - holdings).clamp(-config.max_position_delta, config.max_position_delta)
    target = holdings + delta

    gross = target.abs().sum(dim=-1, keepdim=True)
    max_gross = _scalar(config.max_gross_exposure, gross)
    leverage_scale = torch.where(
        gross > max_gross,
        max_gross / (gross + EPS),
        torch.ones_like(gross),
    )
    target = target * leverage_scale

    action_delta = target - holdings
    turnover = action_delta.abs().mean(dim=-1)
    concentration = target.abs().amax(dim=-1) / (target.abs().sum(dim=-1) + EPS)
    confidence_gap = F.relu(_scalar(config.min_confidence, confidence) - confidence).mean(dim=-1)

    leverage_breach = F.relu(target.abs().sum(dim=-1) - config.max_gross_exposure)
    turnover_breach = F.relu(turnover - config.max_turnover)
    concentration_breach = F.relu(concentration - config.max_concentration)

    oversight_raw = leverage_breach + turnover_breach + concentration_breach + confidence_gap
    oversight_score = torch.tanh(4.0 * oversight_raw).clamp(0.0, 1.0)

    diagnostics = {
        "turnover": turnover,
        "concentration": concentration,
        "confidence_gap": confidence_gap,
        "leverage_breach": leverage_breach,
        "turnover_breach": turnover_breach,
        "concentration_breach": concentration_breach,
    }

    return FiduciaryOverlay(
        target_positions=target,
        action_delta=action_delta,
        oversight_score=oversight_score,
        diagnostics=diagnostics,
    )


def structured_fiduciary_loss(
    control: torch.Tensor,
    returns: torch.Tensor,
    holdings: torch.Tensor,
    config: FiduciaryRLConfig,
    prev_target: Optional[torch.Tensor] = None,
    prev_control: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    Compute fiduciary-aware structured RL objective.

    Returns:
        loss, metric dict, detached target positions for stability continuity.
    """

    overlay = fiduciary_overlay(control=control, holdings=holdings, config=config)
    target = overlay.target_positions
    action_delta = overlay.action_delta

    turnover = overlay.diagnostics["turnover"]
    concentration = overlay.diagnostics["concentration"]
    confidence_gap = overlay.diagnostics["confidence_gap"]
    leverage_breach = overlay.diagnostics["leverage_breach"]
    turnover_breach = overlay.diagnostics["turnover_breach"]
    concentration_breach = overlay.diagnostics["concentration_breach"]

    gross_pnl = (target * returns).sum(dim=-1)
    tx_cost = turnover * (config.transaction_cost_bps / 10_000.0)
    net_pnl = gross_pnl - tx_cost
    downside = F.relu(-net_pnl)

    if prev_target is not None and prev_target.shape == target.shape:
        stability = (target - prev_target).abs().mean(dim=-1)
    elif prev_control is not None and prev_control.shape == control.shape:
        prev_velocity = prev_control[..., 0].clamp(0.0, 1.0)
        prev_confidence = prev_control[..., 1].clamp(0.0, 1.0)
        prev_direction = prev_control[..., 2].clamp(-1.0, 1.0)
        prev_proxy = prev_direction * prev_velocity * prev_confidence
        stability = (target - prev_proxy).abs().mean(dim=-1)
    else:
        stability = torch.zeros_like(net_pnl)

    fiduciary_penalty = leverage_breach + turnover_breach + concentration_breach
    reward = (
        config.pnl_reward_weight * net_pnl
        - config.downside_penalty_weight * downside
        - config.turnover_penalty_weight * turnover_breach
        - config.concentration_penalty_weight * concentration_breach
        - config.confidence_penalty_weight * confidence_gap
        - config.stability_penalty_weight * stability
        - config.fiduciary_penalty_weight * fiduciary_penalty
    )
    loss = -reward.mean()

    alert_ratio = (overlay.oversight_score > config.alert_threshold).to(control.dtype)

    metrics = {
        "alpha_return": _as_float(net_pnl),
        "gross_pnl": _as_float(gross_pnl),
        "tx_cost": _as_float(tx_cost),
        "downside": _as_float(downside),
        "turnover": _as_float(turnover),
        "concentration": _as_float(concentration),
        "consistency": _as_float(stability),
        "fiduciary_penalty": _as_float(fiduciary_penalty),
        "leverage_breach": _as_float(leverage_breach),
        "turnover_breach": _as_float(turnover_breach),
        "concentration_breach": _as_float(concentration_breach),
        "confidence_gap": _as_float(confidence_gap),
        "oversight_score": _as_float(overlay.oversight_score),
        "oversight_alert_ratio": _as_float(alert_ratio),
    }
    return loss, metrics, target.detach()


def build_oversight_flags(
    oversight_score: torch.Tensor,
    diagnostics: Dict[str, torch.Tensor],
    alert_threshold: float = 0.30,
) -> Dict[str, torch.Tensor]:
    """Build executive oversight flags from constrained execution diagnostics."""

    requires_review = oversight_score > alert_threshold
    return {
        "requires_review": requires_review,
        "leverage_breach": diagnostics["leverage_breach"] > 0,
        "turnover_breach": diagnostics["turnover_breach"] > 0,
        "concentration_breach": diagnostics["concentration_breach"] > 0,
        "confidence_breach": diagnostics["confidence_gap"] > 0,
    }
