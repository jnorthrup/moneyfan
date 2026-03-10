"""
Standardized paper-loop drawdown telemetry event shape.

This module owns the canonical telemetry event schema for paper trading
drawdown monitoring. Events are emitted to a JSONL sink consumable by
freqtrade report generation and insight aggregation.

Schema: moneyfan.paper.drawdown.telemetry.v1

Required fields per event:
  - schema: literal "moneyfan.paper.drawdown.telemetry.v1"
  - ts_utc: ISO-8601 timestamp
  - signal_id: str  (hrm signal that triggered or was active at this point)
  - iteration: int
  - drawdown_pct: float  (absolute value, 0.0 = no drawdown)
  - threshold_state: str  ("normal" | "warn" | "derisk" | "halt")
  - threshold_warn: float
  - threshold_derisk: float
  - threshold_halt: float
  - equity: float
  - peak_equity: float
  - mode: str  (paper | live-preview)

Reconciliation metadata (for freqtrade report generation):
  - profile_id: str | None  (stress profile id if running pretesting)
  - guardrail_action_active: bool
  - position_size_scale: float  (1.0 = normal, < 1.0 = scaled down by guardrail)
  - new_entries_allowed: bool
  - effective_top_k: int | None
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


#: Canonical schema identifier — must not change between versions
TELEMETRY_SCHEMA = "moneyfan.paper.drawdown.telemetry.v1"

#: All keys that must be present in every emitted telemetry event
REQUIRED_TELEMETRY_KEYS = (
    "schema",
    "ts_utc",
    "signal_id",
    "iteration",
    "drawdown_pct",
    "threshold_state",
    "threshold_warn",
    "threshold_derisk",
    "threshold_halt",
    "equity",
    "peak_equity",
    "mode",
    # reconciliation
    "guardrail_action_active",
    "position_size_scale",
    "new_entries_allowed",
)

#: Valid threshold states (aligned with guardrail system)
VALID_THRESHOLD_STATES = frozenset(("normal", "warn", "derisk", "halt"))


def build_paper_drawdown_telemetry_event(
    *,
    signal_id: str,
    iteration: int,
    drawdown_pct: float,
    threshold_state: str,
    threshold_warn: float,
    threshold_derisk: float,
    threshold_halt: float,
    equity: float,
    peak_equity: float,
    mode: str,
    guardrail_action_active: bool,
    position_size_scale: float = 1.0,
    new_entries_allowed: bool = True,
    effective_top_k: Optional[int] = None,
    profile_id: Optional[str] = None,
    ts_utc: Optional[str] = None,
) -> dict[str, Any]:
    """Build a canonical paper drawdown telemetry event dict.

    All required keys are always present. Optional reconciliation fields
    are included when provided.

    Args:
        signal_id: The HRM signal ID active or most recently emitted.
        iteration: Current engine iteration counter.
        drawdown_pct: Current portfolio drawdown (absolute value, >= 0.0).
        threshold_state: Current guardrail state string.
        threshold_warn: Warn threshold (fractional, e.g. 0.05).
        threshold_derisk: Derisk threshold.
        threshold_halt: Halt threshold.
        equity: Current portfolio equity.
        peak_equity: Session peak equity.
        mode: Engine mode ("paper" or "live-preview").
        guardrail_action_active: Whether a non-normal guardrail action is live.
        position_size_scale: Active position size multiplier (1.0 = normal).
        new_entries_allowed: Whether the guardrail permits new position entries.
        effective_top_k: Top-k limit currently in effect (None if unmodified).
        profile_id: Active drawdown stress profile id, if any.
        ts_utc: ISO-8601 timestamp override (defaults to now).

    Returns:
        dict conforming to TELEMETRY_SCHEMA.
    """
    if threshold_state not in VALID_THRESHOLD_STATES:
        raise ValueError(
            f"threshold_state must be one of {sorted(VALID_THRESHOLD_STATES)!r},"
            f" got {threshold_state!r}"
        )
    if drawdown_pct < 0.0:
        raise ValueError(
            f"drawdown_pct must be >= 0.0 (absolute value), got {drawdown_pct}"
        )

    event: dict[str, Any] = {
        "schema": TELEMETRY_SCHEMA,
        "ts_utc": ts_utc or datetime.now(timezone.utc).isoformat(),
        "signal_id": str(signal_id),
        "iteration": int(iteration),
        "drawdown_pct": float(drawdown_pct),
        "threshold_state": str(threshold_state),
        "threshold_warn": float(threshold_warn),
        "threshold_derisk": float(threshold_derisk),
        "threshold_halt": float(threshold_halt),
        "equity": float(equity),
        "peak_equity": float(peak_equity),
        "mode": str(mode),
        # reconciliation fields
        "guardrail_action_active": bool(guardrail_action_active),
        "position_size_scale": float(position_size_scale),
        "new_entries_allowed": bool(new_entries_allowed),
    }

    # Optional fields — only present when supplied
    if effective_top_k is not None:
        event["effective_top_k"] = int(effective_top_k)
    if profile_id is not None:
        event["profile_id"] = str(profile_id)

    return event


def validate_telemetry_event(event: dict[str, Any]) -> list[str]:
    """Validate a telemetry event dict against the schema contract.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    for key in REQUIRED_TELEMETRY_KEYS:
        if key not in event:
            errors.append(f"Missing required key: {key!r}")

    if event.get("schema") != TELEMETRY_SCHEMA:
        errors.append(
            f"schema mismatch: expected {TELEMETRY_SCHEMA!r},"
            f" got {event.get('schema')!r}"
        )

    ts = event.get("ts_utc")
    if ts is not None and not isinstance(ts, str):
        errors.append(f"ts_utc must be str, got {type(ts).__name__}")

    state = event.get("threshold_state")
    if state not in VALID_THRESHOLD_STATES:
        errors.append(
            f"threshold_state must be one of {sorted(VALID_THRESHOLD_STATES)!r},"
            f" got {state!r}"
        )

    dd = event.get("drawdown_pct")
    if dd is not None and float(dd) < 0.0:
        errors.append(f"drawdown_pct must be >= 0.0, got {dd}")

    return errors


def is_threshold_crossing(
    prev_state: str,
    current_state: str,
) -> bool:
    """Return True if a guardrail state transition occurred."""
    return prev_state != current_state


def threshold_crossing_direction(
    prev_state: str,
    current_state: str,
) -> str:
    """Return 'escalation', 'de-escalation', or 'no_change'."""
    order = {"normal": 0, "warn": 1, "derisk": 2, "halt": 3}
    prev_rank = order.get(prev_state, -1)
    curr_rank = order.get(current_state, -1)
    if curr_rank > prev_rank:
        return "escalation"
    if curr_rank < prev_rank:
        return "de-escalation"
    return "no_change"
