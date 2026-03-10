"""
HRM rollback and recovery controller.

Provides three surfaces:

1. RollbackController.check_and_apply(record, metrics, registry, gate)
   - Evaluates the currently promoted model against live metrics.
   - If the promotion gate fires a rollback decision, applies the state
     change to the registry (appends rolled_back record) and returns
     the RollbackOutcome.
   - Idempotent: if the record is already rolled_back, returns a no-op outcome.

2. RollbackController.force_rollback(record, registry, reason)
   - Manual operator-initiated rollback. Does not consult the gate.
   - Appends a rolled_back record with the supplied reason.
   - Idempotent: if already rolled_back, returns a no-op outcome.

3. artifact_health_check(artifact_paths) -> ArtifactHealthReport
   - Checks that all required artifact files exist and are non-empty.
   - Returns a report with per-path status and an overall healthy flag.
   - Does NOT load/parse the artifacts; purely a filesystem presence check.

Schema: moneyfan.hrm.rollback.v1 (for RollbackOutcome serialization)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from execution.model_version_registry import (
    ModelVersionRecord,
    ModelVersionRegistry,
)
from execution.promotion_gate import (
    EvaluationMetrics,
    PromotionGate,
)


ROLLBACK_OUTCOME_SCHEMA = "moneyfan.hrm.rollback.v1"

#: Required fields in a serialised RollbackOutcome
REQUIRED_ROLLBACK_OUTCOME_KEYS = (
    "schema",
    "ts_utc",
    "model_id",
    "model_version",
    "triggered",
    "trigger_type",
    "previous_state",
    "next_state",
    "reason",
)

#: Valid trigger types
VALID_TRIGGER_TYPES = frozenset((
    "automatic",   # gate-driven regression detection
    "manual",      # operator force_rollback()
    "no_op",       # record was already in terminal state
))


@dataclass
class ArtifactHealthReport:
    """Result of a filesystem health check on model artifact paths."""

    healthy: bool
    """True if ALL required artifact files exist and are non-empty."""

    path_statuses: dict[str, str]
    """Map of artifact_path_key -> status string ('ok', 'missing', 'empty')."""

    checked_at: str
    """ISO-8601 UTC timestamp of the health check."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "path_statuses": dict(self.path_statuses),
            "checked_at": self.checked_at,
        }


def artifact_health_check(
    artifact_paths: dict[str, str],
    ts_utc: Optional[str] = None,
) -> ArtifactHealthReport:
    """Check that each artifact file exists and is non-empty.

    Args:
        artifact_paths: Dict mapping logical key -> filesystem path string.
        ts_utc: Optional timestamp override (defaults to now UTC).

    Returns:
        ArtifactHealthReport with per-path status and overall healthy flag.
    """
    ts = ts_utc or datetime.now(timezone.utc).isoformat()
    statuses: dict[str, str] = {}
    for key, path_str in artifact_paths.items():
        p = Path(path_str)
        if not p.exists():
            statuses[key] = "missing"
        elif p.stat().st_size == 0:
            statuses[key] = "empty"
        else:
            statuses[key] = "ok"
    healthy = all(s == "ok" for s in statuses.values())
    return ArtifactHealthReport(healthy=healthy, path_statuses=statuses, checked_at=ts)


@dataclass
class RollbackOutcome:
    """Result of a rollback operation (automatic or manual)."""

    triggered: bool
    """True if a rollback was actually applied."""

    trigger_type: str
    """'automatic' | 'manual' | 'no_op'"""

    previous_state: str
    """Promotion state before the rollback."""

    next_state: str
    """Promotion state after the rollback ('rolled_back' or unchanged)."""

    model_id: str
    reason: str
    model_version: str
    ts_utc: str

    gate_failures: Optional[list[str]] = None
    """Gate failure messages that caused an automatic rollback."""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": ROLLBACK_OUTCOME_SCHEMA,
            "ts_utc": self.ts_utc,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "triggered": self.triggered,
            "trigger_type": self.trigger_type,
            "previous_state": self.previous_state,
            "next_state": self.next_state,
            "reason": self.reason,
        }
        if self.gate_failures is not None:
            d["gate_failures"] = list(self.gate_failures)
        return d


def validate_rollback_outcome(outcome_dict: dict[str, Any]) -> list[str]:
    """Validate a serialised RollbackOutcome. Returns list of error strings."""
    errors: list[str] = []
    for key in REQUIRED_ROLLBACK_OUTCOME_KEYS:
        if key not in outcome_dict:
            errors.append(f"Missing required key: {key!r}")
    if outcome_dict.get("schema") != ROLLBACK_OUTCOME_SCHEMA:
        errors.append(
            f"schema mismatch: expected {ROLLBACK_OUTCOME_SCHEMA!r},"
            f" got {outcome_dict.get('schema')!r}"
        )
    trigger = outcome_dict.get("trigger_type")
    if trigger not in VALID_TRIGGER_TYPES:
        errors.append(
            f"trigger_type must be one of {sorted(VALID_TRIGGER_TYPES)!r},"
            f" got {trigger!r}"
        )
    return errors


class RollbackController:
    """Coordinates rollback detection and recovery for the HRM model registry.

    Usage (automatic):
        controller = RollbackController(rollback_log_path=Path("hrm/registry/rollback.jsonl"))
        outcome = controller.check_and_apply(promoted_record, live_metrics, registry, gate)
        if outcome.triggered:
            print(f"Rolled back {outcome.model_id}: {outcome.reason}")

    Usage (manual):
        outcome = controller.force_rollback(record, registry, reason="operator override")
    """

    def __init__(self, rollback_log_path: Optional[Path] = None) -> None:
        self.rollback_log_path = (
            Path(rollback_log_path) if rollback_log_path else None
        )

    def check_and_apply(
        self,
        record: ModelVersionRecord,
        metrics: EvaluationMetrics,
        registry: ModelVersionRegistry,
        gate: PromotionGate,
        ts_utc: Optional[str] = None,
    ) -> RollbackOutcome:
        """Evaluate a promoted model; roll it back via the registry if gate fires.

        Idempotent: if the record is already rolled_back, returns no_op outcome.
        Only acts on records in 'promoted' state.

        Returns:
            RollbackOutcome describing what happened.
        """
        ts = ts_utc or datetime.now(timezone.utc).isoformat()

        if record.promotion_state == "rolled_back":
            return RollbackOutcome(
                triggered=False,
                trigger_type="no_op",
                previous_state="rolled_back",
                next_state="rolled_back",
                model_id=record.model_id,
                model_version=record.version,
                reason="already rolled_back; no action taken",
                ts_utc=ts,
            )

        if record.promotion_state != "promoted":
            return RollbackOutcome(
                triggered=False,
                trigger_type="no_op",
                previous_state=record.promotion_state,
                next_state=record.promotion_state,
                model_id=record.model_id,
                model_version=record.version,
                reason=(
                    f"record is in state {record.promotion_state!r},"
                    " not 'promoted'; check_and_apply only monitors promoted models"
                ),
                ts_utc=ts,
            )

        decision = gate.evaluate(record, metrics, ts_utc=ts)

        if decision.next_state == "rolled_back":
            rolled_back_record = record.with_promotion(
                "rolled_back",
                notes="; ".join(decision.failures),
            )
            registry.append(rolled_back_record)
            outcome = RollbackOutcome(
                triggered=True,
                trigger_type="automatic",
                previous_state=record.promotion_state,
                next_state="rolled_back",
                model_id=record.model_id,
                model_version=record.version,
                reason="; ".join(decision.failures),
                gate_failures=decision.failures,
                ts_utc=ts,
            )
            self._append_rollback_log(outcome.as_dict())
            return outcome

        return RollbackOutcome(
            triggered=False,
            trigger_type="no_op",
            previous_state=record.promotion_state,
            next_state=record.promotion_state,
            model_id=record.model_id,
            model_version=record.version,
            reason="gate passed; no rollback needed",
            ts_utc=ts,
        )

    def force_rollback(
        self,
        record: ModelVersionRecord,
        registry: ModelVersionRegistry,
        reason: str,
        ts_utc: Optional[str] = None,
    ) -> RollbackOutcome:
        """Operator-initiated rollback. Bypasses gate criteria.

        Idempotent: if already rolled_back, returns a no_op outcome.

        Returns:
            RollbackOutcome with trigger_type='manual'.
        """
        ts = ts_utc or datetime.now(timezone.utc).isoformat()

        if record.promotion_state == "rolled_back":
            return RollbackOutcome(
                triggered=False,
                trigger_type="no_op",
                previous_state="rolled_back",
                next_state="rolled_back",
                model_id=record.model_id,
                model_version=record.version,
                reason="already rolled_back; force_rollback is a no-op",
                ts_utc=ts,
            )

        rolled_back_record = record.with_promotion("rolled_back", notes=reason)
        registry.append(rolled_back_record)

        outcome = RollbackOutcome(
            triggered=True,
            trigger_type="manual",
            previous_state=record.promotion_state,
            next_state="rolled_back",
            model_id=record.model_id,
            model_version=record.version,
            reason=reason,
            ts_utc=ts,
        )
        self._append_rollback_log(outcome.as_dict())
        return outcome

    def load_rollback_log(self) -> list[dict[str, Any]]:
        """Load all entries from the rollback log file."""
        if self.rollback_log_path is None or not self.rollback_log_path.exists():
            return []
        entries = []
        with open(self.rollback_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _append_rollback_log(self, entry: dict[str, Any]) -> None:
        if self.rollback_log_path is None:
            return
        self.rollback_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rollback_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
