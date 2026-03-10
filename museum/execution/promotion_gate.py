"""
HRM model governance and promotion gates.

PromotionGate evaluates whether a ModelVersionRecord meets the criteria
for advancement from 'pending' -> 'validated' -> 'promoted', or should
be demoted to 'rolled_back'.

Promotion criteria are expressed as a PromotionPolicy dataclass:
  - min_hit_rate: minimum trade hit rate in evaluation
  - max_mean_loss: maximum acceptable world-model loss
  - min_profit_factor: minimum (gross_profit / gross_loss) ratio
  - min_episodes_evaluated: minimum #episodes that must pass through gate
  - max_drawdown_pct: maximum portfolio drawdown during eval window

Audit trail:
  - Every gate decision is appended to a JSONL audit log (append-only)
  - Schema: moneyfan.hrm.promotion_gate.audit.v1

Multi-slice validation:
  - A PromotionGate.multi_slice_validate() method requires N slices to
    all pass before the gate opens. This prevents single-lucky-run promotion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from execution.model_version_registry import (
    VALID_PROMOTION_STATES,
    ModelVersionRecord,
    ModelVersionRegistry,
)


#: Canonical schema identifier for audit log entries
PROMOTION_AUDIT_SCHEMA = "moneyfan.hrm.promotion_gate.audit.v1"

#: Required fields in a serialised audit log entry
REQUIRED_AUDIT_KEYS = (
    "schema",
    "ts_utc",
    "model_id",
    "model_version",
    "decision",
    "previous_state",
    "next_state",
    "criteria_snapshot",
    "metrics_snapshot",
    "passed",
)

#: Valid gate decision identifiers
VALID_DECISIONS = frozenset((
    "promote_to_validated",
    "promote_to_promoted",
    "rollback",
    "hold_pending",
    "hold_validated",
))


@dataclass(frozen=True)
class PromotionPolicy:
    """Criteria that must all pass for a model to advance through a gate."""

    min_hit_rate: float = 0.50
    """Minimum fraction of profitable trades in the evaluation window."""

    max_mean_loss: float = 0.5
    """Maximum acceptable mean world-model loss (lower is better)."""

    min_profit_factor: float = 1.1
    """Minimum gross_profit / gross_loss ratio (> 1.0 = net positive)."""

    min_episodes_evaluated: int = 3
    """Minimum number of evaluation episodes that must pass through."""

    max_drawdown_pct: float = 0.15
    """Maximum portfolio drawdown fraction during evaluation window."""

    rollback_drawdown_pct: float = 0.20
    """Drawdown level above which a promoted model is automatically rolled back."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_hit_rate": self.min_hit_rate,
            "max_mean_loss": self.max_mean_loss,
            "min_profit_factor": self.min_profit_factor,
            "min_episodes_evaluated": self.min_episodes_evaluated,
            "max_drawdown_pct": self.max_drawdown_pct,
            "rollback_drawdown_pct": self.rollback_drawdown_pct,
        }


@dataclass
class EvaluationMetrics:
    """Metrics from one evaluation pass."""

    hit_rate: float
    """Fraction of profitable trade signals."""

    mean_loss: float
    """Mean world-model loss across evaluation episodes."""

    profit_factor: float
    """gross_profit / gross_loss ratio."""

    episodes_evaluated: int
    """Number of episodes in this evaluation window."""

    max_drawdown_pct: float
    """Maximum drawdown fraction observed during evaluation."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hit_rate": self.hit_rate,
            "mean_loss": self.mean_loss,
            "profit_factor": self.profit_factor,
            "episodes_evaluated": self.episodes_evaluated,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


@dataclass
class GateDecision:
    """Result of a single promotion gate evaluation."""

    passed: bool
    """True if all criteria were met."""

    decision: str
    """One of VALID_DECISIONS."""

    previous_state: str
    """Promotion state before this gate evaluation."""

    next_state: str
    """Promotion state after this gate evaluation."""

    failures: list[str]
    """List of criteria that were not met (empty if passed)."""

    policy: PromotionPolicy
    """Policy used for this evaluation."""

    metrics: EvaluationMetrics
    """Metrics observed during this evaluation."""

    notes: Optional[str] = None
    """Optional human-readable context."""

    def as_audit_dict(
        self,
        model_id: str,
        model_version: str,
        ts_utc: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "schema": PROMOTION_AUDIT_SCHEMA,
            "ts_utc": ts_utc or datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "model_version": model_version,
            "decision": self.decision,
            "previous_state": self.previous_state,
            "next_state": self.next_state,
            "passed": self.passed,
            "failures": list(self.failures),
            "criteria_snapshot": self.policy.as_dict(),
            "metrics_snapshot": self.metrics.as_dict(),
            "notes": self.notes,
        }


class PromotionGate:
    """Evaluates a model record against a PromotionPolicy.

    Usage:
        gate = PromotionGate(policy, audit_log_path=Path("hrm/registry/audit.jsonl"))
        decision = gate.evaluate(record, metrics)
        if decision.passed:
            new_record = record.with_promotion(decision.next_state)
            registry.append(new_record)
    """

    def __init__(
        self,
        policy: PromotionPolicy,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        self.policy = policy
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None

    def evaluate(
        self,
        record: ModelVersionRecord,
        metrics: EvaluationMetrics,
        notes: Optional[str] = None,
        ts_utc: Optional[str] = None,
    ) -> GateDecision:
        """Evaluate a model record against the policy.

        Returns a GateDecision. Does NOT mutate the record or registry.
        Call registry.append(record.with_promotion(decision.next_state)) to apply.
        """
        prev = record.promotion_state
        failures = self._check_criteria(metrics, prev)
        passed = len(failures) == 0

        decision_id, next_state = self._resolve_transition(prev, passed, metrics)

        decision = GateDecision(
            passed=passed,
            decision=decision_id,
            previous_state=prev,
            next_state=next_state,
            failures=failures,
            policy=self.policy,
            metrics=metrics,
            notes=notes,
        )

        if self.audit_log_path is not None:
            self._append_audit(
                decision.as_audit_dict(record.model_id, record.version, ts_utc)
            )

        return decision

    def multi_slice_validate(
        self,
        record: ModelVersionRecord,
        metrics_slices: list[EvaluationMetrics],
        notes: Optional[str] = None,
    ) -> GateDecision:
        """Require every slice in metrics_slices to pass before opening gate.

        The returned decision is 'passed' only if ALL slices pass.
        The failures from all slices are aggregated.
        """
        if len(metrics_slices) < self.policy.min_episodes_evaluated:
            failures = [
                f"multi_slice: only {len(metrics_slices)} slices provided;"
                f" policy requires >= {self.policy.min_episodes_evaluated}"
            ]
            return GateDecision(
                passed=False,
                decision="hold_pending",
                previous_state=record.promotion_state,
                next_state=record.promotion_state,
                failures=failures,
                policy=self.policy,
                metrics=metrics_slices[0] if metrics_slices else EvaluationMetrics(
                    hit_rate=0.0, mean_loss=999.0, profit_factor=0.0,
                    episodes_evaluated=0, max_drawdown_pct=1.0
                ),
                notes=notes,
            )

        all_failures: list[str] = []
        for idx, m in enumerate(metrics_slices):
            slice_failures = self._check_criteria(m, record.promotion_state)
            for f in slice_failures:
                all_failures.append(f"slice[{idx}]: {f}")

        passed = len(all_failures) == 0
        # Use worst-case metrics for audit
        worst = max(metrics_slices, key=lambda m: (
            -m.hit_rate + m.mean_loss - m.profit_factor + m.max_drawdown_pct
        ))
        decision_id, next_state = self._resolve_transition(
            record.promotion_state, passed, worst
        )
        return GateDecision(
            passed=passed,
            decision=decision_id,
            previous_state=record.promotion_state,
            next_state=next_state,
            failures=all_failures,
            policy=self.policy,
            metrics=worst,
            notes=notes,
        )

    def _check_criteria(
        self, metrics: EvaluationMetrics, current_state: str
    ) -> list[str]:
        """Return list of failure strings (empty = pass)."""
        failures: list[str] = []

        if metrics.hit_rate < self.policy.min_hit_rate:
            failures.append(
                f"hit_rate {metrics.hit_rate:.3f} < min {self.policy.min_hit_rate:.3f}"
            )
        if metrics.mean_loss > self.policy.max_mean_loss:
            failures.append(
                f"mean_loss {metrics.mean_loss:.4f} > max {self.policy.max_mean_loss:.4f}"
            )
        if metrics.profit_factor < self.policy.min_profit_factor:
            failures.append(
                f"profit_factor {metrics.profit_factor:.3f} < min {self.policy.min_profit_factor:.3f}"
            )
        if metrics.episodes_evaluated < self.policy.min_episodes_evaluated:
            failures.append(
                f"episodes_evaluated {metrics.episodes_evaluated}"
                f" < min {self.policy.min_episodes_evaluated}"
            )
        if current_state == "promoted":
            # Rollback check uses the stricter rollback_drawdown_pct
            if metrics.max_drawdown_pct > self.policy.rollback_drawdown_pct:
                failures.append(
                    f"max_drawdown {metrics.max_drawdown_pct:.3f}"
                    f" > rollback threshold {self.policy.rollback_drawdown_pct:.3f}"
                )
        else:
            if metrics.max_drawdown_pct > self.policy.max_drawdown_pct:
                failures.append(
                    f"max_drawdown {metrics.max_drawdown_pct:.3f}"
                    f" > max {self.policy.max_drawdown_pct:.3f}"
                )
        return failures

    def _resolve_transition(
        self,
        current_state: str,
        passed: bool,
        metrics: EvaluationMetrics,
    ) -> tuple[str, str]:
        """Return (decision_id, next_state)."""
        if not passed:
            if current_state == "promoted":
                return "rollback", "rolled_back"
            return (
                "hold_pending" if current_state == "pending" else "hold_validated",
                current_state,
            )
        if current_state == "pending":
            return "promote_to_validated", "validated"
        if current_state == "validated":
            return "promote_to_promoted", "promoted"
        if current_state == "promoted":
            return "hold_validated", "promoted"  # already promoted, no change
        return "hold_pending", current_state

    def _append_audit(self, entry: dict[str, Any]) -> None:
        if self.audit_log_path is None:
            return
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def load_audit_log(path: Path) -> list[dict[str, Any]]:
    """Load all entries from a promotion gate audit JSONL log."""
    if not Path(path).exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate_audit_entry(entry: dict[str, Any]) -> list[str]:
    """Validate an audit log entry. Returns list of error strings."""
    errors: list[str] = []
    for key in REQUIRED_AUDIT_KEYS:
        if key not in entry:
            errors.append(f"Missing required key: {key!r}")
    if entry.get("schema") != PROMOTION_AUDIT_SCHEMA:
        errors.append(
            f"schema mismatch: expected {PROMOTION_AUDIT_SCHEMA!r},"
            f" got {entry.get('schema')!r}"
        )
    decision = entry.get("decision")
    if decision not in VALID_DECISIONS:
        errors.append(
            f"decision must be one of {sorted(VALID_DECISIONS)!r},"
            f" got {decision!r}"
        )
    return errors
