"""
HRM model observability and monitoring.

Three surfaces:

1. ModelPerformanceReport
   - Summarises all records from a ModelVersionRegistry into a structured
     report: version series, state distribution, promotion timeline,
     and last-known metrics per version.
   - Schema: moneyfan.hrm.observability.report.v1

2. DegradationAlerter
   - Evaluates EvaluationMetrics against alert thresholds (less strict than
     promotion gate; used for early-warning monitoring).
   - Writes alert events to an append-only JSONL alert log.
   - Schema: moneyfan.hrm.observability.alert.v1

3. DecisionLogIndex
   - Merges audit log entries (from promotion_gate.py) and rollback log
     entries (from rollback_controller.py) into a single chronological
     decision index for a given model_id.
   - Supports filtering by decision type, date range, and model version.
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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

REPORT_SCHEMA = "moneyfan.hrm.observability.report.v1"
ALERT_SCHEMA = "moneyfan.hrm.observability.alert.v1"

REQUIRED_REPORT_KEYS = (
    "schema",
    "generated_at",
    "total_records",
    "state_distribution",
    "version_series",
    "latest_promoted",
)

REQUIRED_ALERT_KEYS = (
    "schema",
    "ts_utc",
    "model_id",
    "model_version",
    "alert_type",
    "metric_name",
    "metric_value",
    "threshold",
    "message",
)

VALID_ALERT_TYPES = frozenset((
    "degradation_warning",
    "degradation_critical",
    "drawdown_warning",
    "drawdown_critical",
    "loss_spike",
))


# ---------------------------------------------------------------------------
# 1. ModelPerformanceReport
# ---------------------------------------------------------------------------

@dataclass
class ModelPerformanceReport:
    """Structured summary of all records in a ModelVersionRegistry."""

    generated_at: str
    total_records: int
    state_distribution: dict[str, int]
    """Count of records per promotion_state."""

    version_series: list[dict[str, Any]]
    """Chronological list of (version, model_id, state, created_at)."""

    latest_promoted: Optional[dict[str, str]]
    """The most recently promoted record's (version, model_id, created_at), or None."""

    rollback_rate: float
    """Fraction of records that ended up rolled_back (0.0 if none)."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "generated_at": self.generated_at,
            "total_records": self.total_records,
            "state_distribution": dict(self.state_distribution),
            "version_series": list(self.version_series),
            "latest_promoted": self.latest_promoted,
            "rollback_rate": self.rollback_rate,
        }


def build_performance_report(
    registry: ModelVersionRegistry,
    ts_utc: Optional[str] = None,
) -> ModelPerformanceReport:
    """Build a ModelPerformanceReport from all records in a registry."""
    ts = ts_utc or datetime.now(timezone.utc).isoformat()
    records = registry.load_all()

    state_dist: dict[str, int] = {}
    for r in records:
        state_dist[r.promotion_state] = state_dist.get(r.promotion_state, 0) + 1

    version_series = [
        {
            "version": r.version,
            "model_id": r.model_id,
            "state": r.promotion_state,
            "created_at": r.created_at,
        }
        for r in records
    ]

    promoted = registry.latest_promoted()
    latest_promoted_dict: Optional[dict[str, str]] = None
    if promoted is not None:
        latest_promoted_dict = {
            "version": promoted.version,
            "model_id": promoted.model_id,
            "created_at": promoted.created_at,
        }

    n_total = len(records)
    n_rolled = state_dist.get("rolled_back", 0)
    rollback_rate = n_rolled / n_total if n_total > 0 else 0.0

    return ModelPerformanceReport(
        generated_at=ts,
        total_records=n_total,
        state_distribution=state_dist,
        version_series=version_series,
        latest_promoted=latest_promoted_dict,
        rollback_rate=rollback_rate,
    )


def validate_report(report_dict: dict[str, Any]) -> list[str]:
    """Validate a serialised ModelPerformanceReport. Returns error strings."""
    errors: list[str] = []
    for key in REQUIRED_REPORT_KEYS:
        if key not in report_dict:
            errors.append(f"Missing required key: {key!r}")
    if report_dict.get("schema") != REPORT_SCHEMA:
        errors.append(
            f"schema mismatch: expected {REPORT_SCHEMA!r},"
            f" got {report_dict.get('schema')!r}"
        )
    return errors


# ---------------------------------------------------------------------------
# 2. DegradationAlerter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlertThresholds:
    """Warning/critical thresholds for metrics-based alerts."""

    hit_rate_warning: float = 0.48
    hit_rate_critical: float = 0.40
    mean_loss_warning: float = 0.45
    mean_loss_critical: float = 0.60
    max_drawdown_warning: float = 0.12
    max_drawdown_critical: float = 0.18

    def as_dict(self) -> dict[str, Any]:
        return {
            "hit_rate_warning": self.hit_rate_warning,
            "hit_rate_critical": self.hit_rate_critical,
            "mean_loss_warning": self.mean_loss_warning,
            "mean_loss_critical": self.mean_loss_critical,
            "max_drawdown_warning": self.max_drawdown_warning,
            "max_drawdown_critical": self.max_drawdown_critical,
        }


@dataclass
class AlertEvent:
    """A single degradation alert event."""

    alert_type: str
    metric_name: str
    metric_value: float
    threshold: float
    model_id: str
    model_version: str
    message: str
    ts_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ALERT_SCHEMA,
            "ts_utc": self.ts_utc,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "alert_type": self.alert_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "message": self.message,
        }


class DegradationAlerter:
    """Evaluates live metrics against thresholds and writes alert events.

    Usage:
        alerter = DegradationAlerter(
            thresholds=AlertThresholds(),
            alert_log_path=Path("hrm/registry/alerts.jsonl"),
        )
        events = alerter.evaluate(record, metrics)
        for event in events:
            print(event.message)
    """

    def __init__(
        self,
        thresholds: Optional[AlertThresholds] = None,
        alert_log_path: Optional[Path] = None,
    ) -> None:
        self.thresholds = thresholds or AlertThresholds()
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None

    def evaluate(
        self,
        record: ModelVersionRecord,
        metrics: "EvaluationMetricsLike",
        ts_utc: Optional[str] = None,
    ) -> list[AlertEvent]:
        """Check metrics against thresholds; return list of alert events fired.

        Does not raise on threshold breach — returns the events for the caller
        to act on. Appends to alert_log_path if set.
        """
        ts = ts_utc or datetime.now(timezone.utc).isoformat()
        events: list[AlertEvent] = []
        t = self.thresholds

        hit_rate = float(metrics.hit_rate)
        if hit_rate < t.hit_rate_critical:
            events.append(AlertEvent(
                alert_type="degradation_critical",
                metric_name="hit_rate",
                metric_value=hit_rate,
                threshold=t.hit_rate_critical,
                model_id=record.model_id,
                model_version=record.version,
                message=(
                    f"CRITICAL: hit_rate {hit_rate:.3f}"
                    f" < critical threshold {t.hit_rate_critical:.3f}"
                ),
                ts_utc=ts,
            ))
        elif hit_rate < t.hit_rate_warning:
            events.append(AlertEvent(
                alert_type="degradation_warning",
                metric_name="hit_rate",
                metric_value=hit_rate,
                threshold=t.hit_rate_warning,
                message=(
                    f"WARNING: hit_rate {hit_rate:.3f}"
                    f" < warning threshold {t.hit_rate_warning:.3f}"
                ),
                model_id=record.model_id,
                model_version=record.version,
                ts_utc=ts,
            ))

        mean_loss = float(metrics.mean_loss)
        if mean_loss > t.mean_loss_critical:
            events.append(AlertEvent(
                alert_type="loss_spike",
                metric_name="mean_loss",
                metric_value=mean_loss,
                threshold=t.mean_loss_critical,
                message=(
                    f"CRITICAL: mean_loss {mean_loss:.4f}"
                    f" > critical threshold {t.mean_loss_critical:.4f}"
                ),
                model_id=record.model_id,
                model_version=record.version,
                ts_utc=ts,
            ))
        elif mean_loss > t.mean_loss_warning:
            events.append(AlertEvent(
                alert_type="loss_spike",
                metric_name="mean_loss",
                metric_value=mean_loss,
                threshold=t.mean_loss_warning,
                message=(
                    f"WARNING: mean_loss {mean_loss:.4f}"
                    f" > warning threshold {t.mean_loss_warning:.4f}"
                ),
                model_id=record.model_id,
                model_version=record.version,
                ts_utc=ts,
            ))

        dd = float(metrics.max_drawdown_pct)
        if dd > t.max_drawdown_critical:
            events.append(AlertEvent(
                alert_type="drawdown_critical",
                metric_name="max_drawdown_pct",
                metric_value=dd,
                threshold=t.max_drawdown_critical,
                message=(
                    f"CRITICAL: max_drawdown {dd:.3f}"
                    f" > critical threshold {t.max_drawdown_critical:.3f}"
                ),
                model_id=record.model_id,
                model_version=record.version,
                ts_utc=ts,
            ))
        elif dd > t.max_drawdown_warning:
            events.append(AlertEvent(
                alert_type="drawdown_warning",
                metric_name="max_drawdown_pct",
                metric_value=dd,
                threshold=t.max_drawdown_warning,
                message=(
                    f"WARNING: max_drawdown {dd:.3f}"
                    f" > warning threshold {t.max_drawdown_warning:.3f}"
                ),
                model_id=record.model_id,
                model_version=record.version,
                ts_utc=ts,
            ))

        if events and self.alert_log_path is not None:
            self._append_alerts(events)

        return events

    def load_alert_log(self) -> list[dict[str, Any]]:
        """Load all entries from the alert JSONL log."""
        if self.alert_log_path is None or not self.alert_log_path.exists():
            return []
        entries = []
        with open(self.alert_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _append_alerts(self, events: list[AlertEvent]) -> None:
        if self.alert_log_path is None:
            return
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alert_log_path, "a") as f:
            for e in events:
                f.write(json.dumps(e.as_dict()) + "\n")


def validate_alert_event(event_dict: dict[str, Any]) -> list[str]:
    """Validate a serialised AlertEvent. Returns list of error strings."""
    errors: list[str] = []
    for key in REQUIRED_ALERT_KEYS:
        if key not in event_dict:
            errors.append(f"Missing required key: {key!r}")
    if event_dict.get("schema") != ALERT_SCHEMA:
        errors.append(
            f"schema mismatch: expected {ALERT_SCHEMA!r},"
            f" got {event_dict.get('schema')!r}"
        )
    alert_type = event_dict.get("alert_type")
    if alert_type not in VALID_ALERT_TYPES:
        errors.append(
            f"alert_type must be one of {sorted(VALID_ALERT_TYPES)!r},"
            f" got {alert_type!r}"
        )
    return errors


# ---------------------------------------------------------------------------
# 3. DecisionLogIndex
# ---------------------------------------------------------------------------

@dataclass
class DecisionLogEntry:
    """A single entry in the unified decision log."""

    ts_utc: str
    source: str
    """'audit' (from promotion_gate) or 'rollback' (from rollback_controller)."""

    model_id: str
    model_version: str
    decision: str
    previous_state: str
    next_state: str
    passed: Optional[bool]
    reason: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts_utc": self.ts_utc,
            "source": self.source,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "decision": self.decision,
            "previous_state": self.previous_state,
            "next_state": self.next_state,
        }
        if self.passed is not None:
            d["passed"] = self.passed
        if self.reason is not None:
            d["reason"] = self.reason
        return d


class DecisionLogIndex:
    """Merges audit log + rollback log into a unified chronological index.

    Usage:
        index = DecisionLogIndex(
            audit_log_path=Path("hrm/registry/audit.jsonl"),
            rollback_log_path=Path("hrm/registry/rollback.jsonl"),
        )
        all_decisions = index.load_all()
        model_decisions = index.for_model("hrm-1.0.0-abc123")
    """

    def __init__(
        self,
        audit_log_path: Optional[Path] = None,
        rollback_log_path: Optional[Path] = None,
    ) -> None:
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.rollback_log_path = Path(rollback_log_path) if rollback_log_path else None

    def load_all(self) -> list[DecisionLogEntry]:
        """Load and merge all entries from audit and rollback logs, sorted by ts_utc."""
        entries: list[DecisionLogEntry] = []
        entries.extend(self._load_audit())
        entries.extend(self._load_rollback())
        entries.sort(key=lambda e: e.ts_utc)
        return entries

    def for_model(self, model_id: str) -> list[DecisionLogEntry]:
        """Return all decisions for a specific model_id, sorted by ts_utc."""
        return [e for e in self.load_all() if e.model_id == model_id]

    def for_version(self, model_version: str) -> list[DecisionLogEntry]:
        """Return all decisions for a specific model version string."""
        return [e for e in self.load_all() if e.model_version == model_version]

    def _load_audit(self) -> list[DecisionLogEntry]:
        if self.audit_log_path is None or not self.audit_log_path.exists():
            return []
        entries = []
        with open(self.audit_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries.append(DecisionLogEntry(
                    ts_utc=d.get("ts_utc", ""),
                    source="audit",
                    model_id=d.get("model_id", ""),
                    model_version=d.get("model_version", ""),
                    decision=d.get("decision", ""),
                    previous_state=d.get("previous_state", ""),
                    next_state=d.get("next_state", ""),
                    passed=d.get("passed"),
                    reason=None,
                ))
        return entries

    def _load_rollback(self) -> list[DecisionLogEntry]:
        if self.rollback_log_path is None or not self.rollback_log_path.exists():
            return []
        entries = []
        with open(self.rollback_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries.append(DecisionLogEntry(
                    ts_utc=d.get("ts_utc", ""),
                    source="rollback",
                    model_id=d.get("model_id", ""),
                    model_version=d.get("model_version", ""),
                    decision=d.get("trigger_type", ""),
                    previous_state=d.get("previous_state", ""),
                    next_state=d.get("next_state", ""),
                    passed=None,
                    reason=d.get("reason"),
                ))
        return entries


# ---------------------------------------------------------------------------
# Protocol-compatible stub for EvaluationMetrics duck-typing
# (avoids circular import — any object with hit_rate, mean_loss, max_drawdown_pct works)
# ---------------------------------------------------------------------------

class EvaluationMetricsLike:
    """Protocol stub — used only for type annotation in DegradationAlerter.evaluate()."""
    hit_rate: float
    mean_loss: float
    max_drawdown_pct: float
