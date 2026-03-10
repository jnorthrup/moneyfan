"""
HRM model version registry and artifact provenance.

Each trained HRM checkpoint is assigned a semantic version and a provenance
record that links it to:
  - the training config (objective weights, trade-step schedule, etc.)
  - the checkpoint artifact paths (weights, model_config, feature_schema, objective_config)
  - a promotion-gate manifest (validation status, promotion decision, rollback flag)

Schema: moneyfan.hrm.model_version.v1

Semantic version format: MAJOR.MINOR.PATCH
  - MAJOR: incremented on breaking changes to model architecture (n_heads, hidden_dim, etc.)
  - MINOR: incremented on objective weight or training schedule changes
  - PATCH: incremented on incremental reruns with same config

The registry is a single JSONL file (append-only) stored at
  hrm/registry/model_version_registry.jsonl

Promotion gates:
  - "pending"   : artifact saved but not yet evaluated
  - "validated" : passed smoke evaluation (loss within expected bounds)
  - "promoted"  : promoted to paper/live serving
  - "rolled_back": demoted after regression
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


#: Canonical schema identifier
MODEL_VERSION_SCHEMA = "moneyfan.hrm.model_version.v1"

#: Valid promotion gate states
VALID_PROMOTION_STATES = frozenset((
    "pending",
    "validated",
    "promoted",
    "rolled_back",
))

#: Required fields in a serialised version record
REQUIRED_VERSION_RECORD_KEYS = (
    "schema",
    "version",
    "model_id",
    "created_at",
    "artifact_paths",
    "architecture_fingerprint",
    "objective_fingerprint",
    "promotion_state",
    "training_config_snapshot",
)

#: Required fields inside artifact_paths
REQUIRED_ARTIFACT_PATH_KEYS = (
    "weights_path",
    "config_path",
    "feature_schema_path",
    "objective_config_path",
)


def _sha256_prefix(data: str, length: int = 16) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:length]


def _architecture_fingerprint(model_config_dict: dict[str, Any]) -> str:
    """Stable fingerprint from architecture-defining config keys."""
    arch_keys = ("hidden_dim", "n_heads", "regime_attn_layers", "tactical_attn_layers", "input_dim")
    stable = {k: model_config_dict.get(k) for k in arch_keys}
    return _sha256_prefix(json.dumps(stable, sort_keys=True))


def _objective_fingerprint(objective_config: dict[str, Any]) -> str:
    """Stable fingerprint from objective weight config."""
    return _sha256_prefix(json.dumps(objective_config, sort_keys=True))


def _model_id(version: str, architecture_fingerprint: str) -> str:
    """Deterministic model_id from version and arch fingerprint."""
    return f"hrm-{version}-{architecture_fingerprint}"


@dataclass
class ModelVersionRecord:
    """One versioned HRM model artifact record."""

    version: str
    """Semantic version string: MAJOR.MINOR.PATCH"""

    model_id: str
    """Unique identifier: hrm-{version}-{arch_fingerprint}"""

    created_at: str
    """ISO-8601 UTC timestamp of record creation."""

    artifact_paths: dict[str, str]
    """Paths to weights, config, feature_schema, objective_config artifacts."""

    architecture_fingerprint: str
    """SHA-256[:16] of architecture-defining model config keys."""

    objective_fingerprint: str
    """SHA-256[:16] of objective weight config."""

    training_config_snapshot: dict[str, Any]
    """Snapshot of training config fields relevant to this version."""

    promotion_state: str = "pending"
    """Promotion gate state: pending | validated | promoted | rolled_back"""

    promotion_notes: Optional[str] = None
    """Optional notes on promotion decision."""

    rollback_reason: Optional[str] = None
    """Populated when state = rolled_back."""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": MODEL_VERSION_SCHEMA,
            "version": self.version,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "artifact_paths": dict(self.artifact_paths),
            "architecture_fingerprint": self.architecture_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "promotion_state": self.promotion_state,
            "training_config_snapshot": dict(self.training_config_snapshot),
        }
        if self.promotion_notes is not None:
            d["promotion_notes"] = self.promotion_notes
        if self.rollback_reason is not None:
            d["rollback_reason"] = self.rollback_reason
        return d

    def with_promotion(self, new_state: str, notes: Optional[str] = None) -> "ModelVersionRecord":
        """Return a new record with updated promotion_state."""
        if new_state not in VALID_PROMOTION_STATES:
            raise ValueError(
                f"promotion_state must be one of {sorted(VALID_PROMOTION_STATES)!r},"
                f" got {new_state!r}"
            )
        rollback = (self.rollback_reason if new_state != "rolled_back" else notes)
        return ModelVersionRecord(
            version=self.version,
            model_id=self.model_id,
            created_at=self.created_at,
            artifact_paths=self.artifact_paths,
            architecture_fingerprint=self.architecture_fingerprint,
            objective_fingerprint=self.objective_fingerprint,
            training_config_snapshot=self.training_config_snapshot,
            promotion_state=new_state,
            promotion_notes=notes if new_state != "rolled_back" else self.promotion_notes,
            rollback_reason=rollback,
        )


def build_version_record(
    *,
    version: str,
    artifact_paths: dict[str, str],
    model_config_dict: dict[str, Any],
    objective_config: dict[str, Any],
    training_config_snapshot: dict[str, Any],
    created_at: Optional[str] = None,
) -> ModelVersionRecord:
    """Build a ModelVersionRecord from training artifacts.

    Args:
        version: Semantic version string (e.g. "1.0.0").
        artifact_paths: Dict with at least the REQUIRED_ARTIFACT_PATH_KEYS.
        model_config_dict: Model architecture config as a dict.
        objective_config: Objective weight config dict.
        training_config_snapshot: Subset of training config for provenance.
        created_at: ISO-8601 timestamp override (defaults to now UTC).

    Returns:
        ModelVersionRecord with promotion_state="pending".
    """
    arch_fp = _architecture_fingerprint(model_config_dict)
    obj_fp = _objective_fingerprint(objective_config)
    model_id = _model_id(version, arch_fp)
    ts = created_at or datetime.now(timezone.utc).isoformat()
    return ModelVersionRecord(
        version=version,
        model_id=model_id,
        created_at=ts,
        artifact_paths=artifact_paths,
        architecture_fingerprint=arch_fp,
        objective_fingerprint=obj_fp,
        training_config_snapshot=training_config_snapshot,
        promotion_state="pending",
    )


def validate_version_record(record_dict: dict[str, Any]) -> list[str]:
    """Validate a serialised version record. Returns list of error strings."""
    errors: list[str] = []
    for key in REQUIRED_VERSION_RECORD_KEYS:
        if key not in record_dict:
            errors.append(f"Missing required key: {key!r}")
    if record_dict.get("schema") != MODEL_VERSION_SCHEMA:
        errors.append(
            f"schema mismatch: expected {MODEL_VERSION_SCHEMA!r},"
            f" got {record_dict.get('schema')!r}"
        )
    state = record_dict.get("promotion_state")
    if state not in VALID_PROMOTION_STATES:
        errors.append(
            f"promotion_state must be one of {sorted(VALID_PROMOTION_STATES)!r},"
            f" got {state!r}"
        )
    paths = record_dict.get("artifact_paths", {})
    for key in REQUIRED_ARTIFACT_PATH_KEYS:
        if key not in paths:
            errors.append(f"artifact_paths: missing required key {key!r}")
    return errors


class ModelVersionRegistry:
    """Append-only JSONL registry for HRM model version records.

    Usage:
        registry = ModelVersionRegistry(Path("hrm/registry/model_version_registry.jsonl"))
        registry.append(record)
        records = registry.load_all()
        latest = registry.latest_promoted()
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: ModelVersionRecord) -> None:
        """Append a version record to the registry file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record.as_dict()) + "\n")

    def load_all(self) -> list[ModelVersionRecord]:
        """Load all records from the registry file."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(_record_from_dict(d))
        return records

    def latest(self) -> Optional[ModelVersionRecord]:
        """Return the most recently appended record (last line)."""
        records = self.load_all()
        return records[-1] if records else None

    def latest_promoted(self) -> Optional[ModelVersionRecord]:
        """Return the most recently appended record with promotion_state='promoted'."""
        records = self.load_all()
        for r in reversed(records):
            if r.promotion_state == "promoted":
                return r
        return None

    def all_by_state(self, state: str) -> list[ModelVersionRecord]:
        return [r for r in self.load_all() if r.promotion_state == state]


def _record_from_dict(d: dict[str, Any]) -> ModelVersionRecord:
    return ModelVersionRecord(
        version=d["version"],
        model_id=d["model_id"],
        created_at=d["created_at"],
        artifact_paths=d["artifact_paths"],
        architecture_fingerprint=d["architecture_fingerprint"],
        objective_fingerprint=d["objective_fingerprint"],
        training_config_snapshot=d.get("training_config_snapshot", {}),
        promotion_state=d.get("promotion_state", "pending"),
        promotion_notes=d.get("promotion_notes"),
        rollback_reason=d.get("rollback_reason"),
    )
