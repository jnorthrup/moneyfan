"""
Tests for HRM model version registry and artifact provenance (Phase 4 slice 1).

Validates:
  - ModelVersionRecord construction and as_dict() schema completeness
  - architecture_fingerprint and objective_fingerprint determinism
  - build_version_record() produces correct initial state
  - with_promotion() state machine transitions
  - validate_version_record() error detection
  - ModelVersionRegistry: append, load_all, latest, latest_promoted, all_by_state
  - Registry JSONL round-trip fidelity
  - JSON-serializability
  - model_id stability: same arch+version -> same model_id
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pytest

from execution.model_version_registry import (
    MODEL_VERSION_SCHEMA,
    REQUIRED_VERSION_RECORD_KEYS,
    REQUIRED_ARTIFACT_PATH_KEYS,
    VALID_PROMOTION_STATES,
    ModelVersionRecord,
    ModelVersionRegistry,
    build_version_record,
    validate_version_record,
    _architecture_fingerprint,
    _objective_fingerprint,
    _model_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-03-10T01:26:00+00:00"

SAMPLE_MODEL_CONFIG = {
    "hidden_dim": 256,
    "n_heads": 8,
    "regime_attn_layers": 4,
    "tactical_attn_layers": 2,
    "input_dim": 48,
    "n_signals": 24,
}

SAMPLE_OBJECTIVE_CONFIG = {
    "world_model_weight": 1.0,
    "trade_head_weight": 0.5,
    "cost_turnover_weight": 0.1,
    "regime_weight_scale": 1.0,
}

SAMPLE_ARTIFACT_PATHS = {
    "weights_path": "/hrm/checkpoints/hrm_latest_weights.npz",
    "config_path": "/hrm/checkpoints/hrm_latest_model_config.json",
    "feature_schema_path": "/hrm/checkpoints/hrm_latest_feature_schema.json",
    "objective_config_path": "/hrm/checkpoints/hrm_latest_objective_config.json",
}

SAMPLE_TRAINING_CONFIG = {
    "n_epoch_episodes": 100,
    "optimizer_name": "adamw",
    "trade_update_prob": 0.3,
}


def _build_record(**overrides) -> ModelVersionRecord:
    kwargs = dict(
        version="1.0.0",
        artifact_paths=SAMPLE_ARTIFACT_PATHS,
        model_config_dict=SAMPLE_MODEL_CONFIG,
        objective_config=SAMPLE_OBJECTIVE_CONFIG,
        training_config_snapshot=SAMPLE_TRAINING_CONFIG,
        created_at=FIXED_TS,
    )
    kwargs.update(overrides)
    return build_version_record(**kwargs)


# ---------------------------------------------------------------------------
# Fingerprint determinism
# ---------------------------------------------------------------------------

def test_architecture_fingerprint_deterministic():
    fp1 = _architecture_fingerprint(SAMPLE_MODEL_CONFIG)
    fp2 = _architecture_fingerprint(SAMPLE_MODEL_CONFIG)
    assert fp1 == fp2


def test_architecture_fingerprint_length():
    fp = _architecture_fingerprint(SAMPLE_MODEL_CONFIG)
    assert len(fp) == 16


def test_architecture_fingerprint_changes_with_hidden_dim():
    config_a = dict(SAMPLE_MODEL_CONFIG, hidden_dim=256)
    config_b = dict(SAMPLE_MODEL_CONFIG, hidden_dim=512)
    assert _architecture_fingerprint(config_a) != _architecture_fingerprint(config_b)


def test_objective_fingerprint_deterministic():
    fp1 = _objective_fingerprint(SAMPLE_OBJECTIVE_CONFIG)
    fp2 = _objective_fingerprint(SAMPLE_OBJECTIVE_CONFIG)
    assert fp1 == fp2


def test_objective_fingerprint_changes_with_weight():
    obj_a = dict(SAMPLE_OBJECTIVE_CONFIG, trade_head_weight=0.5)
    obj_b = dict(SAMPLE_OBJECTIVE_CONFIG, trade_head_weight=1.0)
    assert _objective_fingerprint(obj_a) != _objective_fingerprint(obj_b)


def test_model_id_format():
    mid = _model_id("1.0.0", "abcdef0123456789")
    assert mid == "hrm-1.0.0-abcdef0123456789"


def test_model_id_stable_for_same_inputs():
    arch = _architecture_fingerprint(SAMPLE_MODEL_CONFIG)
    id1 = _model_id("1.0.0", arch)
    id2 = _model_id("1.0.0", arch)
    assert id1 == id2


# ---------------------------------------------------------------------------
# build_version_record
# ---------------------------------------------------------------------------

def test_build_version_record_initial_state():
    r = _build_record()
    assert r.promotion_state == "pending"
    assert r.version == "1.0.0"
    assert r.created_at == FIXED_TS


def test_build_version_record_model_id_contains_version():
    r = _build_record(version="2.1.3")
    assert "2.1.3" in r.model_id


def test_build_version_record_fingerprints_match():
    r = _build_record()
    expected_arch = _architecture_fingerprint(SAMPLE_MODEL_CONFIG)
    expected_obj = _objective_fingerprint(SAMPLE_OBJECTIVE_CONFIG)
    assert r.architecture_fingerprint == expected_arch
    assert r.objective_fingerprint == expected_obj


def test_build_version_record_artifact_paths_propagated():
    r = _build_record()
    assert r.artifact_paths == SAMPLE_ARTIFACT_PATHS


def test_build_version_record_training_config_snapshot():
    r = _build_record()
    assert r.training_config_snapshot == SAMPLE_TRAINING_CONFIG


def test_build_version_record_no_notes_or_rollback():
    r = _build_record()
    assert r.promotion_notes is None
    assert r.rollback_reason is None


# ---------------------------------------------------------------------------
# as_dict() schema
# ---------------------------------------------------------------------------

def test_as_dict_has_required_keys():
    r = _build_record()
    d = r.as_dict()
    for key in REQUIRED_VERSION_RECORD_KEYS:
        assert key in d, f"Missing required key: {key!r}"


def test_as_dict_schema_field():
    r = _build_record()
    assert r.as_dict()["schema"] == MODEL_VERSION_SCHEMA


def test_as_dict_artifact_paths_complete():
    r = _build_record()
    paths = r.as_dict()["artifact_paths"]
    for key in REQUIRED_ARTIFACT_PATH_KEYS:
        assert key in paths, f"Missing artifact path key: {key!r}"


def test_as_dict_promotion_notes_absent_when_none():
    r = _build_record()
    assert "promotion_notes" not in r.as_dict()


def test_as_dict_rollback_reason_absent_when_none():
    r = _build_record()
    assert "rollback_reason" not in r.as_dict()


def test_as_dict_json_serializable():
    r = _build_record()
    json.dumps(r.as_dict())  # must not raise


# ---------------------------------------------------------------------------
# with_promotion() state machine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("next_state", ["pending", "validated", "promoted", "rolled_back"])
def test_with_promotion_valid_states(next_state):
    r = _build_record()
    r2 = r.with_promotion(next_state)
    assert r2.promotion_state == next_state


def test_with_promotion_invalid_state_raises():
    r = _build_record()
    with pytest.raises(ValueError, match="promotion_state"):
        r.with_promotion("INVALID")


def test_with_promotion_notes_propagated():
    r = _build_record()
    r2 = r.with_promotion("validated", notes="loss=0.12 within bounds")
    assert r2.promotion_notes == "loss=0.12 within bounds"


def test_with_promotion_rollback_notes_become_rollback_reason():
    r = _build_record()
    r2 = r.with_promotion("rolled_back", notes="regression on paper")
    assert r2.rollback_reason == "regression on paper"


def test_with_promotion_preserves_other_fields():
    r = _build_record()
    r2 = r.with_promotion("validated")
    assert r2.version == r.version
    assert r2.model_id == r.model_id
    assert r2.artifact_paths == r.artifact_paths


def test_promotion_lifecycle():
    """Full promotion lifecycle: pending -> validated -> promoted -> rolled_back."""
    r = _build_record()
    assert r.promotion_state == "pending"
    r = r.with_promotion("validated", notes="smoke passed")
    assert r.promotion_state == "validated"
    r = r.with_promotion("promoted", notes="approved for paper")
    assert r.promotion_state == "promoted"
    r = r.with_promotion("rolled_back", notes="paper regression detected")
    assert r.promotion_state == "rolled_back"
    assert r.rollback_reason == "paper regression detected"


# ---------------------------------------------------------------------------
# validate_version_record()
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_record():
    r = _build_record()
    errors = validate_version_record(r.as_dict())
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_detects_missing_required_key():
    d = _build_record().as_dict()
    del d["version"]
    errors = validate_version_record(d)
    assert any("version" in e for e in errors)


def test_validate_detects_wrong_schema():
    d = _build_record().as_dict()
    d["schema"] = "wrong.schema"
    errors = validate_version_record(d)
    assert any("schema" in e for e in errors)


def test_validate_detects_invalid_promotion_state():
    d = _build_record().as_dict()
    d["promotion_state"] = "UNKNOWN"
    errors = validate_version_record(d)
    assert any("promotion_state" in e for e in errors)


def test_validate_detects_missing_artifact_path_key():
    d = _build_record().as_dict()
    del d["artifact_paths"]["weights_path"]
    errors = validate_version_record(d)
    assert any("weights_path" in e for e in errors)


def test_validate_empty_dict_reports_all_required_keys():
    errors = validate_version_record({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_VERSION_RECORD_KEYS:
        assert key in missing, f"Expected {key!r} reported as missing"


# ---------------------------------------------------------------------------
# ModelVersionRegistry: JSONL round-trip
# ---------------------------------------------------------------------------

def test_registry_append_and_load_all(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    r = _build_record(version="1.0.0")
    registry.append(r)
    loaded = registry.load_all()
    assert len(loaded) == 1
    assert loaded[0].version == "1.0.0"
    assert loaded[0].promotion_state == "pending"


def test_registry_append_multiple(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    for v in ("1.0.0", "1.1.0", "1.2.0"):
        registry.append(_build_record(version=v))
    loaded = registry.load_all()
    assert len(loaded) == 3
    assert [r.version for r in loaded] == ["1.0.0", "1.1.0", "1.2.0"]


def test_registry_latest(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    for v in ("1.0.0", "1.1.0"):
        registry.append(_build_record(version=v))
    assert registry.latest().version == "1.1.0"


def test_registry_latest_empty(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    assert registry.latest() is None


def test_registry_latest_promoted(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    r1 = _build_record(version="1.0.0")
    r2 = _build_record(version="1.1.0")
    r2 = r2.with_promotion("promoted")
    registry.append(r1)
    registry.append(r2)
    latest = registry.latest_promoted()
    assert latest is not None
    assert latest.version == "1.1.0"


def test_registry_latest_promoted_none_when_no_promoted(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    registry.append(_build_record(version="1.0.0"))
    assert registry.latest_promoted() is None


def test_registry_all_by_state(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    r1 = _build_record(version="1.0.0")
    r2 = _build_record(version="1.1.0").with_promotion("validated")
    r3 = _build_record(version="1.2.0").with_promotion("promoted")
    for r in (r1, r2, r3):
        registry.append(r)
    assert [r.version for r in registry.all_by_state("pending")] == ["1.0.0"]
    assert [r.version for r in registry.all_by_state("validated")] == ["1.1.0"]
    assert [r.version for r in registry.all_by_state("promoted")] == ["1.2.0"]


def test_registry_creates_parent_directory(tmp_path):
    reg_path = tmp_path / "hrm" / "registry" / "model_version_registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    registry.append(_build_record())
    assert reg_path.exists()


def test_registry_round_trip_fidelity(tmp_path):
    """Fields survive JSONL round-trip without mutation."""
    reg_path = tmp_path / "registry.jsonl"
    registry = ModelVersionRegistry(reg_path)
    r = _build_record(version="2.3.1")
    r = r.with_promotion("validated", notes="smoke ok")
    registry.append(r)
    loaded = registry.load_all()[0]
    assert loaded.version == r.version
    assert loaded.model_id == r.model_id
    assert loaded.architecture_fingerprint == r.architecture_fingerprint
    assert loaded.objective_fingerprint == r.objective_fingerprint
    assert loaded.artifact_paths == r.artifact_paths
    assert loaded.training_config_snapshot == r.training_config_snapshot
    assert loaded.promotion_state == "validated"
    assert loaded.promotion_notes == "smoke ok"


def test_registry_load_empty_file(tmp_path):
    reg_path = tmp_path / "registry.jsonl"
    reg_path.touch()
    registry = ModelVersionRegistry(reg_path)
    assert registry.load_all() == []


# ---------------------------------------------------------------------------
# VALID_PROMOTION_STATES
# ---------------------------------------------------------------------------

def test_valid_promotion_states_contains_lifecycle():
    assert "pending" in VALID_PROMOTION_STATES
    assert "validated" in VALID_PROMOTION_STATES
    assert "promoted" in VALID_PROMOTION_STATES
    assert "rolled_back" in VALID_PROMOTION_STATES
