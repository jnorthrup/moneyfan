"""
Tests for HRM rollback and recovery controller (Phase 5 Priority 3).

Validates:
  - artifact_health_check: ok/missing/empty per path, overall healthy flag
  - RollbackOutcome: as_dict() schema completeness and JSON-serializability
  - validate_rollback_outcome(): required key and enum checks
  - RollbackController.check_and_apply:
      - rollback fired when gate detects regression on promoted model
      - no-op when gate passes (model survives)
      - idempotent on already-rolled_back record
      - no-op / safe on non-promoted state
      - registry appended with rolled_back record on rollback
      - rollback log appended on rollback
  - RollbackController.force_rollback:
      - always rolls back regardless of metrics
      - idempotent on already-rolled_back record
      - registry appended with rolled_back record
      - rollback log appended
  - load_rollback_log: round-trip, empty for missing file
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from execution.model_version_registry import (
    ModelVersionRecord,
    ModelVersionRegistry,
    build_version_record,
)
from execution.promotion_gate import (
    EvaluationMetrics,
    PromotionGate,
    PromotionPolicy,
)
from execution.rollback_controller import (
    ROLLBACK_OUTCOME_SCHEMA,
    REQUIRED_ROLLBACK_OUTCOME_KEYS,
    VALID_TRIGGER_TYPES,
    ArtifactHealthReport,
    RollbackController,
    RollbackOutcome,
    artifact_health_check,
    validate_rollback_outcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-03-10T03:39:00+00:00"

_ARTIFACT_PATHS = {
    "weights_path": "/hrm/checkpoints/w.npz",
    "config_path": "/hrm/checkpoints/cfg.json",
    "feature_schema_path": "/hrm/checkpoints/schema.json",
    "objective_config_path": "/hrm/checkpoints/obj.json",
}
_MODEL_CFG = {"hidden_dim": 256, "n_heads": 8, "regime_attn_layers": 4,
              "tactical_attn_layers": 2, "input_dim": 48}
_OBJ_CFG = {"world_model_weight": 1.0, "trade_head_weight": 0.5,
            "cost_turnover_weight": 0.1, "regime_weight_scale": 1.0}


def _record(state: str = "promoted") -> ModelVersionRecord:
    r = build_version_record(
        version="1.0.0",
        artifact_paths=_ARTIFACT_PATHS,
        model_config_dict=_MODEL_CFG,
        objective_config=_OBJ_CFG,
        training_config_snapshot={"optimizer": "adamw"},
        created_at=FIXED_TS,
    )
    if state != "pending":
        r = r.with_promotion(state)
    return r


def _good_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(hit_rate=0.65, mean_loss=0.20, profit_factor=1.6,
                             episodes_evaluated=5, max_drawdown_pct=0.05)


def _bad_metrics() -> EvaluationMetrics:
    """Metrics that will trigger rollback on a promoted model (drawdown > rollback threshold)."""
    return EvaluationMetrics(hit_rate=0.65, mean_loss=0.20, profit_factor=1.6,
                             episodes_evaluated=5, max_drawdown_pct=0.25)


def _gate() -> PromotionGate:
    return PromotionGate(PromotionPolicy())


def _registry(tmp_path: Path) -> ModelVersionRegistry:
    return ModelVersionRegistry(tmp_path / "model_version_registry.jsonl")


def _controller(tmp_path: Path) -> RollbackController:
    return RollbackController(rollback_log_path=tmp_path / "rollback.jsonl")


# ---------------------------------------------------------------------------
# artifact_health_check
# ---------------------------------------------------------------------------

def test_health_check_all_ok(tmp_path):
    paths = {}
    for key in ("weights_path", "config_path", "feature_schema_path", "objective_config_path"):
        p = tmp_path / f"{key}.bin"
        p.write_bytes(b"data")
        paths[key] = str(p)
    report = artifact_health_check(paths, ts_utc=FIXED_TS)
    assert report.healthy is True
    assert all(s == "ok" for s in report.path_statuses.values())


def test_health_check_missing_file(tmp_path):
    paths = {"weights_path": str(tmp_path / "nonexistent.npz")}
    report = artifact_health_check(paths, ts_utc=FIXED_TS)
    assert report.healthy is False
    assert report.path_statuses["weights_path"] == "missing"


def test_health_check_empty_file(tmp_path):
    p = tmp_path / "empty.npz"
    p.touch()
    paths = {"weights_path": str(p)}
    report = artifact_health_check(paths, ts_utc=FIXED_TS)
    assert report.healthy is False
    assert report.path_statuses["weights_path"] == "empty"


def test_health_check_mixed(tmp_path):
    good = tmp_path / "good.npz"
    good.write_bytes(b"weight data")
    paths = {"weights_path": str(good), "config_path": str(tmp_path / "missing.json")}
    report = artifact_health_check(paths, ts_utc=FIXED_TS)
    assert report.healthy is False
    assert report.path_statuses["weights_path"] == "ok"
    assert report.path_statuses["config_path"] == "missing"


def test_health_check_as_dict():
    report = ArtifactHealthReport(
        healthy=True, path_statuses={"weights_path": "ok"}, checked_at=FIXED_TS
    )
    d = report.as_dict()
    assert d["healthy"] is True
    assert "path_statuses" in d
    assert "checked_at" in d


# ---------------------------------------------------------------------------
# RollbackOutcome
# ---------------------------------------------------------------------------

def test_rollback_outcome_as_dict_has_required_keys():
    outcome = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="high drawdown", ts_utc=FIXED_TS,
    )
    d = outcome.as_dict()
    for key in REQUIRED_ROLLBACK_OUTCOME_KEYS:
        assert key in d, f"Missing required key: {key!r}"


def test_rollback_outcome_schema_field():
    outcome = RollbackOutcome(
        triggered=True, trigger_type="manual",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="manual", ts_utc=FIXED_TS,
    )
    assert outcome.as_dict()["schema"] == ROLLBACK_OUTCOME_SCHEMA


def test_rollback_outcome_json_serializable():
    outcome = RollbackOutcome(
        triggered=False, trigger_type="no_op",
        previous_state="promoted", next_state="promoted",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="gate passed", ts_utc=FIXED_TS,
    )
    json.dumps(outcome.as_dict())  # must not raise


def test_rollback_outcome_gate_failures_absent_when_none():
    outcome = RollbackOutcome(
        triggered=False, trigger_type="no_op",
        previous_state="promoted", next_state="promoted",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="ok", ts_utc=FIXED_TS,
        gate_failures=None,
    )
    assert "gate_failures" not in outcome.as_dict()


def test_rollback_outcome_gate_failures_present_when_set():
    outcome = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="drawdown", ts_utc=FIXED_TS,
        gate_failures=["max_drawdown 0.25 > rollback threshold 0.20"],
    )
    assert outcome.as_dict()["gate_failures"] == ["max_drawdown 0.25 > rollback threshold 0.20"]


# ---------------------------------------------------------------------------
# validate_rollback_outcome
# ---------------------------------------------------------------------------

def test_validate_rollback_accepts_valid():
    outcome = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="regression", ts_utc=FIXED_TS,
    )
    errors = validate_rollback_outcome(outcome.as_dict())
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_rollback_detects_missing_key():
    d = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="r", ts_utc=FIXED_TS,
    ).as_dict()
    del d["triggered"]
    errors = validate_rollback_outcome(d)
    assert any("triggered" in e for e in errors)


def test_validate_rollback_detects_wrong_schema():
    d = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="r", ts_utc=FIXED_TS,
    ).as_dict()
    d["schema"] = "wrong"
    errors = validate_rollback_outcome(d)
    assert any("schema" in e for e in errors)


def test_validate_rollback_detects_invalid_trigger_type():
    d = RollbackOutcome(
        triggered=True, trigger_type="automatic",
        previous_state="promoted", next_state="rolled_back",
        model_id="hrm-1.0.0-abc", model_version="1.0.0",
        reason="r", ts_utc=FIXED_TS,
    ).as_dict()
    d["trigger_type"] = "INVALID"
    errors = validate_rollback_outcome(d)
    assert any("trigger_type" in e for e in errors)


def test_validate_rollback_empty_dict():
    errors = validate_rollback_outcome({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_ROLLBACK_OUTCOME_KEYS:
        assert key in missing, f"Expected {key!r} reported as missing"


# ---------------------------------------------------------------------------
# RollbackController.check_and_apply
# ---------------------------------------------------------------------------

def test_check_and_apply_fires_rollback_on_regression(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    outcome = controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    assert outcome.triggered is True
    assert outcome.trigger_type == "automatic"
    assert outcome.next_state == "rolled_back"


def test_check_and_apply_appends_rolled_back_to_registry(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    reg.append(r)  # record the promoted version first
    controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    all_records = reg.load_all()
    rolled = [rec for rec in all_records if rec.promotion_state == "rolled_back"]
    assert len(rolled) == 1


def test_check_and_apply_no_rollback_when_gate_passes(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    outcome = controller.check_and_apply(r, _good_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    assert outcome.triggered is False
    assert outcome.trigger_type == "no_op"
    assert outcome.next_state == "promoted"


def test_check_and_apply_no_op_for_already_rolled_back(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("rolled_back")
    outcome = controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    assert outcome.triggered is False
    assert outcome.trigger_type == "no_op"
    assert outcome.next_state == "rolled_back"


def test_check_and_apply_no_op_for_pending_state(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("pending")
    outcome = controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    assert outcome.triggered is False
    assert outcome.trigger_type == "no_op"


def test_check_and_apply_appends_rollback_log(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    log = controller.load_rollback_log()
    assert len(log) == 1
    assert log[0]["schema"] == ROLLBACK_OUTCOME_SCHEMA
    assert log[0]["trigger_type"] == "automatic"


def test_check_and_apply_no_rollback_log_entry_when_gate_passes(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    controller.check_and_apply(r, _good_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    assert controller.load_rollback_log() == []


def test_check_and_apply_outcome_json_serializable(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    outcome = controller.check_and_apply(r, _bad_metrics(), reg, _gate(), ts_utc=FIXED_TS)
    json.dumps(outcome.as_dict())  # must not raise


# ---------------------------------------------------------------------------
# RollbackController.force_rollback
# ---------------------------------------------------------------------------

def test_force_rollback_rolls_back_any_state(tmp_path):
    for state in ("promoted", "validated", "pending"):
        reg = _registry(tmp_path / state)
        controller = _controller(tmp_path / state)
        r = _record(state)
        outcome = controller.force_rollback(r, reg, reason="operator override",
                                            ts_utc=FIXED_TS)
        assert outcome.triggered is True
        assert outcome.trigger_type == "manual"
        assert outcome.next_state == "rolled_back"


def test_force_rollback_appends_to_registry(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    reg.append(r)
    controller.force_rollback(r, reg, reason="security incident", ts_utc=FIXED_TS)
    all_records = reg.load_all()
    rolled = [rec for rec in all_records if rec.promotion_state == "rolled_back"]
    assert len(rolled) == 1
    assert rolled[0].rollback_reason == "security incident"


def test_force_rollback_idempotent_on_already_rolled_back(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("rolled_back")
    outcome = controller.force_rollback(r, reg, reason="again", ts_utc=FIXED_TS)
    assert outcome.triggered is False
    assert outcome.trigger_type == "no_op"
    assert reg.load_all() == []  # nothing appended


def test_force_rollback_appends_rollback_log(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    controller.force_rollback(r, reg, reason="deliberate", ts_utc=FIXED_TS)
    log = controller.load_rollback_log()
    assert len(log) == 1
    assert log[0]["trigger_type"] == "manual"
    assert "deliberate" in log[0]["reason"]


def test_force_rollback_reason_propagated(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    outcome = controller.force_rollback(r, reg, reason="paper regression", ts_utc=FIXED_TS)
    assert outcome.reason == "paper regression"


# ---------------------------------------------------------------------------
# load_rollback_log
# ---------------------------------------------------------------------------

def test_load_rollback_log_empty_when_no_file(tmp_path):
    controller = RollbackController(rollback_log_path=tmp_path / "missing.jsonl")
    assert controller.load_rollback_log() == []


def test_load_rollback_log_round_trip(tmp_path):
    reg = _registry(tmp_path)
    controller = _controller(tmp_path)
    r = _record("promoted")
    controller.force_rollback(r, reg, reason="r1", ts_utc=FIXED_TS)
    r2 = _record("validated")
    controller.force_rollback(r2, reg, reason="r2", ts_utc=FIXED_TS)
    log = controller.load_rollback_log()
    assert len(log) == 2
    assert log[0]["reason"] == "r1"
    assert log[1]["reason"] == "r2"


# ---------------------------------------------------------------------------
# VALID_TRIGGER_TYPES
# ---------------------------------------------------------------------------

def test_valid_trigger_types_contains_expected():
    assert "automatic" in VALID_TRIGGER_TYPES
    assert "manual" in VALID_TRIGGER_TYPES
    assert "no_op" in VALID_TRIGGER_TYPES
