"""
Tests for HRM model governance and promotion gates (Phase 5 Priority 2).

Validates:
  - PromotionPolicy defaults and as_dict()
  - EvaluationMetrics construction and as_dict()
  - PromotionGate: all passing criteria -> correct decision and next_state
  - PromotionGate: each failing criterion detected individually
  - State machine transitions: pending->validated->promoted->rolled_back
  - Rollback: promoted model with excess drawdown triggers rolled_back
  - Hold: failing criteria keeps state unchanged
  - multi_slice_validate: all slices pass -> gate opens
  - multi_slice_validate: any failing slice -> gate stays closed
  - multi_slice_validate: insufficient slices -> failure
  - Audit log: appended on evaluate(), valid schema, JSON-serializable
  - validate_audit_entry(): required keys, schema check, decision enum
  - load_audit_log(): round-trip fidelity
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any

from execution.model_version_registry import (
    ModelVersionRecord,
    build_version_record,
)
from execution.promotion_gate import (
    PROMOTION_AUDIT_SCHEMA,
    REQUIRED_AUDIT_KEYS,
    VALID_DECISIONS,
    EvaluationMetrics,
    PromotionGate,
    PromotionPolicy,
    load_audit_log,
    validate_audit_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-03-10T03:00:00+00:00"

_ARTIFACT_PATHS = {
    "weights_path": "/hrm/checkpoints/w.npz",
    "config_path": "/hrm/checkpoints/cfg.json",
    "feature_schema_path": "/hrm/checkpoints/schema.json",
    "objective_config_path": "/hrm/checkpoints/obj.json",
}
_MODEL_CONFIG = {"hidden_dim": 256, "n_heads": 8, "regime_attn_layers": 4,
                 "tactical_attn_layers": 2, "input_dim": 48}
_OBJECTIVE_CONFIG = {"world_model_weight": 1.0, "trade_head_weight": 0.5,
                     "cost_turnover_weight": 0.1, "regime_weight_scale": 1.0}


def _record(state: str = "pending") -> ModelVersionRecord:
    r = build_version_record(
        version="1.0.0",
        artifact_paths=_ARTIFACT_PATHS,
        model_config_dict=_MODEL_CONFIG,
        objective_config=_OBJECTIVE_CONFIG,
        training_config_snapshot={"optimizer": "adamw"},
        created_at=FIXED_TS,
    )
    if state != "pending":
        r = r.with_promotion(state)
    return r


def _good_metrics(episodes: int = 5) -> EvaluationMetrics:
    """Metrics that pass the default PromotionPolicy."""
    return EvaluationMetrics(
        hit_rate=0.60,
        mean_loss=0.25,
        profit_factor=1.5,
        episodes_evaluated=episodes,
        max_drawdown_pct=0.05,
    )


def _default_gate(audit_path=None) -> PromotionGate:
    return PromotionGate(PromotionPolicy(), audit_log_path=audit_path)


# ---------------------------------------------------------------------------
# PromotionPolicy
# ---------------------------------------------------------------------------

def test_policy_defaults():
    p = PromotionPolicy()
    assert p.min_hit_rate == 0.50
    assert p.max_mean_loss == 0.5
    assert p.min_profit_factor == 1.1
    assert p.min_episodes_evaluated == 3
    assert p.max_drawdown_pct == 0.15
    assert p.rollback_drawdown_pct == 0.20


def test_policy_as_dict_has_all_fields():
    p = PromotionPolicy()
    d = p.as_dict()
    for k in ("min_hit_rate", "max_mean_loss", "min_profit_factor",
              "min_episodes_evaluated", "max_drawdown_pct", "rollback_drawdown_pct"):
        assert k in d


# ---------------------------------------------------------------------------
# EvaluationMetrics
# ---------------------------------------------------------------------------

def test_metrics_as_dict():
    m = _good_metrics()
    d = m.as_dict()
    assert d["hit_rate"] == m.hit_rate
    assert d["mean_loss"] == m.mean_loss
    assert d["profit_factor"] == m.profit_factor
    assert d["episodes_evaluated"] == m.episodes_evaluated
    assert d["max_drawdown_pct"] == m.max_drawdown_pct


# ---------------------------------------------------------------------------
# PromotionGate: passing all criteria
# ---------------------------------------------------------------------------

def test_gate_passes_with_good_metrics():
    gate = _default_gate()
    r = _record("pending")
    decision = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert decision.passed is True
    assert decision.failures == []
    assert decision.next_state == "validated"
    assert decision.decision == "promote_to_validated"


def test_gate_pending_to_validated_transition():
    gate = _default_gate()
    r = _record("pending")
    d = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert d.previous_state == "pending"
    assert d.next_state == "validated"


def test_gate_validated_to_promoted_transition():
    gate = _default_gate()
    r = _record("validated")
    d = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert d.next_state == "promoted"
    assert d.decision == "promote_to_promoted"


def test_gate_promoted_stays_promoted_when_passing():
    gate = _default_gate()
    r = _record("promoted")
    d = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert d.next_state == "promoted"
    assert d.passed is True


# ---------------------------------------------------------------------------
# PromotionGate: failing each criterion
# ---------------------------------------------------------------------------

def test_gate_fails_low_hit_rate():
    gate = _default_gate()
    m = EvaluationMetrics(hit_rate=0.40, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.05)
    d = gate.evaluate(_record(), m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert any("hit_rate" in f for f in d.failures)


def test_gate_fails_high_mean_loss():
    gate = _default_gate()
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.9, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.05)
    d = gate.evaluate(_record(), m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert any("mean_loss" in f for f in d.failures)


def test_gate_fails_low_profit_factor():
    gate = _default_gate()
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.2, profit_factor=0.9,
                          episodes_evaluated=5, max_drawdown_pct=0.05)
    d = gate.evaluate(_record(), m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert any("profit_factor" in f for f in d.failures)


def test_gate_fails_insufficient_episodes():
    gate = _default_gate()
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=1, max_drawdown_pct=0.05)
    d = gate.evaluate(_record(), m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert any("episodes_evaluated" in f for f in d.failures)


def test_gate_fails_excess_drawdown_pending():
    gate = _default_gate()
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.18)
    d = gate.evaluate(_record("pending"), m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert any("max_drawdown" in f for f in d.failures)


# ---------------------------------------------------------------------------
# Rollback: promoted model with excess drawdown
# ---------------------------------------------------------------------------

def test_gate_rolls_back_promoted_model_on_excess_drawdown():
    gate = _default_gate()
    r = _record("promoted")
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.25)  # > rollback 0.20
    d = gate.evaluate(r, m, ts_utc=FIXED_TS)
    assert d.passed is False
    assert d.next_state == "rolled_back"
    assert d.decision == "rollback"


def test_gate_promoted_survives_drawdown_below_rollback_threshold():
    gate = _default_gate()
    r = _record("promoted")
    m = EvaluationMetrics(hit_rate=0.6, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.19)  # < rollback 0.20
    d = gate.evaluate(r, m, ts_utc=FIXED_TS)
    assert d.passed is True
    assert d.next_state == "promoted"


# ---------------------------------------------------------------------------
# Hold: failing criteria keeps state
# ---------------------------------------------------------------------------

def test_gate_hold_pending_on_failure():
    gate = _default_gate()
    r = _record("pending")
    m = EvaluationMetrics(hit_rate=0.3, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.05)
    d = gate.evaluate(r, m, ts_utc=FIXED_TS)
    assert d.next_state == "pending"
    assert d.decision == "hold_pending"


def test_gate_hold_validated_on_failure():
    gate = _default_gate()
    r = _record("validated")
    m = EvaluationMetrics(hit_rate=0.3, mean_loss=0.2, profit_factor=1.5,
                          episodes_evaluated=5, max_drawdown_pct=0.05)
    d = gate.evaluate(r, m, ts_utc=FIXED_TS)
    assert d.next_state == "validated"
    assert d.decision == "hold_validated"


# ---------------------------------------------------------------------------
# multi_slice_validate
# ---------------------------------------------------------------------------

def test_multi_slice_all_pass():
    gate = _default_gate()
    r = _record("pending")
    slices = [_good_metrics(5) for _ in range(3)]
    d = gate.multi_slice_validate(r, slices)
    assert d.passed is True
    assert d.next_state == "validated"


def test_multi_slice_one_failing_blocks_promotion():
    gate = _default_gate()
    r = _record("pending")
    bad = EvaluationMetrics(hit_rate=0.3, mean_loss=0.2, profit_factor=1.5,
                            episodes_evaluated=5, max_drawdown_pct=0.05)
    slices = [_good_metrics(5), _good_metrics(5), bad]
    d = gate.multi_slice_validate(r, slices)
    assert d.passed is False
    assert d.next_state == "pending"
    # The failure must be tagged with its slice index
    assert any("slice[2]" in f for f in d.failures)


def test_multi_slice_insufficient_slices():
    gate = PromotionGate(PromotionPolicy(min_episodes_evaluated=3))
    r = _record("pending")
    slices = [_good_metrics(5)]  # only 1, policy requires 3
    d = gate.multi_slice_validate(r, slices)
    assert d.passed is False
    assert any("multi_slice" in f for f in d.failures)


def test_multi_slice_empty_slices():
    gate = _default_gate()
    r = _record("pending")
    d = gate.multi_slice_validate(r, [])
    assert d.passed is False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_gate_appends_audit_on_evaluate(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    r = _record("pending")
    gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    entries = load_audit_log(audit_path)
    assert len(entries) == 1
    assert entries[0]["model_id"] == r.model_id
    assert entries[0]["schema"] == PROMOTION_AUDIT_SCHEMA


def test_gate_multiple_evaluations_appended(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    r = _record("pending")
    gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert len(load_audit_log(audit_path)) == 2


def test_gate_no_audit_when_path_is_none():
    gate = PromotionGate(PromotionPolicy(), audit_log_path=None)
    r = _record("pending")
    d = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    assert d.passed is True  # just check no crash


def test_audit_entry_json_serializable(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    r = _record("pending")
    gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    raw = audit_path.read_text().strip()
    entry = json.loads(raw)  # must not raise
    assert entry["schema"] == PROMOTION_AUDIT_SCHEMA


def test_audit_entry_has_required_keys(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    gate.evaluate(_record(), _good_metrics(), ts_utc=FIXED_TS)
    entry = load_audit_log(audit_path)[0]
    for key in REQUIRED_AUDIT_KEYS:
        assert key in entry, f"Missing audit key: {key!r}"


def test_audit_entry_criteria_and_metrics_snapshots(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    gate.evaluate(_record(), _good_metrics(), ts_utc=FIXED_TS)
    entry = load_audit_log(audit_path)[0]
    assert "criteria_snapshot" in entry
    assert "metrics_snapshot" in entry
    assert "min_hit_rate" in entry["criteria_snapshot"]
    assert "hit_rate" in entry["metrics_snapshot"]


# ---------------------------------------------------------------------------
# validate_audit_entry()
# ---------------------------------------------------------------------------

def _build_audit_entry() -> dict[str, Any]:
    gate = PromotionGate(PromotionPolicy())
    r = _record("pending")
    d = gate.evaluate(r, _good_metrics(), ts_utc=FIXED_TS)
    return d.as_audit_dict(r.model_id, r.version, ts_utc=FIXED_TS)


def test_validate_audit_accepts_valid_entry():
    entry = _build_audit_entry()
    errors = validate_audit_entry(entry)
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_audit_detects_missing_required_key():
    entry = _build_audit_entry()
    del entry["decision"]
    errors = validate_audit_entry(entry)
    assert any("decision" in e for e in errors)


def test_validate_audit_detects_wrong_schema():
    entry = _build_audit_entry()
    entry["schema"] = "wrong.schema"
    errors = validate_audit_entry(entry)
    assert any("schema" in e for e in errors)


def test_validate_audit_detects_invalid_decision():
    entry = _build_audit_entry()
    entry["decision"] = "INVALID_DECISION"
    errors = validate_audit_entry(entry)
    assert any("decision" in e for e in errors)


def test_validate_audit_empty_dict_reports_all_keys():
    errors = validate_audit_entry({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_AUDIT_KEYS:
        assert key in missing, f"Expected {key!r} reported as missing"


# ---------------------------------------------------------------------------
# load_audit_log round-trip
# ---------------------------------------------------------------------------

def test_load_audit_log_returns_empty_for_missing_file(tmp_path):
    assert load_audit_log(tmp_path / "nonexistent.jsonl") == []


def test_load_audit_log_round_trip(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    gate = PromotionGate(PromotionPolicy(), audit_log_path=audit_path)
    gate.evaluate(_record("pending"), _good_metrics(), ts_utc=FIXED_TS)
    gate.evaluate(_record("validated"), _good_metrics(), ts_utc=FIXED_TS)
    entries = load_audit_log(audit_path)
    assert len(entries) == 2
    assert entries[0]["previous_state"] == "pending"
    assert entries[1]["previous_state"] == "validated"


# ---------------------------------------------------------------------------
# VALID_DECISIONS completeness
# ---------------------------------------------------------------------------

def test_valid_decisions_contains_expected():
    assert "promote_to_validated" in VALID_DECISIONS
    assert "promote_to_promoted" in VALID_DECISIONS
    assert "rollback" in VALID_DECISIONS
    assert "hold_pending" in VALID_DECISIONS
    assert "hold_validated" in VALID_DECISIONS
