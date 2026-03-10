"""
Tests for HRM model observability and monitoring (Phase 5 Priority 4).

Validates:
  - build_performance_report: state_distribution, rollback_rate, version_series,
    latest_promoted, empty registry
  - ModelPerformanceReport.as_dict(): required keys, schema, JSON-serializable
  - validate_report(): missing keys, wrong schema
  - AlertThresholds defaults and as_dict()
  - DegradationAlerter.evaluate(): hit_rate/mean_loss/drawdown warning and critical
  - DegradationAlerter: no alerts on healthy metrics
  - DegradationAlerter: multiple alerts fired in one evaluation
  - Alert log: appended, round-trip, no log when path=None
  - validate_alert_event(): required keys, schema, alert_type enum
  - DecisionLogIndex.load_all(): merged + sorted chronologically
  - DecisionLogIndex.for_model(): filters by model_id
  - DecisionLogIndex.for_version(): filters by version
  - DecisionLogIndex: empty when no files present
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from execution.model_version_registry import (
    ModelVersionRegistry,
    build_version_record,
)
from execution.promotion_gate import (
    EvaluationMetrics,
    PromotionGate,
    PromotionPolicy,
)
from execution.model_observability import (
    ALERT_SCHEMA,
    REPORT_SCHEMA,
    REQUIRED_ALERT_KEYS,
    REQUIRED_REPORT_KEYS,
    VALID_ALERT_TYPES,
    AlertEvent,
    AlertThresholds,
    DecisionLogIndex,
    DegradationAlerter,
    ModelPerformanceReport,
    build_performance_report,
    validate_alert_event,
    validate_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-03-10T08:15:00+00:00"

_ARTIFACT_PATHS = {
    "weights_path": "/hrm/w.npz",
    "config_path": "/hrm/cfg.json",
    "feature_schema_path": "/hrm/schema.json",
    "objective_config_path": "/hrm/obj.json",
}
_MODEL_CFG = {"hidden_dim": 256, "n_heads": 8, "regime_attn_layers": 4,
              "tactical_attn_layers": 2, "input_dim": 48}
_OBJ_CFG = {"world_model_weight": 1.0, "trade_head_weight": 0.5,
            "cost_turnover_weight": 0.1, "regime_weight_scale": 1.0}


def _record(version="1.0.0", state="pending"):
    r = build_version_record(
        version=version,
        artifact_paths=_ARTIFACT_PATHS,
        model_config_dict=_MODEL_CFG,
        objective_config=_OBJ_CFG,
        training_config_snapshot={"optimizer": "adamw"},
        created_at=FIXED_TS,
    )
    if state != "pending":
        r = r.with_promotion(state)
    return r


def _registry(tmp_path, records):
    reg = ModelVersionRegistry(tmp_path / "registry.jsonl")
    for r in records:
        reg.append(r)
    return reg


def _metrics(hit_rate=0.65, mean_loss=0.20, max_drawdown_pct=0.05, episodes=5):
    return EvaluationMetrics(hit_rate=hit_rate, mean_loss=mean_loss,
                             profit_factor=1.5, episodes_evaluated=episodes,
                             max_drawdown_pct=max_drawdown_pct)


# ---------------------------------------------------------------------------
# build_performance_report
# ---------------------------------------------------------------------------

def test_performance_report_empty_registry(tmp_path):
    reg = _registry(tmp_path, [])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.total_records == 0
    assert report.state_distribution == {}
    assert report.version_series == []
    assert report.latest_promoted is None
    assert report.rollback_rate == 0.0


def test_performance_report_state_distribution(tmp_path):
    records = [
        _record("1.0.0", "pending"),
        _record("1.1.0", "validated"),
        _record("1.2.0", "promoted"),
        _record("1.3.0", "rolled_back"),
    ]
    reg = _registry(tmp_path, records)
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.state_distribution["pending"] == 1
    assert report.state_distribution["validated"] == 1
    assert report.state_distribution["promoted"] == 1
    assert report.state_distribution["rolled_back"] == 1


def test_performance_report_total_records(tmp_path):
    records = [_record(str(i) + ".0.0", "pending") for i in range(5)]
    reg = _registry(tmp_path, records)
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.total_records == 5


def test_performance_report_version_series_ordered(tmp_path):
    records = [_record("1.0.0", "pending"), _record("1.1.0", "promoted")]
    reg = _registry(tmp_path, records)
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert len(report.version_series) == 2
    assert report.version_series[0]["version"] == "1.0.0"
    assert report.version_series[1]["version"] == "1.1.0"


def test_performance_report_latest_promoted(tmp_path):
    records = [_record("1.0.0", "pending"), _record("1.1.0", "promoted")]
    reg = _registry(tmp_path, records)
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.latest_promoted is not None
    assert report.latest_promoted["version"] == "1.1.0"


def test_performance_report_latest_promoted_none_when_none(tmp_path):
    reg = _registry(tmp_path, [_record("1.0.0", "pending")])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.latest_promoted is None


def test_performance_report_rollback_rate(tmp_path):
    records = [
        _record("1.0.0", "rolled_back"),
        _record("1.1.0", "rolled_back"),
        _record("1.2.0", "promoted"),
        _record("1.3.0", "pending"),
    ]
    reg = _registry(tmp_path, records)
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert abs(report.rollback_rate - 0.5) < 1e-9


def test_performance_report_rollback_rate_zero_when_none(tmp_path):
    reg = _registry(tmp_path, [_record("1.0.0", "promoted")])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.rollback_rate == 0.0


# ---------------------------------------------------------------------------
# ModelPerformanceReport.as_dict()
# ---------------------------------------------------------------------------

def test_report_as_dict_has_required_keys(tmp_path):
    reg = _registry(tmp_path, [_record()])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    d = report.as_dict()
    for key in REQUIRED_REPORT_KEYS:
        assert key in d, f"Missing required key: {key!r}"


def test_report_as_dict_schema_field(tmp_path):
    reg = _registry(tmp_path, [])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    assert report.as_dict()["schema"] == REPORT_SCHEMA


def test_report_as_dict_json_serializable(tmp_path):
    reg = _registry(tmp_path, [_record("1.0.0", "promoted")])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    json.dumps(report.as_dict())  # must not raise


# ---------------------------------------------------------------------------
# validate_report
# ---------------------------------------------------------------------------

def test_validate_report_accepts_valid(tmp_path):
    reg = _registry(tmp_path, [])
    report = build_performance_report(reg, ts_utc=FIXED_TS)
    errors = validate_report(report.as_dict())
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_report_detects_missing_key(tmp_path):
    reg = _registry(tmp_path, [])
    d = build_performance_report(reg, ts_utc=FIXED_TS).as_dict()
    del d["total_records"]
    errors = validate_report(d)
    assert any("total_records" in e for e in errors)


def test_validate_report_detects_wrong_schema(tmp_path):
    reg = _registry(tmp_path, [])
    d = build_performance_report(reg, ts_utc=FIXED_TS).as_dict()
    d["schema"] = "wrong"
    errors = validate_report(d)
    assert any("schema" in e for e in errors)


def test_validate_report_empty_dict():
    errors = validate_report({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_REPORT_KEYS:
        assert key in missing, f"Expected {key!r} in missing set"


# ---------------------------------------------------------------------------
# AlertThresholds
# ---------------------------------------------------------------------------

def test_alert_thresholds_defaults():
    t = AlertThresholds()
    assert t.hit_rate_warning == 0.48
    assert t.hit_rate_critical == 0.40
    assert t.mean_loss_warning == 0.45
    assert t.mean_loss_critical == 0.60
    assert t.max_drawdown_warning == 0.12
    assert t.max_drawdown_critical == 0.18


def test_alert_thresholds_as_dict_has_all_fields():
    d = AlertThresholds().as_dict()
    for k in ("hit_rate_warning", "hit_rate_critical", "mean_loss_warning",
              "mean_loss_critical", "max_drawdown_warning", "max_drawdown_critical"):
        assert k in d


# ---------------------------------------------------------------------------
# DegradationAlerter — no alert on healthy metrics
# ---------------------------------------------------------------------------

def test_no_alerts_on_healthy_metrics():
    alerter = DegradationAlerter()
    r = _record("1.0.0", "promoted")
    events = alerter.evaluate(r, _metrics(), ts_utc=FIXED_TS)
    assert events == []


# ---------------------------------------------------------------------------
# DegradationAlerter — hit_rate alerts
# ---------------------------------------------------------------------------

def test_alert_hit_rate_warning():
    alerter = DegradationAlerter()
    r = _record("1.0.0", "promoted")
    events = alerter.evaluate(r, _metrics(hit_rate=0.45), ts_utc=FIXED_TS)
    types = [e.alert_type for e in events]
    assert "degradation_warning" in types


def test_alert_hit_rate_critical():
    alerter = DegradationAlerter()
    r = _record("1.0.0", "promoted")
    events = alerter.evaluate(r, _metrics(hit_rate=0.35), ts_utc=FIXED_TS)
    types = [e.alert_type for e in events]
    assert "degradation_critical" in types
    # critical, not warning
    assert "degradation_warning" not in types


def test_no_hit_rate_alert_at_threshold():
    alerter = DegradationAlerter()
    r = _record("1.0.0", "promoted")
    events = alerter.evaluate(r, _metrics(hit_rate=0.50), ts_utc=FIXED_TS)
    assert not any(e.metric_name == "hit_rate" for e in events)


# ---------------------------------------------------------------------------
# DegradationAlerter — mean_loss alerts
# ---------------------------------------------------------------------------

def test_alert_mean_loss_warning():
    alerter = DegradationAlerter()
    r = _record()
    events = alerter.evaluate(r, _metrics(mean_loss=0.50), ts_utc=FIXED_TS)
    assert any(e.metric_name == "mean_loss" for e in events)
    assert any(e.alert_type == "loss_spike" for e in events)


def test_alert_mean_loss_critical():
    alerter = DegradationAlerter()
    r = _record()
    events = alerter.evaluate(r, _metrics(mean_loss=0.70), ts_utc=FIXED_TS)
    loss_events = [e for e in events if e.metric_name == "mean_loss"]
    assert loss_events[0].metric_value == 0.70
    assert loss_events[0].threshold == AlertThresholds().mean_loss_critical


# ---------------------------------------------------------------------------
# DegradationAlerter — drawdown alerts
# ---------------------------------------------------------------------------

def test_alert_drawdown_warning():
    alerter = DegradationAlerter()
    r = _record()
    events = alerter.evaluate(r, _metrics(max_drawdown_pct=0.14), ts_utc=FIXED_TS)
    assert any(e.alert_type == "drawdown_warning" for e in events)


def test_alert_drawdown_critical():
    alerter = DegradationAlerter()
    r = _record()
    events = alerter.evaluate(r, _metrics(max_drawdown_pct=0.22), ts_utc=FIXED_TS)
    assert any(e.alert_type == "drawdown_critical" for e in events)
    assert not any(e.alert_type == "drawdown_warning" for e in events)


# ---------------------------------------------------------------------------
# DegradationAlerter — multiple alerts in one evaluation
# ---------------------------------------------------------------------------

def test_multiple_alerts_fired_simultaneously():
    alerter = DegradationAlerter()
    r = _record()
    # All three metrics bad at once
    events = alerter.evaluate(
        r,
        _metrics(hit_rate=0.35, mean_loss=0.70, max_drawdown_pct=0.22),
        ts_utc=FIXED_TS,
    )
    metric_names = {e.metric_name for e in events}
    assert "hit_rate" in metric_names
    assert "mean_loss" in metric_names
    assert "max_drawdown_pct" in metric_names


# ---------------------------------------------------------------------------
# Alert log
# ---------------------------------------------------------------------------

def test_alert_log_appended(tmp_path):
    alert_path = tmp_path / "alerts.jsonl"
    alerter = DegradationAlerter(alert_log_path=alert_path)
    r = _record()
    alerter.evaluate(r, _metrics(hit_rate=0.35), ts_utc=FIXED_TS)
    entries = alerter.load_alert_log()
    assert len(entries) >= 1
    assert entries[0]["schema"] == ALERT_SCHEMA


def test_alert_log_not_written_when_no_alerts(tmp_path):
    alert_path = tmp_path / "alerts.jsonl"
    alerter = DegradationAlerter(alert_log_path=alert_path)
    alerter.evaluate(_record(), _metrics(), ts_utc=FIXED_TS)
    assert alerter.load_alert_log() == []


def test_alert_log_no_crash_when_path_none():
    alerter = DegradationAlerter(alert_log_path=None)
    events = alerter.evaluate(_record(), _metrics(hit_rate=0.35), ts_utc=FIXED_TS)
    assert len(events) >= 1  # events returned even without file


def test_alert_log_json_serializable(tmp_path):
    alert_path = tmp_path / "alerts.jsonl"
    alerter = DegradationAlerter(alert_log_path=alert_path)
    alerter.evaluate(_record(), _metrics(hit_rate=0.35), ts_utc=FIXED_TS)
    raw = alert_path.read_text().strip().split("\n")[0]
    json.loads(raw)  # must not raise


def test_alert_log_empty_for_missing_file(tmp_path):
    alerter = DegradationAlerter(alert_log_path=tmp_path / "missing.jsonl")
    assert alerter.load_alert_log() == []


# ---------------------------------------------------------------------------
# validate_alert_event
# ---------------------------------------------------------------------------

def test_validate_alert_accepts_valid_event():
    event = AlertEvent(
        alert_type="degradation_warning",
        metric_name="hit_rate",
        metric_value=0.45,
        threshold=0.48,
        model_id="hrm-1.0.0-abc",
        model_version="1.0.0",
        message="WARNING: hit_rate low",
        ts_utc=FIXED_TS,
    )
    errors = validate_alert_event(event.as_dict())
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_alert_detects_missing_key():
    event = AlertEvent(
        alert_type="degradation_warning", metric_name="hit_rate",
        metric_value=0.45, threshold=0.48, model_id="hrm-1.0.0-abc",
        model_version="1.0.0", message="low", ts_utc=FIXED_TS,
    )
    d = event.as_dict()
    del d["metric_name"]
    errors = validate_alert_event(d)
    assert any("metric_name" in e for e in errors)


def test_validate_alert_detects_wrong_schema():
    event = AlertEvent(
        alert_type="degradation_warning", metric_name="hit_rate",
        metric_value=0.45, threshold=0.48, model_id="hrm-1.0.0-abc",
        model_version="1.0.0", message="low", ts_utc=FIXED_TS,
    )
    d = event.as_dict()
    d["schema"] = "wrong"
    errors = validate_alert_event(d)
    assert any("schema" in e for e in errors)


def test_validate_alert_detects_invalid_alert_type():
    event = AlertEvent(
        alert_type="degradation_warning", metric_name="hit_rate",
        metric_value=0.45, threshold=0.48, model_id="hrm-1.0.0-abc",
        model_version="1.0.0", message="low", ts_utc=FIXED_TS,
    )
    d = event.as_dict()
    d["alert_type"] = "INVALID"
    errors = validate_alert_event(d)
    assert any("alert_type" in e for e in errors)


# ---------------------------------------------------------------------------
# DecisionLogIndex
# ---------------------------------------------------------------------------

def _write_audit_entry(path: Path, model_id: str, version: str,
                       decision: str, prev: str, nxt: str,
                       ts: str, passed: bool) -> None:
    entry = {
        "schema": "moneyfan.hrm.promotion_gate.audit.v1",
        "ts_utc": ts,
        "model_id": model_id,
        "model_version": version,
        "decision": decision,
        "previous_state": prev,
        "next_state": nxt,
        "passed": passed,
        "criteria_snapshot": {},
        "metrics_snapshot": {},
        "failures": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_rollback_entry(path: Path, model_id: str, version: str,
                          trigger: str, prev: str, nxt: str,
                          ts: str, reason: str) -> None:
    entry = {
        "schema": "moneyfan.hrm.rollback.v1",
        "ts_utc": ts,
        "model_id": model_id,
        "model_version": version,
        "triggered": True,
        "trigger_type": trigger,
        "previous_state": prev,
        "next_state": nxt,
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def test_decision_log_index_empty_when_no_files(tmp_path):
    index = DecisionLogIndex(
        audit_log_path=tmp_path / "audit.jsonl",
        rollback_log_path=tmp_path / "rollback.jsonl",
    )
    assert index.load_all() == []


def test_decision_log_index_loads_audit_entries(tmp_path):
    audit = tmp_path / "audit.jsonl"
    _write_audit_entry(audit, "hrm-1.0.0-abc", "1.0.0",
                       "promote_to_validated", "pending", "validated",
                       FIXED_TS, True)
    index = DecisionLogIndex(audit_log_path=audit)
    entries = index.load_all()
    assert len(entries) == 1
    assert entries[0].source == "audit"
    assert entries[0].decision == "promote_to_validated"


def test_decision_log_index_loads_rollback_entries(tmp_path):
    rb = tmp_path / "rollback.jsonl"
    _write_rollback_entry(rb, "hrm-1.0.0-abc", "1.0.0",
                          "manual", "promoted", "rolled_back",
                          FIXED_TS, "security incident")
    index = DecisionLogIndex(rollback_log_path=rb)
    entries = index.load_all()
    assert len(entries) == 1
    assert entries[0].source == "rollback"
    assert entries[0].reason == "security incident"


def test_decision_log_index_merged_and_sorted(tmp_path):
    audit = tmp_path / "audit.jsonl"
    rb = tmp_path / "rollback.jsonl"
    ts_early = "2026-03-10T00:00:00+00:00"
    ts_later = "2026-03-10T06:00:00+00:00"
    _write_rollback_entry(rb, "hrm-1.0.0-abc", "1.0.0",
                          "automatic", "promoted", "rolled_back",
                          ts_later, "regression")
    _write_audit_entry(audit, "hrm-1.0.0-abc", "1.0.0",
                       "promote_to_validated", "pending", "validated",
                       ts_early, True)
    index = DecisionLogIndex(audit_log_path=audit, rollback_log_path=rb)
    entries = index.load_all()
    assert len(entries) == 2
    assert entries[0].ts_utc == ts_early  # audit first
    assert entries[1].ts_utc == ts_later  # rollback second


def test_decision_log_index_for_model_filter(tmp_path):
    audit = tmp_path / "audit.jsonl"
    _write_audit_entry(audit, "hrm-1.0.0-aaa", "1.0.0",
                       "promote_to_validated", "pending", "validated",
                       FIXED_TS, True)
    _write_audit_entry(audit, "hrm-1.1.0-bbb", "1.1.0",
                       "promote_to_promoted", "validated", "promoted",
                       FIXED_TS, True)
    index = DecisionLogIndex(audit_log_path=audit)
    filtered = index.for_model("hrm-1.0.0-aaa")
    assert len(filtered) == 1
    assert filtered[0].model_id == "hrm-1.0.0-aaa"


def test_decision_log_index_for_version_filter(tmp_path):
    audit = tmp_path / "audit.jsonl"
    _write_audit_entry(audit, "hrm-1.0.0-aaa", "1.0.0",
                       "promote_to_validated", "pending", "validated",
                       FIXED_TS, True)
    _write_audit_entry(audit, "hrm-1.1.0-bbb", "1.1.0",
                       "promote_to_promoted", "validated", "promoted",
                       FIXED_TS, True)
    index = DecisionLogIndex(audit_log_path=audit)
    filtered = index.for_version("1.0.0")
    assert len(filtered) == 1
    assert filtered[0].model_version == "1.0.0"


# ---------------------------------------------------------------------------
# VALID_ALERT_TYPES completeness
# ---------------------------------------------------------------------------

def test_valid_alert_types_contains_expected():
    assert "degradation_warning" in VALID_ALERT_TYPES
    assert "degradation_critical" in VALID_ALERT_TYPES
    assert "drawdown_warning" in VALID_ALERT_TYPES
    assert "drawdown_critical" in VALID_ALERT_TYPES
    assert "loss_spike" in VALID_ALERT_TYPES
