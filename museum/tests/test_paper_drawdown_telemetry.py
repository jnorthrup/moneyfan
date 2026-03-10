"""
Tests for paper drawdown telemetry schema (Phase 2 of pretesting track).

Validates:
  - Schema completeness: all required keys present in every built event
  - Type correctness: fields have declared types
  - Threshold crossing payloads: events correctly reflect state transitions
  - Validation function: catches missing keys, bad states, negative drawdown
  - Threshold crossing helpers: is_threshold_crossing, direction
  - JSON-serializability for freqtrade handoff compatibility
  - Reconciliation metadata: guardrail fields present and correctly typed
  - Integration with DrawdownStressProfile paths
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pytest

from execution.paper_drawdown_telemetry import (
    TELEMETRY_SCHEMA,
    REQUIRED_TELEMETRY_KEYS,
    build_paper_drawdown_telemetry_event,
    validate_telemetry_event,
    is_threshold_crossing,
    threshold_crossing_direction,
)
from execution.drawdown_stress_profiles import (
    PROFILE_FULL_HALT,
    PROFILE_BENIGN,
    PROFILE_DERISK_PATH,
    PROFILE_OSCILLATING_WARN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_event(**overrides) -> dict:
    """Build a minimal valid telemetry event."""
    kwargs = dict(
        signal_id="hrm-test-signal-001",
        iteration=0,
        drawdown_pct=0.0,
        threshold_state="normal",
        threshold_warn=0.05,
        threshold_derisk=0.08,
        threshold_halt=0.12,
        equity=10_000.0,
        peak_equity=10_000.0,
        mode="paper",
        guardrail_action_active=False,
        position_size_scale=1.0,
        new_entries_allowed=True,
        ts_utc="2026-03-10T01:00:00+00:00",
    )
    kwargs.update(overrides)
    return build_paper_drawdown_telemetry_event(**kwargs)


# ---------------------------------------------------------------------------
# Schema field: schema string
# ---------------------------------------------------------------------------

def test_schema_field_is_canonical():
    event = _minimal_event()
    assert event["schema"] == TELEMETRY_SCHEMA
    assert event["schema"] == "moneyfan.paper.drawdown.telemetry.v1"


# ---------------------------------------------------------------------------
# Required keys completeness
# ---------------------------------------------------------------------------

def test_all_required_keys_present_minimal():
    """A minimal event must contain every REQUIRED_TELEMETRY_KEYS entry."""
    event = _minimal_event()
    for key in REQUIRED_TELEMETRY_KEYS:
        assert key in event, f"Missing required key: {key!r}"


def test_required_keys_do_not_include_optional_fields():
    """Optional fields (profile_id, effective_top_k) are not in REQUIRED_TELEMETRY_KEYS."""
    assert "profile_id" not in REQUIRED_TELEMETRY_KEYS
    assert "effective_top_k" not in REQUIRED_TELEMETRY_KEYS


# ---------------------------------------------------------------------------
# Type correctness
# ---------------------------------------------------------------------------

def test_field_types_minimal():
    event = _minimal_event()
    assert isinstance(event["schema"], str)
    assert isinstance(event["ts_utc"], str)
    assert isinstance(event["signal_id"], str)
    assert isinstance(event["iteration"], int)
    assert isinstance(event["drawdown_pct"], float)
    assert isinstance(event["threshold_state"], str)
    assert isinstance(event["threshold_warn"], float)
    assert isinstance(event["threshold_derisk"], float)
    assert isinstance(event["threshold_halt"], float)
    assert isinstance(event["equity"], float)
    assert isinstance(event["peak_equity"], float)
    assert isinstance(event["mode"], str)
    assert isinstance(event["guardrail_action_active"], bool)
    assert isinstance(event["position_size_scale"], float)
    assert isinstance(event["new_entries_allowed"], bool)


def test_optional_effective_top_k_type():
    event = _minimal_event(effective_top_k=5)
    assert "effective_top_k" in event
    assert isinstance(event["effective_top_k"], int)
    assert event["effective_top_k"] == 5


def test_optional_profile_id_type():
    event = _minimal_event(profile_id="dd_stress_full_halt_v1")
    assert "profile_id" in event
    assert isinstance(event["profile_id"], str)


def test_optional_fields_absent_when_not_supplied():
    event = _minimal_event()
    assert "effective_top_k" not in event
    assert "profile_id" not in event


# ---------------------------------------------------------------------------
# Threshold state validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["normal", "warn", "derisk", "halt"])
def test_valid_threshold_states_accepted(state):
    event = _minimal_event(
        threshold_state=state,
        drawdown_pct=0.0 if state == "normal" else 0.05,
        guardrail_action_active=(state != "normal"),
    )
    assert event["threshold_state"] == state


def test_invalid_threshold_state_raises():
    with pytest.raises(ValueError, match="threshold_state"):
        _minimal_event(threshold_state="UNKNOWN")


def test_negative_drawdown_pct_raises():
    with pytest.raises(ValueError, match="drawdown_pct"):
        _minimal_event(drawdown_pct=-0.01)


# ---------------------------------------------------------------------------
# Threshold crossing payloads
# ---------------------------------------------------------------------------

def test_warn_threshold_crossing_event():
    """Event at exact warn threshold should have threshold_state='warn'."""
    event = _minimal_event(
        drawdown_pct=0.05,
        threshold_state="warn",
        guardrail_action_active=False,
    )
    assert event["threshold_state"] == "warn"
    assert event["drawdown_pct"] == pytest.approx(0.05)
    assert event["guardrail_action_active"] is False


def test_derisk_threshold_crossing_event():
    """Event at derisk threshold should reflect active guardrail action."""
    event = _minimal_event(
        drawdown_pct=0.08,
        threshold_state="derisk",
        guardrail_action_active=True,
        position_size_scale=0.5,
        new_entries_allowed=True,
        effective_top_k=2,
    )
    assert event["threshold_state"] == "derisk"
    assert event["guardrail_action_active"] is True
    assert event["position_size_scale"] == pytest.approx(0.5)
    assert event["effective_top_k"] == 2
    assert event["new_entries_allowed"] is True


def test_halt_threshold_crossing_event():
    """Event at halt should have new_entries_allowed=False and top_k=0."""
    event = _minimal_event(
        drawdown_pct=0.12,
        threshold_state="halt",
        guardrail_action_active=True,
        position_size_scale=0.0,
        new_entries_allowed=False,
        effective_top_k=0,
    )
    assert event["threshold_state"] == "halt"
    assert event["new_entries_allowed"] is False
    assert event["effective_top_k"] == 0
    assert event["position_size_scale"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validate function
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_event():
    event = _minimal_event()
    errors = validate_telemetry_event(event)
    assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_detects_missing_required_key():
    event = _minimal_event()
    del event["signal_id"]
    errors = validate_telemetry_event(event)
    assert any("signal_id" in e for e in errors)


def test_validate_detects_wrong_schema():
    event = _minimal_event()
    event["schema"] = "wrong.schema"
    errors = validate_telemetry_event(event)
    assert any("schema" in e.lower() for e in errors)


def test_validate_detects_invalid_threshold_state():
    event = _minimal_event()
    event["threshold_state"] = "INVALID"
    errors = validate_telemetry_event(event)
    assert any("threshold_state" in e for e in errors)


def test_validate_detects_negative_drawdown():
    event = _minimal_event()
    event["drawdown_pct"] = -0.05
    errors = validate_telemetry_event(event)
    assert any("drawdown_pct" in e for e in errors)


def test_validate_empty_event_reports_all_required_keys():
    errors = validate_telemetry_event({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_TELEMETRY_KEYS:
        assert key in missing, f"Expected {key!r} to be reported as missing"


# ---------------------------------------------------------------------------
# JSON serializability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,dd,active,scale,entries", [
    ("normal",  0.00, False, 1.0, True),
    ("warn",    0.05, False, 1.0, True),
    ("derisk",  0.08, True,  0.5, True),
    ("halt",    0.12, True,  0.0, False),
])
def test_event_is_json_serializable(state, dd, active, scale, entries):
    event = _minimal_event(
        threshold_state=state,
        drawdown_pct=dd,
        guardrail_action_active=active,
        position_size_scale=scale,
        new_entries_allowed=entries,
    )
    serialized = json.dumps(event)
    roundtripped = json.loads(serialized)
    assert roundtripped["schema"] == TELEMETRY_SCHEMA
    assert roundtripped["threshold_state"] == state


# ---------------------------------------------------------------------------
# Threshold crossing helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prev,curr,expected", [
    ("normal", "warn",   True),
    ("warn",   "derisk", True),
    ("derisk", "halt",   True),
    ("halt",   "derisk", True),  # de-escalation
    ("normal", "normal", False),
    ("halt",   "halt",   False),
])
def test_is_threshold_crossing(prev, curr, expected):
    assert is_threshold_crossing(prev, curr) == expected


@pytest.mark.parametrize("prev,curr,direction", [
    ("normal", "warn",    "escalation"),
    ("normal", "derisk",  "escalation"),
    ("warn",   "halt",    "escalation"),
    ("halt",   "derisk",  "de-escalation"),
    ("derisk", "warn",    "de-escalation"),
    ("warn",   "normal",  "de-escalation"),
    ("normal", "normal",  "no_change"),
    ("derisk", "derisk",  "no_change"),
])
def test_threshold_crossing_direction(prev, curr, direction):
    assert threshold_crossing_direction(prev, curr) == direction


# ---------------------------------------------------------------------------
# Integration: simulate stress profile paths and validate all events
# ---------------------------------------------------------------------------

def _events_from_profile(profile, signal_id_prefix="test") -> list[dict]:
    """Simulate a paper-loop telemetry emission by replaying a stress profile."""
    events = []
    for idx, ((iteration, dd_pct), state) in enumerate(
        zip(profile.drawdown_path_pct, profile.expected_guardrail_states)
    ):
        is_halt = state == "halt"
        is_derisk = state == "derisk"
        event = build_paper_drawdown_telemetry_event(
            signal_id=f"{signal_id_prefix}-{idx:04d}",
            iteration=iteration,
            drawdown_pct=dd_pct,
            threshold_state=state,
            threshold_warn=0.05,
            threshold_derisk=0.08,
            threshold_halt=0.12,
            equity=profile.capital * (1.0 - dd_pct),
            peak_equity=profile.capital,
            mode="paper",
            guardrail_action_active=(state in ("derisk", "halt")),
            position_size_scale=0.5 if is_derisk else (0.0 if is_halt else 1.0),
            new_entries_allowed=not is_halt,
            effective_top_k=(0 if is_halt else (2 if is_derisk else None)),
            profile_id=profile.profile_id,
            ts_utc=f"2026-03-10T01:{idx:02d}:00+00:00",
        )
        events.append(event)
    return events


@pytest.mark.parametrize("profile", [
    PROFILE_BENIGN, PROFILE_FULL_HALT, PROFILE_DERISK_PATH, PROFILE_OSCILLATING_WARN,
])
def test_stress_profile_all_events_valid(profile):
    """All events emitted from a stress profile path must pass validation."""
    events = _events_from_profile(profile)
    assert len(events) == len(profile.drawdown_path_pct)
    for idx, event in enumerate(events):
        errors = validate_telemetry_event(event)
        assert errors == [], (
            f"{profile.profile_id}[{idx}]: validation errors: {errors}"
        )


def test_full_halt_profile_emits_halt_events():
    """Full halt profile replay must produce at least one halt-state event."""
    events = _events_from_profile(PROFILE_FULL_HALT)
    halt_events = [e for e in events if e["threshold_state"] == "halt"]
    assert len(halt_events) >= 1
    for e in halt_events:
        assert e["new_entries_allowed"] is False
        assert e["effective_top_k"] == 0


def test_benign_profile_emits_only_normal_events():
    """Benign profile replay must never emit a non-normal threshold state."""
    events = _events_from_profile(PROFILE_BENIGN)
    for e in events:
        assert e["threshold_state"] == "normal"
        assert e["guardrail_action_active"] is False
        assert e["new_entries_allowed"] is True


def test_events_contain_profile_id_when_supplied():
    """All events from a profiled run must carry the profile_id field."""
    events = _events_from_profile(PROFILE_DERISK_PATH)
    for e in events:
        assert e.get("profile_id") == PROFILE_DERISK_PATH.profile_id


def test_threshold_crossings_detected_in_oscillating_profile():
    """Oscillating warn profile must produce crossing events."""
    events = _events_from_profile(PROFILE_OSCILLATING_WARN)
    states = [e["threshold_state"] for e in events]
    crossings = [
        (i, threshold_crossing_direction(states[i - 1], states[i]))
        for i in range(1, len(states))
        if is_threshold_crossing(states[i - 1], states[i])
    ]
    assert len(crossings) >= 2, f"Expected >= 2 crossings, got {crossings}"
    directions = {d for _, d in crossings}
    assert "escalation" in directions
    assert "de-escalation" in directions


def test_all_stress_profile_events_json_serializable():
    """All events from all profiles must be fully JSON-serializable."""
    for profile in (PROFILE_BENIGN, PROFILE_FULL_HALT, PROFILE_DERISK_PATH):
        events = _events_from_profile(profile)
        for idx, event in enumerate(events):
            try:
                json.dumps(event)
            except (TypeError, ValueError) as exc:
                pytest.fail(
                    f"{profile.profile_id}[{idx}] not JSON-serializable: {exc}"
                )
