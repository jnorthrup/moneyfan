"""
Tests for deterministic drawdown stress profiles.

Validates:
  - Profile schema stability (required fields present, types correct)
  - Profile determinism (same profile_id -> same path on every call)
  - Expected guardrail state alignment with default thresholds
  - Source artifact shape for freqtrade handoff compatibility
  - DD band bounds are satisfied by the profile's own path
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pytest

from execution.drawdown_stress_profiles import (
    DRAWDOWN_STRESS_PROFILES,
    PROFILE_BENIGN,
    PROFILE_WARN_BREACH,
    PROFILE_DERISK_PATH,
    PROFILE_FULL_HALT,
    PROFILE_OSCILLATING_WARN,
    SOURCE_ARTIFACT_REQUIRED_KEYS,
    _WARN_PCT,
    _DERISK_PCT,
    _HALT_PCT,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_registry_contains_all_profiles():
    """All named profiles must be registered in DRAWDOWN_STRESS_PROFILES."""
    expected = {
        "dd_stress_benign_v1",
        "dd_stress_warn_breach_v1",
        "dd_stress_derisk_path_v1",
        "dd_stress_full_halt_v1",
        "dd_stress_oscillating_warn_v1",
    }
    assert set(DRAWDOWN_STRESS_PROFILES.keys()) == expected


def test_registry_keys_match_profile_ids():
    """Registry dict key must equal profile.profile_id for every entry."""
    for key, profile in DRAWDOWN_STRESS_PROFILES.items():
        assert key == profile.profile_id, (
            f"Key mismatch: registry key={key!r} but profile_id={profile.profile_id!r}"
        )


# ---------------------------------------------------------------------------
# Schema stability (required fields, types)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_has_required_fields(profile):
    """Every profile must have non-empty profile_id, description, regime_tags."""
    assert isinstance(profile.profile_id, str) and profile.profile_id
    assert isinstance(profile.description, str) and profile.description
    assert isinstance(profile.regime_tags, tuple) and len(profile.regime_tags) >= 1


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_drawdown_path_is_nonempty_and_typed(profile):
    """drawdown_path_pct must be a non-empty tuple of (int, float) pairs."""
    assert len(profile.drawdown_path_pct) >= 1
    for item in profile.drawdown_path_pct:
        assert len(item) == 2
        assert isinstance(item[0], int)
        assert isinstance(item[1], float)


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_guardrail_states_parallel(profile):
    """expected_guardrail_states must have the same length as drawdown_path_pct."""
    assert len(profile.expected_guardrail_states) == len(profile.drawdown_path_pct)


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_guardrail_states_valid_values(profile):
    """All expected guardrail states must be valid state names."""
    valid = {"normal", "warn", "derisk", "halt"}
    for state in profile.expected_guardrail_states:
        assert state in valid, f"Invalid guardrail state {state!r} in {profile.profile_id}"


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_dd_band_has_required_keys(profile):
    """expected_dd_band must contain min_pct and max_pct."""
    assert "min_pct" in profile.expected_dd_band
    assert "max_pct" in profile.expected_dd_band
    assert profile.expected_dd_band["min_pct"] <= profile.expected_dd_band["max_pct"]


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_profile_capital_is_positive(profile):
    assert profile.capital > 0.0


# ---------------------------------------------------------------------------
# Determinism: same profile produces identical path on repeated access
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile_id", list(DRAWDOWN_STRESS_PROFILES.keys()))
def test_profile_is_deterministic(profile_id):
    """Accessing the same profile twice must return identical drawdown paths."""
    p1 = DRAWDOWN_STRESS_PROFILES[profile_id]
    p2 = DRAWDOWN_STRESS_PROFILES[profile_id]
    assert p1.drawdown_path_pct == p2.drawdown_path_pct
    assert p1.expected_guardrail_states == p2.expected_guardrail_states


def test_profile_ids_are_globally_unique():
    """No two profiles in the registry may share a profile_id."""
    ids = [p.profile_id for p in DRAWDOWN_STRESS_PROFILES.values()]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# DD band satisfaction: path max must fall within band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_path_max_within_dd_band(profile):
    """The maximum drawdown in the path must fall within the expected_dd_band."""
    path_max = profile.max_drawdown_pct()
    lo = profile.expected_dd_band["min_pct"]
    hi = profile.expected_dd_band["max_pct"]
    assert path_max >= lo, (
        f"{profile.profile_id}: max_drawdown={path_max:.3f} < band_min={lo:.3f}"
    )
    assert path_max <= hi, (
        f"{profile.profile_id}: max_drawdown={path_max:.3f} > band_max={hi:.3f}"
    )


# ---------------------------------------------------------------------------
# Expected guardrail state alignment with default thresholds
# ---------------------------------------------------------------------------

def _classify_dd(dd_pct: float) -> str:
    """Replicate the guardrail threshold logic with defaults."""
    if dd_pct >= _HALT_PCT:
        return "halt"
    if dd_pct >= _DERISK_PCT:
        return "derisk"
    if dd_pct >= _WARN_PCT:
        return "warn"
    return "normal"


@pytest.mark.parametrize("profile", [
    PROFILE_BENIGN, PROFILE_WARN_BREACH, PROFILE_DERISK_PATH,
    PROFILE_FULL_HALT, PROFILE_OSCILLATING_WARN,
])
def test_expected_states_match_threshold_classification(profile):
    """Expected guardrail states must match _classify_dd() for each path step."""
    for idx, ((_, dd_pct), expected_state) in enumerate(
        zip(profile.drawdown_path_pct, profile.expected_guardrail_states)
    ):
        classified = _classify_dd(dd_pct)
        assert classified == expected_state, (
            f"{profile.profile_id}[{idx}]: dd={dd_pct:.3f} -> "
            f"classified={classified!r} but expected={expected_state!r}"
        )


# ---------------------------------------------------------------------------
# Specific profile properties
# ---------------------------------------------------------------------------

def test_benign_profile_never_leaves_normal():
    """Benign profile must never trigger any guardrail state change."""
    assert all(s == "normal" for s in PROFILE_BENIGN.expected_guardrail_states)
    assert PROFILE_BENIGN.max_drawdown_pct() < _WARN_PCT


def test_full_halt_profile_reaches_halt():
    """Full halt profile must eventually reach the halt state."""
    assert "halt" in PROFILE_FULL_HALT.expected_guardrail_states
    assert PROFILE_FULL_HALT.max_drawdown_pct() >= _HALT_PCT


def test_derisk_path_reaches_derisk_not_halt():
    """Derisk path profile must reach derisk but not halt."""
    assert "derisk" in PROFILE_DERISK_PATH.expected_guardrail_states
    assert "halt" not in PROFILE_DERISK_PATH.expected_guardrail_states


def test_warn_breach_profile_does_not_reach_derisk():
    """Warn breach profile must not reach derisk or halt."""
    for state in PROFILE_WARN_BREACH.expected_guardrail_states:
        assert state in ("normal", "warn"), (
            f"Unexpected state {state!r} in warn_breach profile"
        )


def test_oscillating_warn_alternates():
    """Oscillating warn profile must have at least two transitions between normal and warn."""
    states = PROFILE_OSCILLATING_WARN.expected_guardrail_states
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert transitions >= 2, f"Expected >= 2 transitions, got {transitions}"


# ---------------------------------------------------------------------------
# PnL helper
# ---------------------------------------------------------------------------

def test_pnl_at_step_matches_drawdown_pct():
    """pnl_at_step() must return -(drawdown_pct * capital)."""
    profile = PROFILE_FULL_HALT
    for idx, (_, dd_pct) in enumerate(profile.drawdown_path_pct):
        expected_pnl = -dd_pct * profile.capital
        assert profile.pnl_at_step(idx) == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# Source artifact schema (freqtrade handoff compatibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_source_artifact_has_required_keys(profile):
    """as_source_artifact() must include all SOURCE_ARTIFACT_REQUIRED_KEYS."""
    artifact = profile.as_source_artifact()
    for key in SOURCE_ARTIFACT_REQUIRED_KEYS:
        assert key in artifact, f"{profile.profile_id}: missing artifact key {key!r}"


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_source_artifact_schema_field(profile):
    """Artifact schema field must be the canonical moneyfan schema identifier."""
    artifact = profile.as_source_artifact()
    assert artifact["schema"] == "moneyfan.drawdown.stress_profile.v1"


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_source_artifact_is_json_serializable(profile):
    """as_source_artifact() output must be fully JSON-serializable."""
    artifact = profile.as_source_artifact()
    serialized = json.dumps(artifact)
    roundtripped = json.loads(serialized)
    assert roundtripped["profile_id"] == profile.profile_id


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_source_artifact_path_length_matches(profile):
    """Artifact drawdown_path_pct list length must match the profile's path."""
    artifact = profile.as_source_artifact()
    assert len(artifact["drawdown_path_pct"]) == len(profile.drawdown_path_pct)


@pytest.mark.parametrize("profile", list(DRAWDOWN_STRESS_PROFILES.values()))
def test_source_artifact_path_items_schema(profile):
    """Each drawdown_path_pct artifact item must have 'iteration' and 'drawdown_pct'."""
    artifact = profile.as_source_artifact()
    for item in artifact["drawdown_path_pct"]:
        assert "iteration" in item
        assert "drawdown_pct" in item
        assert isinstance(item["iteration"], int)
        assert isinstance(item["drawdown_pct"], float)
