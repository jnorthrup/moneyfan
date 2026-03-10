"""
Tests for source manifest with stable expression IDs (Phase 3 Slice 3).

Validates:
  - ExpressionEntry construction from codec_name and kotlingrad_fingerprint
  - stability_key is deterministic and stable across repeated calls
  - SourceManifest schema completeness and JSON-serializability
  - validate_manifest() catches all required-key violations
  - stability_fingerprint() is identical for identical codec sets
  - expression_id_source provenance values are enforced
  - CANONICAL_CODEC_NAMES has exactly 24 entries and all are unique
  - build_canonical_manifest() produces a complete, valid manifest
  - Cross-run stability: same inputs -> same stability_key on every call
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pytest

from execution.source_manifest import (
    SOURCE_MANIFEST_SCHEMA,
    REQUIRED_MANIFEST_KEYS,
    REQUIRED_ENTRY_KEYS,
    VALID_EXPRESSION_ID_SOURCES,
    CANONICAL_CODEC_NAMES,
    ExpressionEntry,
    SourceManifest,
    build_canonical_manifest,
    validate_manifest,
    _stability_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-03-10T01:21:00+00:00"


def _canonical() -> SourceManifest:
    return build_canonical_manifest(
        manifest_id="test-manifest-001",
        session_id="test-session-001",
        created_at=FIXED_TS,
        profile_id="dd_stress_benign_v1",
        mode="pretesting",
    )


# ---------------------------------------------------------------------------
# CANONICAL_CODEC_NAMES registry
# ---------------------------------------------------------------------------

def test_canonical_codec_names_count():
    """There must be exactly 24 canonical codec names."""
    assert len(CANONICAL_CODEC_NAMES) == 24


def test_canonical_codec_names_unique():
    """All 24 canonical codec names must be unique."""
    assert len(set(CANONICAL_CODEC_NAMES)) == 24


def test_canonical_codec_names_are_strings():
    for name in CANONICAL_CODEC_NAMES:
        assert isinstance(name, str) and name


def test_canonical_codec_names_all_prefixed():
    """All canonical names must start with 'codec_'."""
    for name in CANONICAL_CODEC_NAMES:
        assert name.startswith("codec_"), f"Expected codec_ prefix: {name!r}"


# ---------------------------------------------------------------------------
# ExpressionEntry: codec_name source
# ---------------------------------------------------------------------------

def test_entry_from_codec_name_fields():
    entry = ExpressionEntry.from_codec_name("codec_01_volatility_breakout")
    assert entry.expression_id == "codec_01_volatility_breakout"
    assert entry.expression_id_source == "codec_name"
    assert entry.label == "codec_01_volatility_breakout"
    assert isinstance(entry.stability_key, str) and len(entry.stability_key) == 16


def test_entry_from_codec_name_description_optional():
    entry = ExpressionEntry.from_codec_name("codec_02_momentum_rsi", description="RSI momentum signal")
    assert entry.description == "RSI momentum signal"


def test_entry_from_codec_name_no_description():
    entry = ExpressionEntry.from_codec_name("codec_03_ma_crossover")
    assert entry.description is None


# ---------------------------------------------------------------------------
# ExpressionEntry: kotlingrad_fingerprint source
# ---------------------------------------------------------------------------

def test_entry_from_kotlingrad_fingerprint_fields():
    fp = "abcdef1234567890"
    entry = ExpressionEntry.from_kotlingrad_fingerprint(fp, label="volatility_breakout")
    assert entry.expression_id == fp
    assert entry.expression_id_source == "kotlingrad_fingerprint"
    assert entry.label == "volatility_breakout"
    assert isinstance(entry.stability_key, str) and len(entry.stability_key) == 16


def test_entry_from_kotlingrad_fingerprint_stability_key_differs_from_codec_name():
    """Same label, different source -> different stability_key."""
    entry_kgrad = ExpressionEntry.from_kotlingrad_fingerprint("FINGERPRINT_X", label="codec_01")
    entry_codec = ExpressionEntry.from_codec_name("codec_01")
    assert entry_kgrad.stability_key != entry_codec.stability_key


# ---------------------------------------------------------------------------
# stability_key determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codec_name", list(CANONICAL_CODEC_NAMES[:6]))
def test_stability_key_is_deterministic(codec_name):
    """Calling _stability_key with the same args must always return the same value."""
    k1 = _stability_key(codec_name, "codec_name")
    k2 = _stability_key(codec_name, "codec_name")
    assert k1 == k2


def test_stability_key_differs_across_codecs():
    """Different codec names must produce different stability keys."""
    keys = {_stability_key(name, "codec_name") for name in CANONICAL_CODEC_NAMES}
    assert len(keys) == 24, "Stability key collision detected across canonical codecs"


def test_stability_key_length():
    key = _stability_key("codec_01_volatility_breakout", "codec_name")
    assert len(key) == 16


# ---------------------------------------------------------------------------
# ExpressionEntry.as_dict()
# ---------------------------------------------------------------------------

def test_entry_as_dict_has_required_keys():
    entry = ExpressionEntry.from_codec_name("codec_01_volatility_breakout")
    d = entry.as_dict()
    for key in REQUIRED_ENTRY_KEYS:
        assert key in d, f"Missing required entry key: {key!r}"


def test_entry_as_dict_description_absent_when_none():
    entry = ExpressionEntry.from_codec_name("codec_01_volatility_breakout")
    d = entry.as_dict()
    assert "description" not in d


def test_entry_as_dict_description_present_when_set():
    entry = ExpressionEntry.from_codec_name("codec_01_volatility_breakout", description="dd trigger kernel")
    d = entry.as_dict()
    assert d["description"] == "dd trigger kernel"


def test_entry_as_dict_json_serializable():
    entry = ExpressionEntry.from_codec_name("codec_06_grid_trading", description="grid")
    json.dumps(entry.as_dict())  # must not raise


# ---------------------------------------------------------------------------
# SourceManifest
# ---------------------------------------------------------------------------

def test_manifest_schema_field():
    m = _canonical()
    d = m.as_dict()
    assert d["schema"] == SOURCE_MANIFEST_SCHEMA
    assert d["schema"] == "moneyfan.source.manifest.v1"


def test_manifest_has_required_keys():
    m = _canonical()
    d = m.as_dict()
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in d, f"Missing required manifest key: {key!r}"


def test_manifest_expression_count_matches():
    m = _canonical()
    d = m.as_dict()
    assert d["expression_count"] == 24
    assert len(d["expression_entries"]) == 24


def test_manifest_profile_id_propagated():
    m = _canonical()
    assert m.as_dict()["profile_id"] == "dd_stress_benign_v1"


def test_manifest_mode_propagated():
    m = _canonical()
    assert m.as_dict()["mode"] == "pretesting"


def test_manifest_json_serializable():
    m = _canonical()
    serialized = json.dumps(m.as_dict())
    roundtripped = json.loads(serialized)
    assert roundtripped["schema"] == SOURCE_MANIFEST_SCHEMA
    assert len(roundtripped["expression_entries"]) == 24


# ---------------------------------------------------------------------------
# stability_fingerprint
# ---------------------------------------------------------------------------

def test_stability_fingerprint_deterministic():
    """Same manifest inputs must produce the same stability fingerprint."""
    m1 = _canonical()
    m2 = _canonical()
    assert m1.stability_fingerprint() == m2.stability_fingerprint()


def test_stability_fingerprint_changes_when_codec_order_changes():
    """Changing codec order must change the stability fingerprint."""
    codecs_a = CANONICAL_CODEC_NAMES
    codecs_b = CANONICAL_CODEC_NAMES[::-1]  # reversed order
    m_a = build_canonical_manifest(
        manifest_id="a", session_id="s", created_at=FIXED_TS, codec_names=codecs_a
    )
    m_b = build_canonical_manifest(
        manifest_id="b", session_id="s", created_at=FIXED_TS, codec_names=codecs_b
    )
    assert m_a.stability_fingerprint() != m_b.stability_fingerprint()


def test_stability_fingerprint_length():
    m = _canonical()
    assert len(m.stability_fingerprint()) == 24


# ---------------------------------------------------------------------------
# expression_ids()
# ---------------------------------------------------------------------------

def test_expression_ids_returns_all_codec_names():
    m = _canonical()
    ids = m.expression_ids()
    assert len(ids) == 24
    assert set(ids) == set(CANONICAL_CODEC_NAMES)


# ---------------------------------------------------------------------------
# validate_manifest()
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_manifest():
    m = _canonical()
    errors = validate_manifest(m.as_dict())
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_detects_missing_required_key():
    d = _canonical().as_dict()
    del d["manifest_id"]
    errors = validate_manifest(d)
    assert any("manifest_id" in e for e in errors)


def test_validate_detects_wrong_schema():
    d = _canonical().as_dict()
    d["schema"] = "wrong.schema"
    errors = validate_manifest(d)
    assert any("schema" in e for e in errors)


def test_validate_detects_empty_expression_entries():
    d = _canonical().as_dict()
    d["expression_entries"] = []
    errors = validate_manifest(d)
    assert any("expression_entries" in e for e in errors)


def test_validate_detects_missing_entry_key():
    d = _canonical().as_dict()
    del d["expression_entries"][0]["stability_key"]
    errors = validate_manifest(d)
    assert any("stability_key" in e for e in errors)


def test_validate_detects_invalid_expression_id_source():
    d = _canonical().as_dict()
    d["expression_entries"][0]["expression_id_source"] = "INVALID_SOURCE"
    errors = validate_manifest(d)
    assert any("expression_id_source" in e for e in errors)


def test_validate_empty_dict_reports_all_required_keys():
    errors = validate_manifest({})
    missing = {e.split(":")[-1].strip().strip("'") for e in errors if "Missing" in e}
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in missing, f"Expected {key!r} reported as missing"


# ---------------------------------------------------------------------------
# Cross-run stability: same inputs -> same stability_key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codec_name", [
    "codec_01_volatility_breakout",
    "codec_06_grid_trading",
    "codec_24_meta_allocator",
])
def test_stability_key_cross_run_stable(codec_name):
    """Stability key must be the same on repeated isolated calls."""
    e1 = ExpressionEntry.from_codec_name(codec_name)
    e2 = ExpressionEntry.from_codec_name(codec_name)
    assert e1.stability_key == e2.stability_key


def test_full_manifest_stability_across_builds():
    """Building the same canonical manifest twice must yield the same fingerprint."""
    m1 = build_canonical_manifest(
        manifest_id="m1", session_id="s1", created_at=FIXED_TS
    )
    m2 = build_canonical_manifest(
        manifest_id="m2", session_id="s2", created_at="2026-03-11T00:00:00+00:00"
    )
    # Different manifest_id/session_id/timestamp but same codec set
    assert m1.stability_fingerprint() == m2.stability_fingerprint()


# ---------------------------------------------------------------------------
# VALID_EXPRESSION_ID_SOURCES
# ---------------------------------------------------------------------------

def test_valid_expression_id_sources_contains_expected():
    assert "kotlingrad_fingerprint" in VALID_EXPRESSION_ID_SOURCES
    assert "codec_name" in VALID_EXPRESSION_ID_SOURCES
    assert "manual" in VALID_EXPRESSION_ID_SOURCES


# ---------------------------------------------------------------------------
# Integration with DrawdownStressProfile
# ---------------------------------------------------------------------------

def test_manifest_can_reference_stress_profile_id():
    from execution.drawdown_stress_profiles import PROFILE_FULL_HALT
    m = build_canonical_manifest(
        manifest_id="halt-manifest-001",
        session_id=PROFILE_FULL_HALT.profile_id,
        created_at=FIXED_TS,
        profile_id=PROFILE_FULL_HALT.profile_id,
        mode="pretesting",
    )
    d = m.as_dict()
    assert d["profile_id"] == PROFILE_FULL_HALT.profile_id
    assert d["session_id"] == PROFILE_FULL_HALT.profile_id
    errors = validate_manifest(d)
    assert errors == []
