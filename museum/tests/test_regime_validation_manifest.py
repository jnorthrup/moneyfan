from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from hrm.regime_validation_manifest import (
    load_regime_validation_manifest,
    manifest_to_validation_profiles,
    summarize_validation_manifest_profiles,
    validate_regime_validation_manifest_payload,
)


def test_validate_manifest_normalizes_slice_fields():
    payload = {
        "version": 1,
        "slices": [
            {
                "name": "trend_anchor",
                "regime": "trend",
                "tags": "trend,anchor",
                "weight": 0.7,
                "mandatory": True,
                "symbols": "BTCUSDT,ETHUSDT",
                "max_steps": 120,
            },
            {
                "name": "chop_probe",
                "regime": "chop",
                "weight": 0.3,
                "mandatory": False,
                "symbols": "BTCUSDT,ETHUSDT",
            },
        ],
    }

    out = validate_regime_validation_manifest_payload(payload)

    assert out["version"] == 1
    assert len(out["slices"]) == 2
    assert out["slices"][0]["tags"] == ["trend", "anchor"]
    assert out["slices"][0]["weight"] == pytest.approx(0.7)
    assert out["slices"][0]["mandatory"] is True
    assert out["slices"][1]["tags"] == []
    assert out["slices"][1]["weight"] == pytest.approx(0.3)


def test_validate_manifest_rejects_negative_weight():
    payload = {
        "version": 1,
        "slices": [
            {
                "name": "bad",
                "regime": "trend",
                "weight": -0.1,
                "symbols": "BTCUSDT",
            }
        ],
    }
    with pytest.raises(ValueError, match="weight"):
        validate_regime_validation_manifest_payload(payload)


def test_validate_manifest_requires_nonempty_slices():
    with pytest.raises(ValueError, match="slices"):
        validate_regime_validation_manifest_payload({"version": 1, "slices": []})


def test_load_default_manifest_file_contains_goals_regimes():
    manifest_path = Path("config/regime_validation_manifest.json")
    out = load_regime_validation_manifest(manifest_path)
    regimes = {str(s["regime"]) for s in out["slices"]}
    assert {"trend", "chop", "high_vol", "shock"}.issubset(regimes)


def test_manifest_to_validation_profiles_merges_base_defaults():
    manifest = {
        "version": 1,
        "name": "demo",
        "slices": [
            {"name": "trend_anchor", "regime": "trend", "weight": 0.7, "mandatory": True, "symbols": "BTCUSDT"},
            {"name": "chop_probe", "regime": "chop", "weight": 0.3, "symbols": "ETHUSDT"},
        ],
    }
    base = {
        "top_k": 1,
        "signal_threshold": 0.55,
        "no_mechanical_veto": True,
    }

    profiles = manifest_to_validation_profiles(manifest, base, source_path="/tmp/demo.json")

    assert len(profiles) == 2
    assert profiles[0]["top_k"] == 1
    assert profiles[0]["signal_threshold"] == pytest.approx(0.55)
    assert profiles[0]["regime"] == "trend"
    assert profiles[0]["weight"] == pytest.approx(0.7)
    assert profiles[0]["mandatory"] is True
    assert profiles[0]["manifest_name"] == "demo"
    assert profiles[0]["manifest_source_path"] == "/tmp/demo.json"
    assert profiles[1]["mandatory"] is False


def test_summarize_validation_manifest_profiles_reports_provenance_and_coverage():
    profiles = [
        {
            "name": "trend_anchor",
            "regime": "trend",
            "weight": 0.7,
            "mandatory": True,
            "manifest_source_path": "/tmp/demo.json",
            "manifest_name": "demo",
            "manifest_version": 1,
        },
        {
            "name": "shock_guardrail",
            "regime": "shock",
            "weight": 0.3,
            "mandatory": False,
            "manifest_source_path": "/tmp/demo.json",
            "manifest_name": "demo",
            "manifest_version": 1,
        },
    ]
    summary = summarize_validation_manifest_profiles(profiles)
    assert summary is not None
    assert summary["manifest_source_path"] == "/tmp/demo.json"
    assert summary["manifest_name"] == "demo"
    assert summary["manifest_version"] == 1
    assert summary["slice_count"] == 2
    assert summary["regimes"] == ["trend", "shock"]
    assert summary["weights_by_slice"]["trend_anchor"] == pytest.approx(0.7)
    assert summary["mandatory_slices"] == ["trend_anchor"]
