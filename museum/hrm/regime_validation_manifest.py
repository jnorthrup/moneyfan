from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_REGIME_VALIDATION_MANIFEST_PATH = Path("config/regime_validation_manifest.json")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _as_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raise ValueError("tags must be a string or list")


def _normalize_slice(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"slice #{index} must be an object")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError(f"slice #{index} missing required field 'name'")

    regime = str(raw.get("regime") or "").strip()
    if not regime:
        raise ValueError(f"slice '{name}' missing required field 'regime'")

    weight = _safe_float(raw.get("weight", 1.0), 1.0)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"slice '{name}' has invalid weight")

    out = dict(raw)
    out["name"] = name
    out["regime"] = regime
    out["tags"] = _as_tags(raw.get("tags"))
    out["weight"] = float(weight)
    out["mandatory"] = bool(raw.get("mandatory", False))
    if "symbols" in out and out["symbols"] is not None:
        out["symbols"] = str(out["symbols"])
    return out


def validate_regime_validation_manifest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("manifest payload must be an object")

    version = int(payload.get("version", 1) or 1)
    raw_slices = payload.get("slices")
    if not isinstance(raw_slices, list) or not raw_slices:
        raise ValueError("manifest 'slices' must be a non-empty list")

    normalized_slices: List[Dict[str, Any]] = []
    names_seen = set()
    for i, row in enumerate(raw_slices, start=1):
        s = _normalize_slice(row, i)
        if s["name"] in names_seen:
            raise ValueError(f"duplicate slice name '{s['name']}'")
        names_seen.add(s["name"])
        normalized_slices.append(s)

    out = dict(payload)
    out["version"] = int(version)
    out["slices"] = normalized_slices
    return out


def load_regime_validation_manifest(path: Path | str | None = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_REGIME_VALIDATION_MANIFEST_PATH
    with open(p, "r") as f:
        payload = json.load(f)
    out = validate_regime_validation_manifest_payload(payload)
    out["source_path"] = str(p.resolve())
    return out


def manifest_to_validation_profiles(
    manifest: Dict[str, Any],
    base_profile: Dict[str, Any],
    *,
    source_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    validated = validate_regime_validation_manifest_payload(manifest)
    base = dict(base_profile or {})
    profiles: List[Dict[str, Any]] = []
    for row in list(validated.get("slices") or []):
        spec = dict(base)
        spec.update(dict(row))
        spec["name"] = str(spec.get("name") or "slice")
        spec["regime"] = str(spec.get("regime") or spec["name"])
        spec["tags"] = _as_tags(spec.get("tags"))
        spec["weight"] = float(max(0.0, _safe_float(spec.get("weight", 1.0), 1.0)))
        spec["mandatory"] = bool(spec.get("mandatory", False))
        spec["manifest_source_path"] = str(source_path or validated.get("source_path") or "")
        spec["manifest_name"] = str(validated.get("name") or "")
        spec["manifest_version"] = int(validated.get("version", 1) or 1)
        profiles.append(spec)
    return profiles


def summarize_validation_manifest_profiles(profiles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [dict(x) for x in (profiles or []) if isinstance(x, dict)]
    if not rows:
        return None
    first_path = str(rows[0].get("manifest_source_path") or "").strip()
    first_name = str(rows[0].get("manifest_name") or "").strip()
    first_version = rows[0].get("manifest_version")
    if not any([first_path, first_name, first_version is not None]):
        return None

    regimes = [str(r.get("regime") or r.get("name") or "") for r in rows]
    weights_by_slice = {
        str(r.get("name") or f"slice_{i+1:02d}"): float(max(0.0, _safe_float(r.get("weight"), 0.0)))
        for i, r in enumerate(rows)
    }
    mandatory_slices = [
        str(r.get("name") or f"slice_{i+1:02d}")
        for i, r in enumerate(rows)
        if bool(r.get("mandatory", False))
    ]
    return {
        "manifest_source_path": first_path or None,
        "manifest_name": first_name or None,
        "manifest_version": (int(first_version) if first_version is not None else None),
        "slice_count": int(len(rows)),
        "regimes": regimes,
        "weights_by_slice": weights_by_slice,
        "mandatory_slices": mandatory_slices,
        "total_weight": float(sum(weights_by_slice.values())),
    }


__all__ = [
    "DEFAULT_REGIME_VALIDATION_MANIFEST_PATH",
    "load_regime_validation_manifest",
    "manifest_to_validation_profiles",
    "summarize_validation_manifest_profiles",
    "validate_regime_validation_manifest_payload",
]
