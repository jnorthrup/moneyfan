"""
Source manifest for moneyfan drawdown source artifacts.

A source manifest records which expression IDs (Kotlingrad DSEL nodes / codec
kernels) were active during a pretesting or paper-trading session, enabling:

  - Cross-run expression-ID stability checks
  - Freqtrade insight aggregator ingestion (expression_id linkage)
  - Drift detection when codec/kernel definitions change

Schema: moneyfan.source.manifest.v1

Key concepts
------------
expression_id : str
    A stable opaque identifier for a feature kernel (codec).  When available,
    this is sourced from the Kotlingrad DSEL fingerprint of the expression tree.
    When not available (e.g. in pure-Python mode), it falls back to the codec
    name string (e.g. "codec_01_volatility_breakout").

expression_id_source : "kotlingrad_fingerprint" | "codec_name" | "manual"
    Provenance of the expression_id value.  "kotlingrad_fingerprint" is the
    preferred source; "codec_name" is the stable fallback used when running
    without Kotlingrad compilation.

stability_key : str
    A hash of (expression_id, expression_id_source) that must remain stable
    across runs that use the same kernel definition.  Used by compatibility
    checks.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


#: Canonical schema identifier
SOURCE_MANIFEST_SCHEMA = "moneyfan.source.manifest.v1"

#: Valid expression_id_source values
VALID_EXPRESSION_ID_SOURCES = frozenset((
    "kotlingrad_fingerprint",
    "codec_name",
    "manual",
))

#: Required top-level fields in a serialised manifest
REQUIRED_MANIFEST_KEYS = (
    "schema",
    "manifest_id",
    "session_id",
    "created_at",
    "expression_entries",
    "profile_id",
    "mode",
)

#: Required fields per expression entry
REQUIRED_ENTRY_KEYS = (
    "expression_id",
    "expression_id_source",
    "stability_key",
    "label",
)


def _stability_key(expression_id: str, source: str) -> str:
    """Derive a stable hash from (expression_id, source).

    This is intentionally a simple SHA-256 prefix so it remains stable across
    Python versions and OS environments.
    """
    raw = f"{expression_id}:{source}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExpressionEntry:
    """One codec/kernel entry in a source manifest."""

    expression_id: str
    """Stable ID — Kotlingrad fingerprint when available, else codec name."""

    expression_id_source: str
    """Provenance: 'kotlingrad_fingerprint' | 'codec_name' | 'manual'."""

    label: str
    """Human-readable name (e.g. 'volatility_breakout')."""

    stability_key: str
    """SHA-256[:16] of (expression_id, expression_id_source)."""

    description: Optional[str] = None
    """Optional longer description of the expression."""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "expression_id": self.expression_id,
            "expression_id_source": self.expression_id_source,
            "label": self.label,
            "stability_key": self.stability_key,
        }
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_codec_name(cls, codec_name: str, description: Optional[str] = None) -> "ExpressionEntry":
        """Build an entry using the codec name as the expression_id (stable fallback)."""
        source = "codec_name"
        return cls(
            expression_id=codec_name,
            expression_id_source=source,
            label=codec_name,
            stability_key=_stability_key(codec_name, source),
            description=description,
        )

    @classmethod
    def from_kotlingrad_fingerprint(
        cls,
        fingerprint: str,
        label: str,
        description: Optional[str] = None,
    ) -> "ExpressionEntry":
        """Build an entry using a Kotlingrad expression fingerprint."""
        source = "kotlingrad_fingerprint"
        return cls(
            expression_id=fingerprint,
            expression_id_source=source,
            label=label,
            stability_key=_stability_key(fingerprint, source),
            description=description,
        )


@dataclass(frozen=True)
class SourceManifest:
    """Immutable source manifest for one pretesting/paper session."""

    manifest_id: str
    """Stable identifier: e.g. 'manifest-smoke-dd-v1'."""

    session_id: str
    """Session or run identifier (may be a profile_id or a run UUID)."""

    created_at: str
    """ISO-8601 timestamp of manifest creation."""

    expression_entries: tuple[ExpressionEntry, ...]
    """Ordered tuple of expression entries active in this session."""

    profile_id: Optional[str]
    """Drawdown stress profile id, if applicable."""

    mode: str
    """Engine mode: 'paper' | 'pretesting' | 'live-preview'."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "manifest_id": self.manifest_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expression_entries": [e.as_dict() for e in self.expression_entries],
            "profile_id": self.profile_id,
            "mode": self.mode,
            "expression_count": len(self.expression_entries),
        }

    def stability_fingerprint(self) -> str:
        """Hash of all stability_keys; stable if and only if the same set of codecs are present."""
        concatenated = ":".join(e.stability_key for e in self.expression_entries)
        return hashlib.sha256(concatenated.encode()).hexdigest()[:24]

    def expression_ids(self) -> tuple[str, ...]:
        return tuple(e.expression_id for e in self.expression_entries)


def validate_manifest(manifest_dict: dict[str, Any]) -> list[str]:
    """Validate a serialised manifest dict. Returns list of error strings."""
    errors: list[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest_dict:
            errors.append(f"Missing required manifest key: {key!r}")
    if manifest_dict.get("schema") != SOURCE_MANIFEST_SCHEMA:
        errors.append(
            f"schema mismatch: expected {SOURCE_MANIFEST_SCHEMA!r},"
            f" got {manifest_dict.get('schema')!r}"
        )
    entries = manifest_dict.get("expression_entries")
    if not isinstance(entries, list) or len(entries) == 0:
        errors.append("expression_entries must be a non-empty list")
    else:
        for idx, entry in enumerate(entries):
            for key in REQUIRED_ENTRY_KEYS:
                if key not in entry:
                    errors.append(f"expression_entries[{idx}]: missing key {key!r}")
            src = entry.get("expression_id_source")
            if src not in VALID_EXPRESSION_ID_SOURCES:
                errors.append(
                    f"expression_entries[{idx}]: "
                    f"expression_id_source {src!r} not in {sorted(VALID_EXPRESSION_ID_SOURCES)!r}"
                )
    return errors


# ---------------------------------------------------------------------------
# Canonical codec expression table for moneyfan museum
# (stable across runs — codec definitions don't change unless the kernel does)
# ---------------------------------------------------------------------------

#: The 24 canonical codec names from the museum candle pipeline
CANONICAL_CODEC_NAMES: tuple[str, ...] = (
    "codec_01_volatility_breakout",
    "codec_02_momentum_rsi",
    "codec_03_ma_crossover",
    "codec_04_bollinger_squeeze",
    "codec_05_volume_anomaly",
    "codec_06_grid_trading",
    "codec_07_regime_garch",
    "codec_08_vwap_reversion",
    "codec_09_stochastic",
    "codec_10_supertrend",
    "codec_11_ema_ribbon",
    "codec_12_donchian_breakout",
    "codec_13_keltner_channel",
    "codec_14_macd_momentum",
    "codec_15_adx_trend_filter",
    "codec_16_pair_correlation",
    "codec_17_zscore_stat_arb",
    "codec_18_sector_rotation",
    "codec_19_ob_decay",
    "codec_20_energy_routing",
    "codec_21_hyperbolic_horizon",
    "codec_22_countercoin_router",
    "codec_23_regime_switch",
    "codec_24_meta_allocator",
)


def build_canonical_manifest(
    *,
    manifest_id: str,
    session_id: str,
    created_at: str,
    profile_id: Optional[str] = None,
    mode: str = "pretesting",
    codec_names: tuple[str, ...] = CANONICAL_CODEC_NAMES,
) -> SourceManifest:
    """Build a SourceManifest from the canonical codec name list.

    Uses 'codec_name' as the expression_id_source (Kotlingrad fingerprints
    are emitted by the compiler pass; this is the stable identity fallback).
    """
    entries = tuple(
        ExpressionEntry.from_codec_name(name) for name in codec_names
    )
    return SourceManifest(
        manifest_id=manifest_id,
        session_id=session_id,
        created_at=created_at,
        expression_entries=entries,
        profile_id=profile_id,
        mode=mode,
    )
