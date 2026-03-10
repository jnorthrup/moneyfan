"""
Deterministic drawdown stress profiles for pretesting and paper evaluation.

Each profile is a named, deterministic fixture that defines:
  - profile_id: stable string identifier (used by freqtrade handoff artifacts)
  - regime_tags: list of regime descriptors (e.g. ["trending", "high_volatility"])
  - drawdown_path_pct: ordered list of (iteration, drawdown_pct) steps
  - expected_dd_band: dict with min_pct, max_pct bounds for the full path
  - guardrail_crossings: expected guardrail state transitions at each step
  - capital: notional capital used to compute absolute PnL from pct
  - description: human-readable purpose label

These profiles are consumed by:
  1. Unit tests (schema stability, determinism)
  2. Paper-loop telemetry tests (threshold crossing payloads)
  3. Source artifacts for freqtrade insight ingestion

Profiles must be deterministic: same profile_id must always produce exactly
the same drawdown_path_pct regardless of runtime environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrawdownStressProfile:
    """Immutable, deterministic drawdown stress profile fixture."""

    profile_id: str
    description: str
    regime_tags: tuple[str, ...]
    # sequence of (iteration_index, drawdown_pct) — deterministic
    drawdown_path_pct: tuple[tuple[int, float], ...]
    # expected guardrail state at each path step: parallel to drawdown_path_pct
    expected_guardrail_states: tuple[str, ...]
    # aggregate DD band across the full path
    expected_dd_band: dict[str, float]
    capital: float = 10_000.0

    def pnl_at_step(self, step_index: int) -> float:
        """Return the (negative) PnL at the given path step as absolute dollars."""
        _, dd_pct = self.drawdown_path_pct[step_index]
        return -dd_pct * self.capital

    def max_drawdown_pct(self) -> float:
        return max(dd for _, dd in self.drawdown_path_pct)

    def as_source_artifact(self) -> dict[str, Any]:
        """Emit the profile as a freqtrade-compatible source artifact dict."""
        return {
            "schema": "moneyfan.drawdown.stress_profile.v1",
            "profile_id": self.profile_id,
            "description": self.description,
            "regime_tags": list(self.regime_tags),
            "drawdown_path_pct": [
                {"iteration": i, "drawdown_pct": dd}
                for i, dd in self.drawdown_path_pct
            ],
            "expected_guardrail_states": list(self.expected_guardrail_states),
            "expected_dd_band": self.expected_dd_band,
            "capital": self.capital,
        }


# ---------------------------------------------------------------------------
# Canonical profile registry
# ---------------------------------------------------------------------------

#: Guardrail default thresholds (mirrors TradingConfig defaults)
_WARN_PCT = 0.05
_DERISK_PCT = 0.08
_HALT_PCT = 0.12


def _band(lo: float, hi: float) -> dict[str, float]:
    return {"min_pct": lo, "max_pct": hi}


# Profile 1: benign — stays below warn threshold throughout
PROFILE_BENIGN = DrawdownStressProfile(
    profile_id="dd_stress_benign_v1",
    description="Benign session: drawdown stays below warn threshold throughout",
    regime_tags=("low_volatility", "trending_bull"),
    drawdown_path_pct=(
        (0, 0.00),
        (1, 0.01),
        (2, 0.02),
        (3, 0.03),
        (4, 0.04),
        (5, 0.04),
        (6, 0.03),
        (7, 0.02),
    ),
    expected_guardrail_states=(
        "normal", "normal", "normal", "normal",
        "normal", "normal", "normal", "normal",
    ),
    expected_dd_band=_band(0.00, 0.05),
)

# Profile 2: warn breach — crosses warn, stabilises, does not hit derisk
PROFILE_WARN_BREACH = DrawdownStressProfile(
    profile_id="dd_stress_warn_breach_v1",
    description="Warn breach: crosses 5% warn threshold then recovers",
    regime_tags=("medium_volatility", "ranging"),
    drawdown_path_pct=(
        (0, 0.00),
        (1, 0.03),
        (2, 0.05),   # warn trigger
        (3, 0.06),
        (4, 0.06),
        (5, 0.04),   # recovery
        (6, 0.03),
    ),
    expected_guardrail_states=(
        "normal", "normal", "warn", "warn", "warn", "normal", "normal",
    ),
    expected_dd_band=_band(0.05, 0.08),
)

# Profile 3: derisk path — crosses warn then derisk, does not halt
PROFILE_DERISK_PATH = DrawdownStressProfile(
    profile_id="dd_stress_derisk_path_v1",
    description="Derisk path: monotonic drawdown crossing warn and derisk thresholds",
    regime_tags=("high_volatility", "trending_bear"),
    drawdown_path_pct=(
        (0, 0.00),
        (1, 0.03),
        (2, 0.05),   # warn
        (3, 0.07),
        (4, 0.08),   # derisk
        (5, 0.09),
        (6, 0.09),
        (7, 0.08),
    ),
    expected_guardrail_states=(
        "normal", "normal", "warn", "warn", "derisk", "derisk", "derisk", "derisk",
    ),
    expected_dd_band=_band(0.08, 0.12),
)

# Profile 4: full halt path — monotonic through all thresholds
PROFILE_FULL_HALT = DrawdownStressProfile(
    profile_id="dd_stress_full_halt_v1",
    description="Full halt: monotonic drawdown escalating through all guardrail states to halt",
    regime_tags=("crash", "high_volatility", "trending_bear"),
    drawdown_path_pct=(
        (0, 0.00),
        (1, 0.04),
        (2, 0.05),   # warn
        (3, 0.07),
        (4, 0.08),   # derisk
        (5, 0.10),
        (6, 0.12),   # halt
        (7, 0.14),
    ),
    expected_guardrail_states=(
        "normal", "normal", "warn", "warn", "derisk", "derisk", "halt", "halt",
    ),
    expected_dd_band=_band(0.12, 0.20),
)

# Profile 5: oscillating — bounces around warn threshold (tests confirmation window)
PROFILE_OSCILLATING_WARN = DrawdownStressProfile(
    profile_id="dd_stress_oscillating_warn_v1",
    description="Oscillating warn: DD crosses warn threshold repeatedly without sustained breach",
    regime_tags=("choppy", "medium_volatility"),
    drawdown_path_pct=(
        (0, 0.00),
        (1, 0.05),   # warn touch
        (2, 0.04),   # recovery
        (3, 0.06),   # warn again
        (4, 0.04),   # recovery
        (5, 0.05),   # warn again
        (6, 0.03),
    ),
    expected_guardrail_states=(
        "normal", "warn", "normal", "warn", "normal", "warn", "normal",
    ),
    expected_dd_band=_band(0.04, 0.07),
)


#: Registry: profile_id -> DrawdownStressProfile
DRAWDOWN_STRESS_PROFILES: dict[str, DrawdownStressProfile] = {
    p.profile_id: p
    for p in (
        PROFILE_BENIGN,
        PROFILE_WARN_BREACH,
        PROFILE_DERISK_PATH,
        PROFILE_FULL_HALT,
        PROFILE_OSCILLATING_WARN,
    )
}

# Required schema keys for source artifact compatibility
SOURCE_ARTIFACT_REQUIRED_KEYS = (
    "schema",
    "profile_id",
    "description",
    "regime_tags",
    "drawdown_path_pct",
    "expected_guardrail_states",
    "expected_dd_band",
    "capital",
)
