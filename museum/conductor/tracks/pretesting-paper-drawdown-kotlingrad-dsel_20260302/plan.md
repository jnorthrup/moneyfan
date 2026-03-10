# Plan: Moneyfan Pretesting + Paper Testing Drawdown Feeds (Freqtrade Insights Sink)

## Purpose

Prepare and validate moneyfan-side pretesting/paper-testing drawdown inputs so that finalized insights are centralized in `../freqtrade/conductor/insights/`.

## Scope

- Produce deterministic drawdown stress profiles for training/pretesting
- Align paper-testing telemetry shape with drawdown reconciliation needs
- Emit handoff artifacts consumable by freqtrade insight writers
- Keep insights ownership in `freqtrade` (moneyfan provides source artifacts)

## Bounded Corpus

- `train.py`
- `run.py`
- `execution/`
- `runtime/`
- `tests/`
- `conductor/tracks/pretesting-paper-drawdown-kotlingrad-dsel_20260302/`

## Phase 1: Pretesting Drawdown Inputs

- [x] Task: Define deterministic pretesting stress profiles for drawdown-focused evaluation
  - files: `execution/drawdown_stress_profiles.py`
  - 5 profiles: benign, warn_breach, derisk_path, full_halt, oscillating_warn
- [x] Task: Add profile metadata (`profile_id`, regime tags, expected DD bands)
  - files: `execution/drawdown_stress_profiles.py`
  - each profile has profile_id, regime_tags, expected_dd_band, expected_guardrail_states
- [x] Task: Add tests validating profile determinism and schema stability
  - files: `tests/test_drawdown_stress_profiles.py`
  - 79/79 tests pass: schema, determinism, DD-band bounds, threshold alignment
- [x] Task: Emit source artifacts intended for `freqtrade` insight ingestion
  - files: `execution/drawdown_stress_profiles.py` `as_source_artifact()`
  - schema: moneyfan.drawdown.stress_profile.v1; JSON-serializable; freqtrade-compatible

## Phase 2: Paper Testing Drawdown Telemetry

- [x] Task: Standardize paper-loop drawdown telemetry event shape (`signal_id`, DD%, threshold state)
  - files: `execution/paper_drawdown_telemetry.py`
  - schema: moneyfan.paper.drawdown.telemetry.v1; REQUIRED_TELEMETRY_KEYS constant
- [x] Task: Add reconciliation metadata needed by freqtrade report generation
  - files: `execution/paper_drawdown_telemetry.py`
  - fields: guardrail_action_active, position_size_scale, new_entries_allowed, effective_top_k, profile_id
- [x] Task: Add tests for telemetry completeness and compatibility
  - files: `tests/test_paper_drawdown_telemetry.py`
  - 49/49 tests pass: schema, crossing payloads, validate(), helpers, stress profile replay
- [ ] Task: Emit handoff artifacts for `../freqtrade/conductor/insights/paper/`
  - blocked: freqtrade insights directory ownership and handoff batch cadence not yet decided

## Phase 3: Kotlingrad DSEL Source Alignment

- [ ] Task: Add source manifest fields that reference Kotlingrad expression IDs where available
- [ ] Task: Add compatibility checks for expression-id stability across runs
- [ ] Task: Add smoke test for source manifest handoff into freqtrade-side aggregator

## 100% Slices (Zero-Discovery)

- [x] Slice: Add one deterministic drawdown stress profile fixture + validation test
  - delivered: 5 profiles + 79 tests in execution/drawdown_stress_profiles.py and tests/test_drawdown_stress_profiles.py
- [x] Slice: Add one paper telemetry schema test including threshold crossing payloads
  - delivered: execution/paper_drawdown_telemetry.py + tests/test_paper_drawdown_telemetry.py (49/49 pass)
  - crossing payloads at all 4 states; stress-profile path replay; validate() function
- [ ] Slice: Add one source-manifest fixture with stable expression IDs

## Decision Queue

- [x] Decide minimum telemetry fields required before freqtrade accepts a handoff batch
  - Resolved: REQUIRED_TELEMETRY_KEYS in paper_drawdown_telemetry.py (15 fields)
- [ ] Decide whether moneyfan should include optional explanatory fields or keep a strict minimal schema
  - Current: optional fields (profile_id, effective_top_k) present only when supplied; strict core required
- [ ] Decide update cadence for source artifact publishing (per run vs daily batch)

## Backlog

- [ ] Add adaptive drawdown profile generation based on prior run outcomes
- [ ] Add richer regime clustering tags for stress-profile selection
- [ ] Add data-quality score in source artifacts before handoff
