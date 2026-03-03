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

- [ ] Task: Define deterministic pretesting stress profiles for drawdown-focused evaluation
- [ ] Task: Add profile metadata (`profile_id`, regime tags, expected DD bands)
- [ ] Task: Add tests validating profile determinism and schema stability
- [ ] Task: Emit source artifacts intended for `freqtrade` insight ingestion

## Phase 2: Paper Testing Drawdown Telemetry

- [ ] Task: Standardize paper-loop drawdown telemetry event shape (`signal_id`, DD%, threshold state)
- [ ] Task: Add reconciliation metadata needed by freqtrade report generation
- [ ] Task: Add tests for telemetry completeness and compatibility
- [ ] Task: Emit handoff artifacts for `../freqtrade/conductor/insights/paper/`

## Phase 3: Kotlingrad DSEL Source Alignment

- [ ] Task: Add source manifest fields that reference Kotlingrad expression IDs where available
- [ ] Task: Add compatibility checks for expression-id stability across runs
- [ ] Task: Add smoke test for source manifest handoff into freqtrade-side aggregator

## 100% Slices (Zero-Discovery)

- [ ] Slice: Add one deterministic drawdown stress profile fixture + validation test
- [ ] Slice: Add one paper telemetry schema test including threshold crossing payloads
- [ ] Slice: Add one source-manifest fixture with stable expression IDs

## Decision Queue

- [ ] Decide minimum telemetry fields required before freqtrade accepts a handoff batch
- [ ] Decide whether moneyfan should include optional explanatory fields or keep a strict minimal schema
- [ ] Decide update cadence for source artifact publishing (per run vs daily batch)

## Backlog

- [ ] Add adaptive drawdown profile generation based on prior run outcomes
- [ ] Add richer regime clustering tags for stress-profile selection
- [ ] Add data-quality score in source artifacts before handoff
