# Plan: Runtime Drawdown Guardrails and Rollback-Safe Execution

## Purpose

Add an auditable drawdown guardrail layer to runtime so adverse equity behavior can trigger deterministic protective actions and safer operator recovery.

## Scope Capture / Planning Handoff

## Concrete Features

- Drawdown guardrail state machine integrated into runtime loops.
- Configurable threshold policy for `warn`/`de-risk`/`halt`.
- Deterministic protective actions tied to state.
- Transition artifact emission for audit and post-run review.
- Explicit resume semantics after halted execution.

## Author Context Hints

- `run.py` likely owns the top-level trading loop entry points.
- `execution/` likely owns order/trade dispatch controls.
- `runtime/` likely hosts runtime artifacts and loop-level state outputs.
- Existing tests under `tests/` should define fixture style and naming.

## Candidate Workstreams

- Guardrail policy model + config parsing/defaults.
- Runtime transition engine + action hooks.
- Artifact/log emission + schema tests.
- Resume policy and operator runbook validation.

## First Maneuver (Line It Up / Knock It Down)

Pin down one deterministic transition path (`normal` -> `warn` -> `de-risk` -> `halt`) with a minimal fixture and test harness first. This aligns state semantics before wiring broad runtime side effects.

Suggested context to load first:

- `run.py`
- `execution/` modules currently controlling trade rate/size/dispatch
- existing runtime artifact writers
- nearest runtime test files

## Unknowns / Decisions / Risks

- Decision: absolute drawdown thresholds vs regime-relative thresholds for v1.
- Decision: whether `de-risk` scales order size, reduces signal count, or both.
- Risk: threshold jitter causing oscillation without confirmation windows.
- Risk: interaction with existing veto/override controls causing policy conflicts.
- Risk: halt/resume semantics may require operator UX conventions not yet standardized.

## Deferred / Out-of-Scope for This Track

- Dynamic threshold tuning by volatility regime.
- Freqtrade-side visualization/dashboard implementation.
- Autonomous resume logic without operator acknowledgment.
- Live broker kill-switch integration beyond current runtime boundaries.

Detailed `100%` slices and bounded-corpus estimates will be declared at execution start in `conductor implement`.

## Phase 1: Guardrail Policy and Schema

- [x] Task: Define guardrail config schema and defaults
  - files: `run.py`, runtime config modules
  - adjacent nexus: CLI arg parsing, config serialization helpers
- [x] Task: Implement state transition rules and confirmation-window logic
  - files: runtime loop/state modules
  - adjacent nexus: equity/drawdown metric producers
- [x] Task: Add unit tests for deterministic transition behavior
  - files: `tests/`
  - adjacent nexus: existing runtime/policy fixture patterns

## Phase 2: Runtime Enforcement Hooks

- [x] Task: Map guardrail states to runtime actions (`warn`, `de-risk`, `halt`)
  - files: `execution/guardrail_actions.py`, `run.py`
  - adjacent nexus: GuardrailActionMapper wired into TradingEngine; _update_guardrail_action,_get_effective_top_k,_get_effective_signal_threshold,_get_effective_position_size_scale, _should_allow_new_entries all live
- [x] Task: Ensure guardrail actions are mode-aware (`paper` vs `live-preview`)
  - files: runtime mode handlers
  - adjacent nexus: guardrail_enabled config flag; default disabled; CLI flags wired in main()
- [x] Task: Add tests validating behavior changes under each guardrail state
  - files: `tests/test_runtime_drawdown_guardrails.py`
  - adjacent nexus: 19/19 tests pass covering action wiring, top-k scaling, threshold raising, position size scaling, halt blocking

## Phase 3: Auditability and Resume Safety

- [x] Task: Emit transition/action events to runtime artifacts with stable schema
  - files: `run.py` `_emit_guardrail_event`, `runtime/guardrail_events.jsonl`
  - adjacent nexus: JSONL path, moneyfan.runtime.guardrail.event.v1 schema emitted and tested
- [x] Task: Implement explicit halt-resume semantics and persistence
  - files: `run.py` `_save_state`, `_load_state`
  - adjacent nexus: guardrail_state / candidate window fields survive crash-safe restart; invalid values silently default
- [x] Task: Add artifact completeness tests and resume-flow tests
  - files: `tests/test_runtime_drawdown_guardrails.py`
  - adjacent nexus: 24/24 tests pass; schema completeness, save/load round-trip, invalid-state fallback, halt-resume all covered

## Phase 4: Operator Verification and Hardening

- [x] Task: Add operator runbook for guardrail-triggered sessions
  - files: `conductor/tracks/runtime_drawdown_guardrails_rollback_20260303/operator_runbook.md`
  - adjacent nexus: state reading, halt-resume procedure, threshold adjustment CLI, validation commands
- [x] Task: Run targeted validation command set and record expected outcomes
  - files: `operator_runbook.md` (validation section)
  - adjacent nexus: `PYTHONPATH=... pytest tests/test_runtime_drawdown_guardrails.py` — 24/24 pass
- [x] Task: Capture rollout notes and fallback toggles for safe adoption
  - files: `operator_runbook.md` (rollout section)
  - adjacent nexus: guardrails off-by-default; backward-compat state loading; no broker API side effects

## 100% Slices (Zero-Discovery Candidate Seeds)

- [x] Slice: Add a minimal transition test fixture covering one monotonic drawdown path.
- [x] Slice: Add schema contract test for one guardrail transition artifact event.
- [x] Slice: Add config-default test ensuring guardrails are disabled by default.

## Decision Queue

- [x] Decide initial default threshold set for `warn`/`de-risk`/`halt`.
  - Resolved: warn=5%, derisk=8%, halt=12% (configurable via CLI flags)
- [x] Decide first-pass `de-risk` behavior (size scaling, frequency reduction, or hybrid).
  - Resolved: hybrid — 50% position size scale + 50% top-k reduction + +10% confidence boost
- [ ] Decide required operator acknowledgment mechanism for resume.
  - Current: `--ignore-saved-halt-state` flag required to override a saved halt

## Backlog

- [ ] Add volatility-adaptive guardrail thresholds.
- [ ] Add per-strategy guardrail overrides.
- [ ] Add guardrail outcome comparison reports across paper runs.
