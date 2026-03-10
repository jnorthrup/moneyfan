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

- [ ] Task: Map guardrail states to runtime actions (`warn`, `de-risk`, `halt`)
  - files: `execution/`, `run.py`
  - adjacent nexus: trade gating/veto or dispatch controls
- [ ] Task: Ensure guardrail actions are mode-aware (`paper` vs `live-preview`)
  - files: runtime mode handlers
  - adjacent nexus: mode flags and mode-specific execution adapters
- [ ] Task: Add tests validating behavior changes under each guardrail state
  - files: `tests/`
  - adjacent nexus: current paper/live-preview simulation tests

## Phase 3: Auditability and Resume Safety

- [ ] Task: Emit transition/action events to runtime artifacts with stable schema
  - files: `runtime/`, artifact emitters
  - adjacent nexus: existing `metrics.json`/`trades.json` writers
- [ ] Task: Implement explicit halt-resume semantics and persistence
  - files: `run.py`, runtime state persistence modules
  - adjacent nexus: session lifecycle and startup restore logic
- [ ] Task: Add artifact completeness tests and resume-flow tests
  - files: `tests/`
  - adjacent nexus: artifact validation and replay-style tests

## Phase 4: Operator Verification and Hardening

- [ ] Task: Add operator runbook for guardrail-triggered sessions
  - files: this track folder and/or runtime docs
  - adjacent nexus: existing daily operator runbooks
- [ ] Task: Run targeted validation command set and record expected outcomes
  - files: test/runtime command docs
  - adjacent nexus: `workflow.md` verification protocol
- [ ] Task: Capture rollout notes and fallback toggles for safe adoption
  - files: this track folder
  - adjacent nexus: config flags and release toggles

## 100% Slices (Zero-Discovery Candidate Seeds)

- [x] Slice: Add a minimal transition test fixture covering one monotonic drawdown path.
- [x] Slice: Add schema contract test for one guardrail transition artifact event.
- [x] Slice: Add config-default test ensuring guardrails are disabled by default.

## Decision Queue

- [ ] Decide initial default threshold set for `warn`/`de-risk`/`halt`.
- [ ] Decide first-pass `de-risk` behavior (size scaling, frequency reduction, or hybrid).
- [ ] Decide required operator acknowledgment mechanism for resume.

## Backlog

- [ ] Add volatility-adaptive guardrail thresholds.
- [ ] Add per-strategy guardrail overrides.
- [ ] Add guardrail outcome comparison reports across paper runs.
