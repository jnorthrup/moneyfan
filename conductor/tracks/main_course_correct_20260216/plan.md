# Plan: Main track course correction: convergence/perturb/repeat optimization with alpha, PnL, and standard trading metrics

Track ID: `main_course_correct_20260216`

---

## Phase 1: Objective Re-baseline

- [x] Task: Define canonical scorecard schema for alpha, PnL, standard metrics [a099d95]
- [x] Task: Add metric computation module and test coverage [1a06abc]
    - [ ] Write failing tests for win_rate and convergence computation (Red)
    - [ ] Implement metric pipeline (Green)
    - [ ] Add regression tests for alpha/PnL consistency
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

---

## Phase 2: Convergence Engine Hardening

- [x] Task: Implement convergence detection over model signal histories
    - [x] Write failing tests for agreement/confidence convergence signals (Red)
    - [x] Implement convergence scoring and thresholds (Green)
    - [x] Expose convergence outputs to controller + oversight surfaces
- [x] Task: Integrate convergence metrics into decision payloads
    - [x] Add convergence fields to normalized decision/event schema
    - [x] Validate schema compatibility with existing consumers
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

---

## Phase 3: Perturbation Policy Loop

- [ ] Task: Implement bounded perturbation policy operators
    - [ ] Add perturbation primitives (weights, thresholds, intensity)
    - [ ] Enforce fiduciary bounds during perturbation
- [ ] Task: Add retain/revert decisioning per cycle
    - [ ] Write failing tests for accept/reject based on alpha/PnL/win_rate deltas (Red)
    - [ ] Implement cycle comparator and rollback hooks (Green)
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

---

## Phase 4: Coinbase Realtime + Shadow Evaluation

- [ ] Task: Validate realtime IO quality gates
    - [ ] Sequence-gap diagnostics, reconnect behavior, heartbeat continuity
    - [ ] Event normalization parity checks
- [ ] Task: Run shadow-mode evaluation for convergence/perturb/repeat
    - [ ] Report alpha/PnL/win_rate by cycle and by symbol universe
    - [ ] Document failure modes and mitigation actions
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

---

## Phase 5: Main Track Replacement and Cleanup

- [ ] Task: Mark superseded track status and link migration notes
    - [ ] Update `conductor/tracks.md` ordering/status annotations
    - [ ] Update old track metadata to `superseded` (or `archived` with note)
- [ ] Task: Final verification and handoff
    - [ ] Ensure all planned tasks are reflected in track artifacts
    - [ ] Publish final course-correction summary
- [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)

---
