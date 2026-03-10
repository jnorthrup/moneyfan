# Plan: Torch Option with MPS Fallback and ANE Gap Mapping

## Phase 1: Torch Training Surface
- [x] Task: Create the minimal HRM Torch model entrypoints and expected inputs.
- [x] Task: Implement a Torch training smoke profile with fixed seed.
- [ ] Task: Add unit tests for the smoke profile output shapes and loss sanity.
- [ ] Task: Fix the focused Torch tests so gradient expectations and loss-reduction checks match actual model behavior.

## Phase 2: MPS Device Routing
- [ ] Task: Add device routing helper that selects `mps` when available.
- [ ] Task: Add tests for device selection on non-MPS hosts.
- [ ] Task: Document MPS limitations in the gap register.

## Phase 3: Core ML Export and Parity Check
- [ ] Task: Implement a Torch to Core ML export path for the selected HRM module.
- [ ] Task: Add a lightweight inference parity check against Torch output.
- [ ] Task: Record export constraints and unsupported ops in the gap register.

## Phase 4: ANE Gap Register
- [x] Task: Create `ane_gap_register.md` with current known gaps and mitigations.
- [ ] Task: Wire a simple update checklist into the track artifacts.

## Active Slice
- [x] Slice: Record the current Torch/MPS/Core ML/ANE gap register against the actual root-level repo layout.
- [x] Slice: Create the first minimal Torch HRM module and deterministic smoke harness.
- [ ] Slice: Reconcile the new Torch module against failing focused tests and then promote it as the canonical training seed surface.

## Delegation Status
- [x] Attempted delegated `kilo` execution for the first product slice on 2026-03-10.
- [x] Initial worker drifted in discovery and produced no immediate bounded product file changes.
- [x] Product artifacts later appeared in `hrm/` and `tests/`; authenticity verified by direct file inspection and focused commands.
- [ ] Re-issue the next slice with a tighter contract to fix the remaining focused test failures.

## Verification Evidence
- [x] `python3 tests/smoke_train.py` passed and emitted `smoke_test_results_mps_20260310_153428.json`.
- [x] `python3 -m pytest tests/test_hrm_outputs.py -q` ran and exposed two failures that must be fixed before Phase 1 is closed.
