# ANE Gap Register

## Purpose
Record the current gaps between the desired Torch-first HRM path and the verified execution surfaces available in this repo.

## Runtime Route
- Delegated runtime route for this track: `kilo`.

## Verified Current Surfaces
- `kotlin/native/ane/src/moneyfan_ane_bridge.m`: private ANE bridge for a sample `1x1 -> 16x16` coder.
- `kotlin/src/nativeMain/kotlin/borg/moneyfan/hrm/ane/HrmAne16x16Coder.kt`: Kotlin wrapper around the sample ANE bridge.
- `kotlin/src/commonMain/kotlin/borg/moneyfan/hrm/ane/SampleNet16x16.kt`: tiny convergence harness over the sample coder.
- `hrm/torch_hrm.py`: minimal Torch HRM module now present in the root-level repo tree.
- `tests/smoke_train.py`: deterministic smoke harness present and runnable.
- `tests/test_hrm_outputs.py`: focused tests present, but not yet clean.
- `hrm/checkpoints/`: checkpoint artifacts remain present for future bridge work.

## Gap Register

### G1. Torch HRM source exists, but its focused test surface is not yet clean
- Severity: blocking
- Evidence: `hrm/torch_hrm.py` and `tests/test_hrm_outputs.py` exist, but the focused pytest run still has two failures.
- Impact: the Torch surface cannot yet be treated as a stable canonical seed for export or neighbor integration.
- Mitigation: fix the unused-gradient expectation around `embedding.hash_weights` and replace the brittle loss-decrease assertion with a deterministic, stable criterion.

### G2. No device-routing layer exists for Torch backends
- Severity: medium
- Evidence: `hrm/torch_hrm.py` now exposes `get_device()`, but coverage is still limited to the current focused tests and smoke script.
- Impact: backend routing exists, but is not yet hardened as a reusable interface contract.
- Mitigation: keep the routing helper, add a dedicated interface test, and route all future Torch entrypoints through one owner.

### G3. No Core ML export surface exists for the current root-level repo
- Severity: high
- Evidence: no Torch module or export helper is present in the current tree.
- Impact: there is no inference parity route to ANE-backed Core ML.
- Mitigation: add export only after the first Torch module is stable enough to serialize.

### G4. Current ANE bridge is a sample harness, not a training backend
- Severity: high
- Evidence: `kotlin/native/ane/src/moneyfan_ane_bridge.m` compiles a fixed sample conv and `kotlin/src/commonMain/kotlin/borg/moneyfan/hrm/ane/SampleNet16x16.kt` trains only a trivial head.
- Impact: it cannot serve as a general Torch autograd backend.
- Mitigation: treat ANE as experimental inference acceleration until a generic kernel interface exists.

### G5. Weight mutability and symmetric training controls are still missing on the ANE side
- Severity: medium
- Evidence: prior reverse-engineering results already recorded for this track show compile-baked weights and missing direct Torch-like runtime semantics.
- Impact: direct ANE step-wise training remains speculative.
- Mitigation: use Torch training plus `mps` fallback first, then export static inference graphs to Core ML/ANE where justified.

### G6. Track ownership needed correction from stale repo assumptions
- Severity: medium
- Evidence: earlier planning assumed a larger `museum/` Python tree that is not present in this repo checkout.
- Impact: stale file ownership would send delegates to nonexistent paths.
- Mitigation: this track now treats root-level `hrm/` and `kotlin/` as the canonical owners.

## Immediate Next Slice
- Fix the two failing focused Torch tests.
- Keep ANE integration out of the test-repair slice.
- Only expand toward Core ML and neighbor interfaces after the Torch seed surface is clean.

## Update Checklist
- Confirm root-level owners still exist and remain canonical.
- Re-check whether Torch source has been added before reopening G1.
- Re-run the focused Torch pytest surface before downgrading G1.
- Re-check whether Core ML export exists before reopening G3.
- Re-check whether the ANE bridge has moved beyond the sample coder before downgrading G4.
