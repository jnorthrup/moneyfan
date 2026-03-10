# Interface Matrix

## Purpose
Name the current interface seams between this repo and neighboring project surfaces from the standpoint of this repo as the superproject control point.

| Seam ID | Neighbor Surface | Local Owner | Current Contract | Evidence | Risk | Mitigation |
|---|---|---|---|---|---|---|
| S1 | Root Torch HRM training seed | `hrm/torch_hrm.py` | Minimal Torch model API plus `get_device()` | `hrm/torch_hrm.py`, `tests/test_hrm_outputs.py` | High | Fix focused test failures before promoting this as canonical. |
| S2 | Torch smoke artifact handoff | `tests/smoke_train.py` | Deterministic smoke run emits JSON result artifact | `tests/smoke_train.py`, `smoke_test_results_mps_20260310_153428.json` | Medium | Stabilize artifact schema and file naming. |
| S3 | Kotlin to ANE private bridge | `kotlin/native/ane/src/moneyfan_ane_bridge.m` and `kotlin/src/nativeMain/kotlin/borg/moneyfan/hrm/ane/HrmAne16x16Coder.kt` | Sample-only `1x1 -> 16x16` coder bridge | `kotlin/native/ane/src/moneyfan_ane_bridge.m`, `kotlin/src/nativeMain/kotlin/borg/moneyfan/hrm/ane/HrmAne16x16Coder.kt` | High | Keep isolated from the Torch training seed until a generic kernel contract exists. |
| S4 | Checkpoint artifact seam | `hrm/checkpoints/` | Feature schema, model config, objective config, and weights live together | `hrm/checkpoints/hrm_latest_feature_schema.json`, `hrm/checkpoints/hrm_latest_model_config.json`, `hrm/checkpoints/hrm_latest_objective_config.json`, `hrm/checkpoints/hrm_latest_weights.npz` | Medium | Define one checkpoint contract owned by the Torch track before export work begins. |
| S5 | Future Torch to Core ML export seam | none yet | Planned inference-only export surface | `conductor/tracks/torch_mps_ane_option_20260310/spec.md` | High | Do not open until S1 is clean and checkpoint ownership is explicit. |
| S6 | Neighbor ANE research seam | sibling ANE research project inferred from current conductor truth | Reverse-engineered ANE findings inform local mitigation, not direct product ownership | `conductor/tracks/torch_mps_ane_option_20260310/ane_gap_register.md` | Medium | Keep this as evidence input only unless a later explicit cross-repo slice is opened. |

## Highest-Risk Current Seam
- `S1` is the current highest-risk seam because the Torch seed surface exists but its focused tests are not yet clean.

## Local vs Neighbor Classification
- Local-only: `S1`, `S2`, `S3`, `S4`
- Future-neighbor: `S5`
- Sibling-repo evidence input: `S6`

## Next Product Slice Candidate
- Repair `S1` by reconciling `tests/test_hrm_outputs.py` with the actual behavior of `hrm/torch_hrm.py`, then formalize the Torch-to-checkpoint contract in `S4`.
