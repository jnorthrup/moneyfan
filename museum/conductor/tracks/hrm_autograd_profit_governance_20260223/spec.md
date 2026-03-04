# Track Spec: Profit-Driven MLX HRM Training Loop

## Track Summary
Implement the training-side bridge between HRM world-space representation quality and profit-relevant behavior by making the MLX HRM autograd objective explicit, configurable, and auditable.

This track is training-only. Runtime trading, simmer promotion, regime-manifest governance, and backtest orchestration are out of scope for this track.

## Scope Correction (Operator Direction)
This track originally included regime-governance and promotion tasks. Per operator direction, all non-training tasks are removed from this track.

The codebase may still contain prior runtime/governance work, but this track now governs only MLX HRM training changes.

## Problem Statement
The current MLX HRM trainer can update weights, but profit-relevant training behavior is still under-specified and under-audited:

1. The autograd objective structure is not fully visible in training artifacts.
2. Profit-oriented objective weights are not consistently traceable in all training outputs.
3. Trade-head optimization signal density and cost/turnover pressure are not strong enough or explicit enough for efficient improvement.
4. Training iteration baselines are not standardized for fast operator loops.

## Why This Track Matters (Money via Training)
If the HRM is expected to make money, the MLX training loop must:
- update the shared representation with useful gradients
- expose what objective actually drove those updates
- make profit-oriented tradeoffs (alpha vs churn/cost) tunable and measurable
- support fast iteration cycles that produce comparable artifacts

## Scope

### In Scope
- MLX HRM training objective telemetry in `train.py`
- Profit-oriented autograd objective controls (weights, penalties, regime weighting on training side)
- Training artifact metadata persistence for objective settings
- Trade-step scheduling / density improvements in the MLX HRM training loop
- Training-focused evidence capture and bounded smoke profiles

### Out of Scope (This Track)
- Simmer manager promotion logic
- Regime manifest validation for runtime promotion
- Calibration governor orchestration
- Paper/live runtime execution changes
- Broker/exchange integration work
- Backtest/reporting workflow changes unrelated to training outputs

## Functional Requirements

### FR1. Autograd Objective Decomposition Telemetry
Training outputs/checkpoints/results SHALL expose MLX HRM autograd objective components sufficient to answer:
- what scalar objective terms were active
- observed magnitude of each term (or proxy where not yet differentiable)
- which terms are direct losses vs proxies

### FR2. Profit-Oriented Objective Weight Controls
Training configuration SHALL expose objective weight controls for profit-relevant terms (world-model, trade-head, cost/turnover, regime weighting) and SHALL persist them in training artifacts.

### FR3. Artifact Traceability (Training)
Training checkpoints/results and deployable HRM training artifacts SHALL preserve objective configuration metadata so later analysis can tie model weights to the training objective used.

### FR4. Trade-Head Training Signal Density
The MLX HRM training loop SHALL support configurable trade-step scheduling that can increase alpha-learning signal density without requiring runtime changes.

### FR5. Cost/Turnover Objective Integration
The MLX trade objective SHALL support an explicit differentiable cost/turnover term (or a staged equivalent with clear telemetry during rollout).

### FR6. Training Evidence Baselines
The system SHALL provide bounded training configurations and artifact expectations to support rapid iteration and comparison of training runs.

## Non-Functional Requirements

### NFR1. Auditability
Training decisions and objective configuration must be reconstructable from JSON artifacts and CLI logs.

### NFR2. Iteration Speed
New training telemetry/controls should preserve fast local experimentation and bounded smoke runs.

### NFR3. Brownfield Compatibility
Default training behavior should remain compatible unless new objective controls are explicitly changed.

### NFR4. MLX-First Practicality
Changes should target the MLX HRM path directly and avoid broad refactors that do not improve training throughput or signal quality.

## Design Notes

### Autograd Strategy (Training-Only)
Autograd is the mechanism that converts world-space understanding into weight updates. The track focuses on:
- making the optimized scalar objective visible
- making objective tradeoffs controllable
- improving the density/quality of training gradients for the trade heads

### Staged Delivery
Use staged rollout for objective upgrades:
1. telemetry + config controls (done)
2. stronger trade-step scheduling and density
3. explicit cost/turnover differentiable term
4. training-side regime weighting controls

## Acceptance Criteria
1. Training outputs record autograd objective decomposition and objective weights.
2. Training checkpoints/results and saved HRM artifacts preserve objective configuration metadata.
3. MLX training supports configurable trade-step scheduling improvements with telemetry.
4. Cost/turnover autograd term is implemented (or staged with explicit telemetry) and auditable.
5. Bounded MLX training smoke profiles and expected artifacts are documented for operator use.
6. Tests cover new training telemetry/config serialization and new training objective controls.

## Risks and Mitigations
- **Risk:** Over-tuning objective controls before trade-step signal density improves.
  - **Mitigation:** Prioritize scheduling/density telemetry before complex weighting changes.
- **Risk:** Telemetry/proxy terms create false confidence about actual gradients.
  - **Mitigation:** Label direct vs proxy terms explicitly and move terms in-graph incrementally.
- **Risk:** Training changes reduce iteration speed.
  - **Mitigation:** Maintain bounded smoke profiles and keep MLX-first patches local.

## Target Files (Expected)
- `/Users/jim/work/moneyfan/train.py`
- `/Users/jim/work/moneyfan/hrm/hierarchical_codec_mlx.py`
- `/Users/jim/work/moneyfan/tests/test_training_objective_telemetry.py`
- new/updated training-focused tests under `/Users/jim/work/moneyfan/tests/`
