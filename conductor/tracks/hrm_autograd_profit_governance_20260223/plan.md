# Plan: Profit-Driven MLX HRM Training Loop

## Scope Correction (Training-Only)

Per operator direction, this track now includes only tasks directly related to training the MLX HRM.

De-scoped from this plan (no longer tracked as tasks here):
- regime manifest / simmer promotion / calibration-governor integration work
- backtest/paper runtime governance tasks
- runbook and cross-system validation tasks not specific to training

Historical note:
- Some non-training work was already implemented before this scope correction. It remains in the codebase but is not part of this track plan anymore.

## Phase 1: Autograd Objective Telemetry and Controls (MLX HRM Training)

- [x] Task: Expose autograd objective decomposition in training outputs
    - [x] Write tests for training result/checkpoint serialization of objective components
    - [x] Implement objective decomposition telemetry (world-model, trade-head, cost/turnover, regime weighting terms)
- [x] Task: Add auditable profit-oriented autograd objective weight controls
    - [x] Write tests for config parsing/defaulting and persistence of objective weight parameters
    - [x] Implement configuration fields and logging for objective weight controls in training outputs / CLI
- [x] Task: Link objective configuration to saved artifacts for downstream training governance
    - [x] Write tests ensuring objective config metadata is preserved in checkpoint/result artifacts
    - [x] Implement artifact metadata persistence for objective configuration and relevant autograd settings
- [~] Task: Conductor - User Manual Verification 'Phase 1: Autograd Objective Telemetry and Controls (MLX HRM Training)' (Protocol in workflow.md)

## Phase 2: Profit-Oriented MLX HRM Training Objective Execution

- [ ] Task: Increase trade-head training signal density in MLX HRM training loop
    - [ ] Write tests for trade-step scheduling / counters and expected summary telemetry changes
    - [ ] Implement configurable trade-step scheduling strategy (rate, gating, and minimum sample density targets)
- [ ] Task: Add explicit cost/turnover term to the MLX trade objective (differentiable)
    - [ ] Write tests for config parsing/defaulting and serialization of cost/turnover autograd term parameters
    - [ ] Implement cost/turnover penalty inside MLX trade-step objective and expose telemetry for its contribution
- [ ] Task: Add regime-weighted/autograd replay weighting controls for MLX training
    - [ ] Write tests for regime weighting config persistence and summary telemetry
    - [ ] Implement autograd-facing regime weighting controls (separate from runtime governance) in training
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Profit-Oriented MLX HRM Training Objective Execution' (Protocol in workflow.md)

## Phase 3: MLX HRM Training Evidence and Iteration Baseline

- [ ] Task: Produce bounded MLX training smoke profiles for fast iteration
    - [ ] Write tests (or validation checks) for training profile config generation/defaults where applicable
    - [ ] Implement operator-facing examples for fast train loops and artifact expectations
- [ ] Task: Record training baseline evidence and remaining MLX training debt
    - [ ] Write/update tests for any training reporting helpers introduced in this phase
    - [ ] Implement baseline evidence capture/report updates focused on training throughput and objective behavior
- [ ] Task: Conductor - User Manual Verification 'Phase 3: MLX HRM Training Evidence and Iteration Baseline' (Protocol in workflow.md)
