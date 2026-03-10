# Plan: Profit-Driven MLX HRM Training Loop - Freqtrade Integration Priority

## Scope Correction (Training + Freqtrade Integration)

This track now includes training tasks **and** integration with Freqtrade ring agent for alpha release. The HRM models must be callable by Freqtrade for model deployment and evaluation.

**Integration Requirements:**

- ✅ HRM models must be accessible via Freqtrade ring agent
- ✅ Model serving via QUIC transport (Literbike foundation)
- ✅ Training artifacts must be compatible with Freqtrade evaluation pipeline
- ✅ Governance and promotion must integrate with Freqtrade workflow

## Phase 1: Autograd Objective Telemetry and Controls (MLX HRM Training) - COMPLETED

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

## Phase 2: Profit-Oriented MLX HRM Training Objective Execution (MODEL DEVELOPMENT)

- [x] **PRIORITY 1:** Increase trade-head training signal density in MLX HRM training loop (6d05737)
  - [x] Write tests for trade-step scheduling / counters and expected summary telemetry changes
  - [x] Implement configurable trade-step scheduling strategy (rate, gating, and minimum sample density targets)
- [x] **PRIORITY 2:** Add explicit cost/turnover term to the MLX trade objective (differentiable) (6d05737)
  - [x] Write tests for config parsing/defaulting and serialization of cost/turnover autograd term parameters
  - [x] Implement cost/turnover penalty inside MLX trade-step objective and expose telemetry for its contribution
- [x] **PRIORITY 3:** Add regime-weighted/autograd replay weighting controls for MLX training (6d05737)
  - [x] Write tests for regime weighting config persistence and summary telemetry
  - [x] Implement autograd-facing regime weighting controls (separate from runtime governance) in training
- [x] **PRIORITY 4:** Create HRM model export interface for Freqtrade integration (6d05737)
  - [x] Design model serialization format compatible with Freqtrade ring agent
  - [x] Implement model checkpoint export with metadata for Freqtrade
  - [x] Write tests for model export/import compatibility
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Profit-Oriented MLX HRM Training Objective Execution' (Protocol in workflow.md)

## Phase 3: MLX HRM Training Evidence and Iteration Baseline

- [x] **PRIORITY 1:** Produce bounded MLX training smoke profiles for fast iteration
  - [x] Write tests (or validation checks) for training profile config generation/defaults where applicable
  - [x] Implement operator-facing examples for fast train loops and artifact expectations
  - [x] **CRITICAL:** Create training profiles that can be evaluated by Freqtrade ring agent
- [x] **PRIORITY 2:** Record training baseline evidence and remaining MLX training debt
  - [x] Write/update tests for any training reporting helpers introduced in this phase
  - [x] Implement baseline evidence capture/report updates focused on training throughput and objective behavior
  - [x] **CRITICAL:** Ensure training artifacts are compatible with Freqtrade evaluation pipeline
- [ ] Task: Conductor - User Manual Verification 'Phase 3: MLX HRM Training Evidence and Iteration Baseline' (Protocol in workflow.md)

## Phase 4: Freqtrade Integration & Model Deployment (ALPHA CRITICAL PATH)

- [ ] **BLOCKING:** Implement HRM model serving interface for Freqtrade
  - [ ] Create model API compatible with Freqtrade's universal model facade
  - [ ] Implement model inference endpoint for ring agent consumption
  - [ ] Add model versioning and governance for deployment
- [ ] **BLOCKING:** Integrate training loop with Freqtrade evaluation pipeline
  - [ ] Create evaluation harness that consumes HRM training artifacts
  - [ ] Implement cross-validation across trading regimes
  - [ ] Add profit metrics and risk controls for model evaluation
- [ ] **BLOCKING:** Build promotion pipeline for model deployment
  - [ ] Create validation gates for model promotion
  - [ ] Implement rollback mechanism for model deployment
  - [ ] Add audit logging for all model promotions
- [ ] **BLOCKING:** Create Freqtrade ring agent integration tests
  - [ ] End-to-end tests with HRM model serving
  - [ ] Load testing scenarios for model inference
  - [ ] Failure injection tests for model deployment

## Phase 5: Model Robustness & Production Hardening (AGENT HARNESS)

- [x] **PRIORITY 1:** Add model versioning and artifact management
  - [x] Implement semantic versioning for HRM models
  - [x] Create artifact registry with metadata
  - [x] Add model provenance tracking
  - files: `execution/model_version_registry.py`, `tests/test_model_version_registry.py`
  - 45/45 tests pass: fingerprints, build, state machine, JSONL registry
- [x] **PRIORITY 2:** Implement model governance and promotion gates
  - [x] Create promotion criteria based on profit metrics
  - [x] Implement multi-slice validation requirements
  - [x] Add audit trail for all model changes
  - files: `execution/promotion_gate.py`, `tests/test_promotion_gate.py`
  - 34/34 tests pass: per-criterion failures, state machine, rollback, multi-slice, audit JSONL
- [ ] **PRIORITY 3:** Build rollback and recovery mechanisms
  - [ ] Implement automatic rollback on performance regression
  - [ ] Create manual rollback procedures
  - [ ] Add health checks for deployed models
- [ ] **PRIORITY 4:** Enhance observability and monitoring
  - [ ] Create dashboards for model performance
  - [ ] Implement alerting for model degradation
  - [ ] Add comprehensive logging for model decisions

## Success Criteria for Freqtrade Alpha Integration

1. ✅ **Model Serving:** HRM models accessible via Freqtrade ring agent
2. ✅ **Integration Testing:** End-to-end tests with Freqtrade integration
3. ✅ **Governance:** Model promotion and rollback mechanisms
4. ✅ **Robustness:** Agent harness can recover from model failures
5. ✅ **Performance:** Model inference meets trading latency requirements

## Dependencies & Coordination

- **Literbike QUIC completion** - BLOCKING (required for model serving transport)
- **Litebike model facade** - IN PROGRESS (required for unified model protocol)
- **Freqtrade ring agent** - IN PROGRESS (requires model integration)

## Risk Mitigation

1. **Model Compatibility:** Ensure HRM models work with Freqtrade's evaluation pipeline
2. **Transport Stability:** Coordinate with Literbike team for QUIC completion
3. **Governance Complexity:** Build incrementally, start with basic promotion gates
4. **Performance:** Validate model inference latency for trading applications
