# Track Spec: Runtime Drawdown Guardrails and Rollback-Safe Execution

## Overview
This track introduces explicit drawdown guardrails in moneyfan runtime loops (paper and live-preview) so the operator can cap risk escalation, halt unsafe behavior quickly, and preserve auditable evidence for post-run analysis and promotion decisions.

## Problem
The current runtime can enforce trade-level controls, but there is no single, auditable drawdown guardrail state machine that:
- reacts deterministically to equity drawdown transitions,
- enforces graded protective actions (warn, de-risk, hard stop),
- and records enough event evidence to explain why trading behavior changed.

Without this, strong backtest/paper metrics can still degrade into extended adverse runs before operator intervention.

## Goals
1. Add deterministic drawdown guardrail states to runtime.
2. Attach each guardrail action to clear event telemetry and artifacts.
3. Support fast operator rollback/safe-resume decisions after guardrail triggers.
4. Keep behavior compatible by default when guardrails are not enabled.

## Functional Requirements

### FR1. Guardrail State Machine
Runtime SHALL implement a drawdown state machine with at least:
- `normal`
- `warn`
- `de-risk`
- `halt`

State transitions SHALL be based on configurable drawdown thresholds and minimum confirmation windows to reduce noise-triggered oscillation.

### FR2. Protective Actions
Each non-normal state SHALL map to explicit runtime actions:
- `warn`: no execution suppression, event-only signaling
- `de-risk`: reduced trade throughput/risk budget (configurable)
- `halt`: hard execution stop until operator resume policy is satisfied

### FR3. Audit Events and Artifacts
Every transition/action SHALL be emitted to machine-readable artifacts and logs with:
- timestamp
- prior/new state
- drawdown value and threshold context
- runtime mode (paper/live-preview)
- action applied

### FR4. Operator Resume and Rollback Hooks
Runtime SHALL expose explicit resume semantics after `halt` and persist the reason/history needed to support rollback-safe operation.

### FR5. Brownfield Compatibility
Default behavior SHALL remain unchanged unless guardrail configuration is explicitly enabled.

## Acceptance Criteria
1. Runtime transitions through guardrail states deterministically under test fixtures.
2. `de-risk` and `halt` actions visibly alter execution behavior as configured.
3. Guardrail transition artifacts are emitted with complete required fields.
4. Resume behavior after `halt` is explicit, tested, and auditable.
5. Existing flows remain compatible when guardrails are disabled/defaulted.

## Out of Scope
- Exchange/broker transport changes
- Strategy alpha model retraining
- Freqtrade-side insight rendering changes
- Full production live trading automation

## Expected Files and Modules
- `/Users/jim/work/moneyfan/museum/run.py`
- `/Users/jim/work/moneyfan/museum/execution/`
- `/Users/jim/work/moneyfan/museum/runtime/`
- `/Users/jim/work/moneyfan/museum/tests/`
- `/Users/jim/work/moneyfan/museum/conductor/tracks/runtime_drawdown_guardrails_rollback_20260303/`
