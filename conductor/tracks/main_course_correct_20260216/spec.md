# Spec: Main Track Course Correction — Convergence/Perturb/Repeat

Track intent: replace the prior main track with a tighter optimization objective and execution loop focused on market momentum capture quality.

## Overview
This track redefines the primary system objective to:
1. Identify convergence in momentum-aligned model signals.
2. Apply controlled perturbations to improve decision quality.
3. Repeat adaptively with measurable improvement.

The system remains candle-core and Coinbase-realtime driven, with HRM as meta-controller over multi-model signal histories.

## Functional Requirements

1. Objective Framework
- The primary optimization objective MUST report and optimize:
  - alpha
  - PnL
  - KD ratio
- KD ratio definition for this project:
  - kill = positive momentum regime captures
  - death = failed momentum captures

2. Convergence Engine
- The pipeline MUST compute convergence scores from model-signal agreement and confidence over rolling windows.
- Convergence signals MUST be available both for execution decisions and oversight diagnostics.

3. Perturbation Policy
- The controller MUST support bounded perturbations to model weighting, allocation intensity, and decision thresholds.
- Perturbations MUST be policy-limited by fiduciary/risk constraints.

4. Repeat/Adapt Loop
- The system MUST execute iterative cycles:
  - detect convergence
  - perturb policy
  - evaluate delta on alpha/PnL/KD
  - retain or revert changes
- The loop MUST be trackable with experiment metadata per cycle.

5. Coinbase Realtime IO
- Realtime ingestion MUST preserve sequence integrity and normalized event schema.
- Channel subscriptions MUST remain compatible with Coinbase Advanced Trade websocket semantics.

6. Governance
- Fiduciary constraints (exposure, turnover, concentration, confidence floors) MUST gate execution actions.
- Oversight signals MUST be emitted from the same decision tensors used for execution.

## Non-Functional Requirements
- Latency-sensitive hot path remains candle-first.
- Backward compatibility preserved for existing scripts via compatibility adapters.
- Deterministic experiment logging for convergence/perturb/repeat cycles.

## Acceptance Criteria
- A unified scorecard reports alpha, PnL, KD ratio per run and per cycle.
- KD ratio computation is implemented with momentum-capture semantics (kills/deaths).
- Convergence/perturb/repeat loop can run in backtest and realtime-shadow mode.
- Realtime websocket adapter emits normalized events with sequence-gap diagnostics.
- Fiduciary guardrails can veto or clip actions during perturbation cycles.

## Out of Scope
- Replacing Coinbase as execution venue.
- Non-momentum KD variants unless explicitly added in a follow-up track.
- Broad UI/dashboard redesign.
