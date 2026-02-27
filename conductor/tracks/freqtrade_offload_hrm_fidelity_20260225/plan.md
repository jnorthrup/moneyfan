# Plan: Freqtrade Offload + HRM Fidelity Audit Loop

## Purpose

Track the runtime execution-offload and fidelity-audit work that was intentionally de-scoped from the MLX HRM training-only track.

This plan exists to keep Conductor visibility for:
- Freqtrade handoff/offload seams
- execution bridge + receiver plumbing
- HRM fidelity reconciliation/reporting
- operator runbook workflows

## Recap / Main TODO (Current)

### Recap (Implemented)
- HRM runtime can offload ranked intents to Freqtrade-oriented JSONL handoff payloads.
- HRM fidelity dispatch log now captures `signal_id`-keyed prediction/calibration/risk snapshots.
- Bridge, receiver, normalizer, reconciler, and markdown reporting utilities exist.
- Local runbook orchestrates receiver/bridge/pipeline/report/replay workflows.
- Compare report generator exists for baseline vs candidate reconciliation deltas.
- Snapshot history indexing helper exists for timestamped report/compare review.
- Bridge supports explicit receiver profile validation (`production_v1`) for outbound payload shape and webhook response contract checks.
- Contract-compliant receiver/proxy skeleton exists for `production_v1` acceptance responses and optional downstream webhook forwarding.
- Contract proxy supports configurable downstream payload mapping modes (including a Freqtrade-oriented webhook mapping that preserves `signal_id` in metadata).
- Integration smoke harness exists for the `bridge -> contract-proxy -> fill-receiver` path with `production_v1` bridge validation and `signal_id` propagation checks.
- Repeatable sample traffic replay harness exists for multiple bridge passes / mixed long-short handoffs through the production-like local path.
- Pipeline/reconciliation artifacts now carry `exchange_target` and `data_source` tags (for example `coinbase_advanced` target vs `binance` source).
- `pandas muxer` contract doc is captured as the stable abstraction boundary for exchange-agnostic structure.
- Pair-context sampler contract doc is captured for stochastic variable-width frames (ranker-guided sampling + masks + reproducibility metadata).
- Initial code-level pair-context sampler trace schema + JSONL audit writer utilities exist (`moneyfan.pair_context_sampler.v1`).
- Initial muxer->sampler conformance checker exists (required columns/nullability/timestamp guardrails with JSON report).
- Initial sampler trace audit/report utility exists (width histograms, ranker/sampler version counts, source/target summaries in JSON/markdown).
- Runbook exposes sampler audit workflows (`sampler-conformance`, `sampler-trace-report`, `sampler-audit`) for operator use.
- Sampler audit smoke harness exists (sample muxer rows + sampler traces -> conformance/report artifacts) with a runbook entry point (`sampler-smoke`).
- Sampler audit threshold validator exists (JSON + markdown pass/fail artifacts with named profiles) with a runbook entry point (`sampler-validate`).
- Thresholded replay validation now emits pass/fail JSON + markdown artifacts (with timestamped snapshots) keyed by `exchange_target` / `data_source`.
- Replay validation supports named threshold profiles (for example `coinbase_advanced__binance`) with CLI/runbook overrides for operator cadence checks.
- Multiple strict/relaxed threshold profiles are now available, with a small operator profile-table doc.

### Main TODO (Conductor Recap)
- Integrate/tune the contract proxy downstream payload mapping against the real Freqtrade endpoint schema and validate end-to-end responses under production-like traffic.
- Apply and tune retention/pruning policy defaults in real operator cadence (preview daily, prune weekly or as needed).
- Tune the expanded named threshold profiles for replay validation using production-like traffic samples and expected ranges per target/source context.
- (Nice-to-have) add compare-report history review helpers (e.g., timestamped compare retention + diff indexing).
  - Progress: snapshot history indexing is implemented; diff indexing/summary ranking remains optional.

## Phase 1: Offload Seam and HRM Dispatch Fidelity

- [x] Task: Add normalized Freqtrade handoff adapter payload from HRM intents
- [x] Task: Add runtime offload path in `run.py` (JSONL handoff instead of internal execution)
- [x] Task: Add `signal_id` correlation and HRM fidelity dispatch log snapshots
- [x] Task: Add tests covering handoff payload and offload behavior

## Phase 2: Execution Plumbing and Fill Canonicalization

- [x] Task: Add handoff bridge (JSONL -> webhook/dry-run) with offset state and ack log
- [x] Task: Add Freqtrade trade update receiver (raw ingest + canonical fill-event writes)
- [x] Task: Add canonical fill-event normalizer and reject logging
- [x] Task: Add tests for bridge/receiver/normalizer paths

## Phase 3: Fidelity Reconciliation and Reporting

- [x] Task: Add `signal_id`-keyed reconciliation (dispatch + ack + fill)
- [x] Task: Add single-run markdown fidelity report generator
- [x] Task: Add baseline-vs-candidate markdown compare report generator
- [x] Task: Add tests for reconciliation/report/compare report rendering

## Phase 4: Operator Workflow / Runbook

- [x] Task: Add local fidelity pipeline orchestrator (bridge -> normalize -> reconcile)
- [x] Task: Add runbook wrapper for receiver / bridge / pipeline / replay / report
- [x] Task: Add replay mode (`--skip-bridge`) for fast re-analysis
- [x] Task: Add timestamped markdown report snapshots

## Phase 5: Stubbed Next Tasks (Pending)

- [x] Task: Runbook `compare` subcommand wrapping `execution.freqtrade_fidelity_compare_report`
- [x] Task: Timestamped compare report snapshots (latest + historical)
- [x] Task: Artifact retention/pruning helper (keep last N reports/snapshots, prune stale runtime files)
- [x] Task: Canonical production receiver contract doc/example payloads for Freqtrade integration
- [x] Task: Daily operator runbook markdown (step-by-step, expected files, troubleshooting)
