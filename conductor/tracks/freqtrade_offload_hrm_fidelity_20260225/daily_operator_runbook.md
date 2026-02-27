# Daily Operator Runbook: Freqtrade Offload + HRM Fidelity Audit

## Purpose

Run the local end-to-end audit loop for HRM signal fidelity when execution is offloaded toward Freqtrade.

This runbook covers:
- receiver/bridge/pipeline/report commands
- expected runtime artifacts
- replay/compare workflow for iteration
- retention/pruning defaults
- common troubleshooting checks

## Preconditions

- Working directory: `/Users/jim/work/moneyfan`
- Python available as `python3`
- HRM runtime (`/Users/jim/work/moneyfan/run.py`) configured to emit:
  - handoff JSONL (`freqtrade_handoff.jsonl`)
  - fidelity dispatch log (`hrm_fidelity_dispatch.jsonl`)
- Local runtime directory exists or will be created by runbook:
  - `/Users/jim/work/moneyfan/runtime`

Note: `pytest` is not required for operations. Runtime validation here is command/output and artifact based.

## Core Commands (Daily Path)

### 1. Inspect resolved commands/paths (safe)

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh print
```

Use this first if you changed `RUNTIME_DIR`, ports, or compare paths.

### 2. Run local end-to-end loop (receiver + bridge + pipeline + report)

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh all
```

What it does:
- starts local fill-event receiver
- runs bridge (handoff -> webhook to local receiver)
- runs fidelity pipeline (normalize + reconcile)
- renders markdown fidelity report (latest + timestamped snapshot)
- prints a non-destructive prune preview (`prune --dry-run`)

### 3. Fast re-analysis without re-dispatching (replay)

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh replay
```

Use when:
- tuning normalization/reconciliation/report code
- re-running after fill/raw data cleanup
- avoiding duplicate bridge dispatches

### 4. Regenerate markdown report from existing reconciliation JSON

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh report
```

### 5. Compare baseline vs candidate reconciliation outputs

If comparing two files:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh \
  --baseline-recon-json /absolute/path/baseline.json \
  --candidate-recon-json /absolute/path/candidate.json \
  compare
```

If flags are omitted, compare defaults to the current runtime reconciliation JSON for both sides (mostly useful as a wiring check).

### 6. Sampler audit (muxer -> sampler conformance + trace report)

Use this when validating the `pandas muxer` + variable-width pair-context sampler path.

Gated bundle (conformance first, trace report second):

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-audit
```

Quick generated-data smoke (validates the sampler audit toolchain end-to-end without requiring real muxer/trace inputs):

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-smoke
```

Thresholded validation over the sampler smoke/toolchain (writes JSON + markdown pass/fail artifacts):

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-validate
```

Standalone steps:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-conformance
```

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-trace-report
```

Optional custom inputs:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh \
  --sampler-muxer-rows-jsonl /absolute/path/pair_context_muxer_rows.jsonl \
  --sampler-trace-jsonl /absolute/path/pair_context_sampler_trace.jsonl \
  sampler-audit
```

## Expected Runtime Artifacts

Default runtime directory:
- `/Users/jim/work/moneyfan/runtime`

Primary artifacts:
- `freqtrade_handoff.jsonl`
  - HRM -> Freqtrade handoff intents (`moneyfan.freqtrade.handoff.v1`)
- `hrm_fidelity_dispatch.jsonl`
  - HRM dispatch snapshots keyed by `signal_id`
- `freqtrade_handoff_bridge_state.json`
  - bridge byte offset state
- `freqtrade_dispatch_ack.jsonl`
  - bridge ack results (`moneyfan.freqtrade.dispatch_ack.v1`)
- `freqtrade_trade_updates_raw.jsonl`
  - raw receiver ingest rows (`moneyfan.freqtrade.trade_update_ingest.v1`)
- `freqtrade_fill_events.jsonl`
  - canonical fill events (`moneyfan.freqtrade.fill_event.v1`)
- `freqtrade_fill_event_rejects.jsonl`
  - rejects from receiver/normalizer
- `hrm_freqtrade_fidelity_reconciliation.json`
  - reconciliation summary + records
- `hrm_freqtrade_fidelity_reconciliation.csv`
  - tabular reconciliation rows
- `hrm_freqtrade_fidelity_report.md`
  - latest single-run markdown report
- `hrm_freqtrade_fidelity_report_YYYYMMDD_HHMMSS.md`
  - timestamped report snapshots
- `hrm_freqtrade_fidelity_compare_report.md`
  - latest compare markdown report
- `hrm_freqtrade_fidelity_compare_report_YYYYMMDD_HHMMSS.md`
  - timestamped compare report snapshots
- `pair_context_sampler_conformance.json`
  - muxer->sampler readiness report (`moneyfan.pair_context_sampler_conformance.v1`)
- `pair_context_sampler_trace_report.json`
  - sampler trace audit summary (`moneyfan.pair_context_sampler_trace_report.v1`)
- `pair_context_sampler_trace_report.md`
  - human-readable sampler trace audit summary
- `pair_context_sampler_trace_report_YYYYMMDD_HHMMSS.md`
  - timestamped sampler trace report snapshots

## Retention / Pruning Defaults (Operator Policy)

Retention helper:
- `/Users/jim/work/moneyfan/execution/freqtrade_fidelity_retention.py`

Current default policy (recommended daily cadence):
- keep latest stable reports (`*.md`, reconciliation JSON/CSV) always
- keep newest `14` single-run report snapshots
- keep newest `14` compare report snapshots
- prune runtime `.jsonl/.json/.csv/.log` files older than `14` days

Preview planned deletions (safe):

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh prune
```

Apply pruning (explicit):

```bash
python3 -m execution.freqtrade_fidelity_retention \
  --runtime-dir /Users/jim/work/moneyfan/runtime \
  --keep-report-snapshots 14 \
  --keep-compare-snapshots 14 \
  --stale-days 14
```

Recommended operator cadence:
- daily: review prune preview only
- weekly: run actual prune after confirming reports are archived/reviewed

## Sampler Audit Inputs / Outputs

Default sampler audit inputs:

- muxer rows JSONL input:
  - `/Users/jim/work/moneyfan/runtime/pair_context_muxer_rows.jsonl`
- sampler trace JSONL input:
  - `/Users/jim/work/moneyfan/runtime/pair_context_sampler_trace.jsonl`

Default sampler audit outputs:

- `/Users/jim/work/moneyfan/runtime/pair_context_sampler_conformance.json`
- `/Users/jim/work/moneyfan/runtime/pair_context_sampler_trace_report.json`
- `/Users/jim/work/moneyfan/runtime/pair_context_sampler_trace_report.md`
- timestamped sampler trace report snapshots next to the stable markdown path

Notes:

- `sampler-audit` will stop after conformance if the muxer rows fail validation.
- Conformance is a gate for sampler trace reporting, not a replacement for trace-level audit metrics.
- Sampler trace reports summarize variable-width behavior (`pair_width` distribution), sampler/ranker versions, and `exchange_target` / `data_source` tags.

## Troubleshooting

### No reconciled rows / empty metrics

Checks:
- `signal_id` exists in dispatch, ack, and fill/trade updates
- fill updates actually reached receiver (`freqtrade_trade_updates_raw.jsonl`)
- canonical fill events were produced (`freqtrade_fill_events.jsonl`)
- bridge ack status shows accepted/forwarded rows

Useful commands:

```bash
python3 -m execution.freqtrade_fidelity_pipeline --skip-bridge --print-summary
```

```bash
python3 -m execution.freqtrade_fidelity_report \
  --reconciliation-json /Users/jim/work/moneyfan/runtime/hrm_freqtrade_fidelity_reconciliation.json \
  --out-md /Users/jim/work/moneyfan/runtime/hrm_freqtrade_fidelity_report.md \
  --also-write-timestamped
```

### Many `orphan_ack` or `orphan_fill` rows

Likely causes:
- downstream system dropped or rewrote `signal_id`
- delayed fill updates arrived after reconciliation run
- receiver accepted payload but could not canonicalize later event shape

Action:
- validate payloads against `/Users/jim/work/moneyfan/execution/FREQTRADE_RECEIVER_CONTRACT.md`
- inspect `freqtrade_fill_event_rejects.jsonl`
- rerun `replay` after late events arrive

### Receiver rejects payloads (`400`)

Likely causes:
- invalid JSON body
- non-object payload
- missing `signal_id`

Action:
- inspect `/Users/jim/work/moneyfan/runtime/freqtrade_fill_event_rejects.jsonl`
- confirm producer preserves `signal_id` exactly
- verify fields match examples in `/Users/jim/work/moneyfan/execution/FREQTRADE_RECEIVER_CONTRACT.md`

### Bridge re-sends old handoffs unexpectedly

Likely causes:
- handoff file truncated/rotated (bridge offset reset by design)
- bridge state file removed

Action:
- inspect `/Users/jim/work/moneyfan/runtime/freqtrade_handoff_bridge_state.json`
- if replaying intentionally, prefer `replay` (skip bridge) instead of re-running `bridge`

### Sampler conformance fails (`sampler-audit` stops before trace report)

Likely causes:
- missing required muxer columns (`pair`, `symbol`, `ts_utc`)
- null/empty values in non-nullable columns
- invalid timestamp values (`ts_utc`)
- non-monotonic timestamps (if monotonic mode is enabled in direct CLI use)

Action:
- inspect `/Users/jim/work/moneyfan/runtime/pair_context_sampler_conformance.json`
- verify muxer output against `/Users/jim/work/moneyfan/conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/pandas_muxer_contract.md`
- verify sampler expectations in `/Users/jim/work/moneyfan/conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/pair_context_sampler_contract.md`

### Sampler trace report looks wrong (widths/ranker versions/source-target tags)

Likely causes:
- sampler trace JSONL not using `moneyfan.pair_context_sampler.v1`
- stale trace file from previous runs
- missing or inconsistent `sampling_metadata` fields

Action:
- inspect `/Users/jim/work/moneyfan/runtime/pair_context_sampler_trace.jsonl`
- regenerate trace rows with `/Users/jim/work/moneyfan/execution/pair_context_sampler_trace.py` utilities
- rerun `sampler-trace-report` and review:
  - `pair_width_histogram`
  - `ranker_versions`
  - `exchange_targets`
  - `data_sources`

## Production Adapter Note

The local receiver is a compatibility/test harness. Production integration should follow:
- `/Users/jim/work/moneyfan/execution/FREQTRADE_RECEIVER_CONTRACT.md`

Non-negotiable:
- preserve `signal_id` end-to-end for fidelity reconciliation.
