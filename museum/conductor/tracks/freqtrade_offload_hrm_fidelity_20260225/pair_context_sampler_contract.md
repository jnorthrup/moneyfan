# Pair-Context Sampler Contract (Variable-Width Frames)

## Purpose

Formalize the contract for building variable-width training/inference frames from the `pandas muxer` output.

This supports:

- stochastic pair-width sampling
- random sampling per frame
- exchange-wide ranker-guided pair selection
- focal tradepair + ranked contextual peers

while preserving:

- reproducibility
- leakage controls
- compatibility with HRM variable-width sequence/set inputs

## Relationship to the Muxer

The sampler sits **after** the muxer and **before** HRM batching.

- `pandas muxer` = stable per-pair feature record contract
- pair-context sampler = variable-width frame constructor
- HRM = masked consumer of sampled pair-context frames

The sampler must not mutate core muxer semantics. It only selects/reorders/group frames.

Reference:
- `/Users/jim/work/moneyfan/conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/pandas_muxer_contract.md`

## Core Concepts

- `focal_pair`
  - the tradepair being evaluated/targeted for a frame
- `candidate_universe`
  - eligible peer pairs at that frame timestamp (after filters)
- `ranker`
  - exchange-wide ranking function used to prioritize candidates
- `selected_pairs`
  - sampled contextual peers included in the frame
- `pair_width`
  - number of pair slots in the frame (including focal pair if modeled as a slot)

## Required Sampler Outputs (Per Frame)

Each emitted frame should carry:

- `frame_id`
  - unique identifier for the sampled frame
- `frame_ts_utc`
  - frame timestamp used for pair selection
- `focal_pair`
  - canonical pair symbol (for example `BTC/USDT`)
- `pair_width`
  - realized slot count before padding
- `max_pair_width`
  - batch/model slot capacity if padded
- `slot_mask`
  - presence mask for valid sampled slots
- `slot_pairs`
  - ordered list of selected pairs (canonical symbols)
- `slot_features`
  - muxer-derived feature rows/arrays for selected slots
- `sampling_metadata`
  - required sampler/ranker provenance (see below)

## Required Sampling Metadata (Provenance / Reproducibility)

### Sampler provenance

- `sampler_schema`
- `sampler_version`
- `sampler_policy`
  - for example `rank_weighted_without_replacement`
- `random_seed` (or deterministic seed derivation inputs)
- `seed_scope`
  - for example per-epoch / per-frame / per-batch

### Ranker provenance

- `ranker_name`
- `ranker_version`
- `ranker_score_timestamp_policy`
  - confirms no future leakage
- `ranker_feature_cutoff_ts_utc` (or equivalent)

### Universe provenance

- `exchange_target`
- `data_source`
- `universe_filter_version`
- `candidate_universe_size`
- `candidate_pairs` (optional full list; required for audit mode)
- `excluded_pairs` with reasons (optional but recommended)

### Sampling result diagnostics

- `focal_pair_rank` (if ranker assigns it)
- `selected_pair_ranks`
- `selection_probabilities` (if stochastic weighted sampling)
- `replacement` (`true/false`)

## Ordering / Invariance Rules

Choose one and make it explicit in the sampler contract:

### Option A: Order-aware (rank-ordered slots)

Use a canonical slot ordering, for example:
- focal pair first
- remaining pairs sorted by rank score descending
- deterministic tie-breaker (`pair` symbol)

Required metadata:
- `slot_ordering = "focal_then_rank_desc_pair_tiebreak"`

### Option B: Order-invariant (set semantics)

If HRM aggregation is order-invariant:
- still emit deterministic audit ordering
- model should ignore slot order except identity/features

Required metadata:
- `slot_ordering = "audit_sorted_only"`
- `model_slot_order_invariant = true`

## Leakage / Bias Controls (Non-Negotiable)

- Ranker inputs must be point-in-time valid at `frame_ts_utc`.
- No future returns or post-frame outcomes in ranker/sampler features.
- Ranker version changes must be logged.
- Candidate-universe filters must be versioned and deterministic for replay.
- If sampling policy changes, `sampler_version` must change.

## Padding / Masking Expectations (HRM Compatibility)

For variable widths:

- pad slot features to `max_pair_width`
- emit `slot_mask`
- ensure loss/attention respects mask
- log actual `pair_width` distribution for curriculum/debugging

Recommended additional diagnostics:

- `pair_width_histogram` per run
- `mean_pair_width`, `p95_pair_width`
- focal-pair inclusion rate (should be 100% unless intentionally ablated)

## Minimal Frame Example (JSON-ish)

```json
{
  "frame_id": "pcs-20260226-0001",
  "frame_ts_utc": "2026-02-26T00:00:00Z",
  "focal_pair": "ETH/USDT",
  "pair_width": 5,
  "max_pair_width": 8,
  "slot_mask": [1, 1, 1, 1, 1, 0, 0, 0],
  "slot_pairs": ["ETH/USDT", "BTC/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
  "slot_ordering": "focal_then_rank_desc_pair_tiebreak",
  "sampling_metadata": {
    "sampler_schema": "moneyfan.pair_context_sampler.v1",
    "sampler_version": "pcs_v1",
    "sampler_policy": "rank_weighted_without_replacement",
    "random_seed": 424242,
    "seed_scope": "frame",
    "ranker_name": "exchange_pair_target_ranker",
    "ranker_version": "epr_v3",
    "ranker_score_timestamp_policy": "point_in_time_only",
    "ranker_feature_cutoff_ts_utc": "2026-02-26T00:00:00Z",
    "exchange_target": "coinbase_advanced",
    "data_source": "binance",
    "universe_filter_version": "uf_v2",
    "candidate_universe_size": 142,
    "focal_pair_rank": 3,
    "selected_pair_ranks": [3, 1, 7, 10, 14],
    "replacement": false
  }
}
```

## Artifact Tagging Alignment (Implemented Elsewhere)

The execution-validation pipeline already records:

- `exchange_target`
- `data_source`

The sampler contract should emit the same tags so training/eval/execution diagnostics can be compared without role ambiguity.

## Next Steps

- Add a concrete sampler artifact schema in code (`moneyfan.pair_context_sampler.v1`)
- Add replayable sampler traces for training batches (audit mode)
- Add muxer->sampler conformance checks (required columns, nullability, point-in-time guards)
- Add ranker version registry/manifest linkage for experiment reproducibility

## Code Reference (Implemented)

Initial code-level trace schema + JSONL audit writer utilities:

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_trace.py`

Initial muxer->sampler conformance checker (required columns / nullability / timestamp guardrails):

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_conformance.py`

Initial sampler trace audit/report utility (width distributions + sampler/ranker/source-target summaries):

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_trace_report.py`

End-to-end sampler audit smoke harness (sample muxer rows + traces + conformance/report artifacts):

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_audit_smoke.py`

Thresholded sampler-audit validator (JSON + markdown pass/fail artifacts with profile support):

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_audit_validate.py`

Runbook entry points (operator-facing):

- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-conformance`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-trace-report`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-audit`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-smoke`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-validate`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-profiles`

Profile reference:

- `/Users/jim/work/moneyfan/conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/sampler_threshold_profiles.md`
