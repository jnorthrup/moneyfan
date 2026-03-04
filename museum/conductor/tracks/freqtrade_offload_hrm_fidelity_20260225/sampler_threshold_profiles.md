# Sampler Validation Threshold Profiles (Operator Presets)

## Purpose

Named threshold profiles for:

- `/Users/jim/work/moneyfan/execution/pair_context_sampler_audit_validate.py`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-validate`

Profiles are keyed by:

- `exchange_target`
- `data_source`

This keeps sampler audit validation consistent across operator cadence without repeatedly passing raw threshold flags.

## Current Profiles

| Profile | Intended Use | `min_trace_rows_valid` | `max_focal_pair_inclusion_failures` | `min_pair_width_p95` | `max_pair_width_p95` | `require_conformance_pass` |
|---|---|---:|---:|---:|---:|---|
| `default_strict` | Fallback strict profile | `1` | `0` | `1.0` | `null` | `true` |
| `coinbase_advanced__binance` | Default strict local target/source context | `2` | `0` | `1.0` | `64.0` | `true` |
| `coinbase_advanced__binance_relaxed` | Early tuning / noisy integration bring-up | `1` | `0` | `1.0` | `128.0` | `true` |
| `coinbase_advanced__coinbase_advanced` | Target-source aligned strict checks | `2` | `0` | `1.0` | `64.0` | `true` |
| `coinbase_advanced__coinbase_advanced_relaxed` | Target-source aligned but tolerant for bring-up | `1` | `0` | `1.0` | `96.0` | `true` |
| `freqtrade_paper__binance` | Paper/offload path sampling validation | `1` | `0` | `1.0` | `128.0` | `true` |

## Usage

Runbook:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh \
  --sampler-validate-threshold-profile coinbase_advanced__binance_relaxed \
  sampler-validate
```

List available profiles:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh sampler-profiles
```

Direct validator:

```bash
python3 -m execution.pair_context_sampler_audit_validate \
  --threshold-profile coinbase_advanced__binance \
  --exchange-target coinbase_advanced \
  --data-source binance
```

## Overrides

Named profiles are defaults, not locks. You can override thresholds with CLI flags:

- `--min-trace-rows-valid`
- `--max-focal-pair-inclusion-failures`
- `--min-pair-width-p95`
- `--max-pair-width-p95`
- `--require-conformance-pass` / `--no-require-conformance-pass`

Validation JSON/markdown artifacts include:

- selected profile name
- resolved profile thresholds
- effective thresholds after overrides

## Tuning Guidance

- Start strict for deterministic local checks.
- Use relaxed profiles only during bring-up/noisy adaptation phases.
- Tighten relaxed profiles after root-causing focal inclusion or width-distribution drift.
