# Replay Threshold Profiles (Operator Presets)

## Purpose

Named threshold profiles for:

- `/Users/jim/work/moneyfan/execution/freqtrade_contract_path_replay_validate.py`
- `/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh traffic-validate`

Profiles are keyed by:

- `exchange_target`
- `data_source`

This keeps operator cadence validation consistent and avoids hand-entering threshold flags for common contexts.

## Current Profiles

| Profile | Intended Use | `min_forward_rate` | `max_fill_rejects` | `max_proxy_rejects` | `require_all_signal_ids_seen` |
|---|---|---:|---:|---:|---|
| `default_strict` | Fallback strict profile | `1.0` | `0` | `0` | `true` |
| `coinbase_advanced__binance` | Default strict local target/source context | `1.0` | `0` | `0` | `true` |
| `coinbase_advanced__binance_relaxed` | Early tuning / noisy integration bring-up | `0.95` | `1` | `0` | `true` |
| `coinbase_advanced__coinbase_advanced` | Target-source aligned strict checks | `1.0` | `0` | `0` | `true` |
| `coinbase_advanced__coinbase_advanced_relaxed` | Target-source aligned but tolerant for bring-up | `0.98` | `1` | `0` | `true` |
| `freqtrade_paper__binance` | Freqtrade paper execution validation with Binance-source replay | `1.0` | `0` | `0` | `true` |

## Usage

Runbook:

```bash
/Users/jim/work/moneyfan/freqtrade_fidelity_runbook.sh \
  --traffic-validate-threshold-profile coinbase_advanced__binance_relaxed \
  traffic-validate
```

Direct validator:

```bash
python3 -m execution.freqtrade_contract_path_replay_validate \
  --threshold-profile coinbase_advanced__binance \
  --exchange-target coinbase_advanced \
  --data-source binance
```

## Overrides

Named profiles are defaults, not locks. You can override any threshold with CLI flags:

- `--min-forward-rate`
- `--max-fill-rejects`
- `--max-proxy-rejects`
- `--require-all-signal-ids-seen` / `--no-require-all-signal-ids-seen`

The validation JSON/markdown reports include:

- selected profile name
- resolved profile thresholds
- effective thresholds after overrides

## Tuning Guidance

- Start strict (`coinbase_advanced__binance`) for local deterministic checks.
- Use relaxed profiles temporarily during adapter/schema bring-up.
- Tighten relaxed profiles back to strict after root-causing rejects or signal propagation gaps.
- Treat repeated relaxed-profile passes as a signal to update the strict profile only after reviewing artifacts and operator cadence behavior.
