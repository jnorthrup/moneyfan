# Guardrail Operator Runbook

Track: `runtime_drawdown_guardrails_rollback_20260303`

---

## Overview

The drawdown guardrail system automatically reduces risk exposure and halts trading
when portfolio drawdown exceeds configured thresholds. This runbook covers:

- Reading the current guardrail state
- Interpreting each state and its effect on trading
- Safe resume after a guardrail-triggered halt
- Adjusting thresholds for a session

---

## Guardrail States

| State | Drawdown Trigger (default) | Effect |
|-------|---------------------------|--------|
| `normal` | < 5% | No modification to trading |
| `warn` | ≥ 5% | Log-only. No parameter changes (WARN is informational) |
| `derisk` | ≥ 8% | Position size × 0.5, top-k × 0.5, confidence threshold + 10% |
| `halt` | ≥ 12% | New entries blocked (top-k → 0), engine stops, state saved |

Transitions require the condition to persist for `--guardrail-confirmation-window`
consecutive iterations (default: 1 = immediate).

---

## Reading the Current State

The current guardrail state is logged in `trading_state.json`:

```bash
jq '.guardrail_state, .halt_reason' trading_state.json
```

Transition events are appended to (default) `runtime/guardrail_events.jsonl`:

```bash
tail -5 runtime/guardrail_events.jsonl | python3 -m json.tool
```

Each event has the schema `moneyfan.runtime.guardrail.event.v1`:

```json
{
  "schema": "moneyfan.runtime.guardrail.event.v1",
  "timestamp": "...",
  "old_state": "normal",
  "new_state": "warn",
  "drawdown_pct": 0.052,
  "threshold_warn": 0.05,
  "threshold_derisk": 0.08,
  "threshold_halt": 0.12,
  "mode": "paper",
  "iteration": 42
}
```

---

## After a Guardrail Halt

When the engine halts via guardrail:

1. `trading_state.json` will contain `"guardrail_state": "halt"` and
   `"halt_reason": "guardrail_halt_triggered"`
2. Open positions are **not** force-closed by default — they remain in state
3. The engine will refuse to restart unless you explicitly acknowledge the halt

### Safe Resume (after investigating and deciding to continue)

```bash
# Option 1: resume respecting the saved halt (default - will refuse to start)
python3 run.py --guardrail-enabled ...

# Option 2: override the saved halt and resume trading (use only after investigation)
python3 run.py --guardrail-enabled --ignore-saved-halt-state ...
```

> **WARNING:** Only use `--ignore-saved-halt-state` after manually verifying that
> the drawdown condition has stabilised. The guardrail will immediately re-evaluate
> on the first iteration and may halt again.

---

## Adjusting Thresholds

All thresholds are CLI flags and can be changed between sessions:

```bash
python3 run.py \
  --guardrail-enabled \
  --guardrail-warn-drawdown-pct 0.03 \
  --guardrail-derisk-drawdown-pct 0.06 \
  --guardrail-halt-drawdown-pct 0.10 \
  --guardrail-confirmation-window 2 \
  --guardrail-events-log-path runtime/guardrail_events.jsonl \
  ...
```

- `--guardrail-confirmation-window 2` requires 2 consecutive violation iterations
  before a state transition, reducing sensitivity to short-lived spikes.
- Guardrails are **disabled by default** — they must be explicitly enabled
  with `--guardrail-enabled`.

---

## Validation Command Set

Run after any change touching guardrail code or config:

```bash
# From museum/
PYTHONPATH=/path/to/moneyfan python3 -m pytest tests/test_runtime_drawdown_guardrails.py -v

# Expected: 24/24 passed
```

Inspect saved state guardrail fields are present:

```bash
jq '{guardrail_state, guardrail_candidate_state, guardrail_candidate_iterations, halt_reason}' \
  trading_state.json
```

Confirm JSONL event schema:

```bash
python3 -c "
import json, sys
line = open('runtime/guardrail_events.jsonl').readline()
ev = json.loads(line)
assert ev['schema'] == 'moneyfan.runtime.guardrail.event.v1', 'bad schema'
print('OK:', ev['old_state'], '->', ev['new_state'])
"
```

---

## Rollout Notes and Fallback Toggles

- **Safe to deploy**: guardrails are **off by default**. Existing sessions without
  `--guardrail-enabled` are completely unaffected.
- **Fallback**: simply omit `--guardrail-enabled` to run as before.
- **State file backward compat**: old state files without guardrail fields load
  cleanly — missing fields default to `"normal"` / `0`.
- **No broker API side effects**: guardrail halt only blocks *new* entries in
  preview/paper mode and the freqtrade handoff path. Existing positions are not
  force-exited unless the operator separately configures that behaviour.
