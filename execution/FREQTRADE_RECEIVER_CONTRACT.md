# Freqtrade Receiver Contract (Production Profile v1)

## Purpose

Define the payload contract between:

- `execution.freqtrade_handoff_bridge` (sender)
- production Freqtrade-facing receiver/adapter (consumer)
- `execution.freqtrade_fill_event_receiver` / `execution.freqtrade_fill_event_normalizer` (local compatibility path)

Primary requirement: preserve `signal_id` end-to-end so HRM fidelity reconciliation can join dispatch, ack, and realized fill/trade outcomes.

## Contract Scope

This document locks the production adapter profile for **bridge outbound webhook payloads** and the **minimum fill/trade update fields** that must be returned/logged for reconciliation.

- Bridge outbound schema: `moneyfan.freqtrade.bridge.webhook.v1`
- Canonical fill-event schema (internal normalized target): `moneyfan.freqtrade.fill_event.v1`

## 1. Bridge Outbound Webhook Payload (Required)

Producer:
- `/Users/jim/work/moneyfan/execution/freqtrade_handoff_bridge.py`
- `handoff_to_freqtrade_webhook_payload(...)`

Schema:
- `moneyfan.freqtrade.bridge.webhook.v1`

Required top-level fields:
- `schema`
- `ts_utc`
- `signal_id`
- `pair`
- `side` (`long` | `short`)
- `action` (`enter_long` | `enter_short`)
- `enter_long` (0|1)
- `enter_short` (0|1)
- `stake_fraction`
- `stoploss`
- `take_profit_pct`
- `metadata`

Required metadata subfields (for fidelity):
- `metadata.source_schema`
- `metadata.hrm.confidence` (nullable allowed)
- `metadata.hrm.pred_fwd_return` (nullable allowed)
- `metadata.hrm.net_effective_predicted_edge_bps` (nullable allowed)

Example payload:

```json
{
  "schema": "moneyfan.freqtrade.bridge.webhook.v1",
  "ts_utc": "2026-02-25T22:10:00Z",
  "signal_id": "hrm-sig-20260225-221000-000123",
  "pair": "BTC/USDT",
  "side": "long",
  "action": "enter_long",
  "enter_long": 1,
  "enter_short": 0,
  "stake_fraction": 0.15,
  "stoploss": -0.02,
  "take_profit_pct": 0.03,
  "metadata": {
    "source_schema": "moneyfan.freqtrade.handoff.v1",
    "source_dispatch": {
      "iteration": 4821,
      "source_mode": "paper",
      "source_broker_label": "freqtrade"
    },
    "hrm": {
      "confidence": 0.87,
      "pred_fwd_return": 0.0042,
      "score": 1.54,
      "score_mode": "calibrated",
      "passes_edge_gate": true,
      "net_effective_predicted_edge_bps": 28.0,
      "trade_head_calibration_loaded": true,
      "risk_tier": "normal",
      "raw_vetoed": false,
      "raw_veto_reason": null,
      "veto_overridden": false
    }
  }
}
```

## 2. Production Receiver Requirements

Your production receiver/adapter may transform this webhook payload into any Freqtrade-specific action/API call, but it must:

1. Preserve `signal_id`
2. Preserve `pair` and `side`
3. Persist/log enough information to emit a later fill/trade update including `signal_id`
4. Return an HTTP status/body suitable for bridge ack logging

Recommended receiver response body (optional but useful):

```json
{
  "ok": true,
  "signal_id": "hrm-sig-20260225-221000-000123",
  "accepted": true,
  "receiver_schema": "your.freqtrade.receiver.accept.v1",
  "freqtrade_request_id": "ftreq-abc123"
}
```

## 3. Fill/Trade Update Contract Back Into Moneyfan (Required Fields)

The reconciliation path only requires a subset of fields, but `signal_id` is mandatory.

Accepted source shapes:
- direct flat JSON object
- nested Freqtrade-like `trade.*` object
- local receiver ingest wrapper `moneyfan.freqtrade.trade_update_ingest.v1` with `payload`

Required field:
- `signal_id` (or `trade.signal_id`, or `metadata.signal_id`)

Strongly recommended fields:
- `pair`
- `side`
- `status` (e.g. `open`, `closed`, `filled`, `exit_filled`)
- `entry_price`
- `exit_price` (for closed trades)
- `pnl_pct` (or enough data to derive realized return)
- `fill_ts_utc`
- `exchange_trade_id`

Example fill/trade update payload (receiver -> local fill receiver or raw log):

```json
{
  "schema": "freqtrade.trade_update.example.v1",
  "signal_id": "hrm-sig-20260225-221000-000123",
  "pair": "BTC/USDT",
  "side": "long",
  "status": "closed",
  "entry_price": 52000.0,
  "exit_price": 52135.0,
  "pnl_pct": 0.002596,
  "fees_abs": 1.25,
  "fill_ts_utc": "2026-02-25T22:35:11Z",
  "exchange_trade_id": "987654321"
}
```

Freqtrade-like nested example (also accepted by normalizer/reconciler):

```json
{
  "trade": {
    "id": 12345,
    "pair": "ETH/USDT",
    "is_short": true,
    "open_rate": 3000.0,
    "close_rate": 2975.0,
    "close_profit_abs": 12.1,
    "close_profit": 0.0083,
    "close_date": "2026-02-25T23:00:04Z"
  },
  "status": "closed",
  "metadata": {
    "signal_id": "hrm-sig-20260225-225800-000124"
  }
}
```

## 4. Local Compatibility Path (Reference)

For local testing, the existing receiver already accepts POSTed trade updates and writes:

- raw ingest: `moneyfan.freqtrade.trade_update_ingest.v1`
- canonical fill events: `moneyfan.freqtrade.fill_event.v1`

Reference implementation:
- `/Users/jim/work/moneyfan/execution/freqtrade_fill_event_receiver.py`
- `/Users/jim/work/moneyfan/execution/freqtrade_fill_event_normalizer.py`

HTTP endpoints:
- `POST /trade-update`
- `POST /fill`
- `POST /fill-event`
- `POST /ingest`

## 5. Non-Negotiable Fidelity Rules

- `signal_id` must be stable and unchanged across dispatch, bridge ack, receiver logs, and fill/trade updates.
- Do not generate a new `signal_id` in downstream adapters.
- If a downstream system cannot store custom metadata, maintain an external sidecar mapping from the downstream trade/order identifier back to `signal_id`.
- Prefer logging both acceptance and fill updates to simplify debugging of orphan ack/fill rows.

## 6. Versioning / Changes

If the production payload shape changes materially:

1. bump schema suffix (`.v2`)
2. update bridge adapter + receiver docs together
3. preserve `signal_id` semantics
4. add/adjust tests before rollout
