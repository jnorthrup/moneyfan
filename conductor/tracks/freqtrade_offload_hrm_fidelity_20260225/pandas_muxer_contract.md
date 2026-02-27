# Pandas Muxer Contract (Stable Spine)

## Intent

Preserve the `pandas muxer` layer as the stable abstraction boundary across:

- abundant pretraining data sources (for example `binance`)
- target execution venues (for example `coinbase_advanced`)
- execution adapters (`freqtrade`, direct exchange adapters, simulations)
- diagnostics (reconciliation / "fidelity" reports)

The muxer is the structure we hold onto. Exchange-specific behavior belongs in adapters and calibration layers, not in the core muxed dataframe contract.

## Role Separation

- `data_source`
  - where market/training/eval observations came from
  - examples: `binance`, `coinbase_advanced`, `mixed`, `synthetic`
- `exchange_target`
  - where execution behavior is being optimized/evaluated
  - examples: `coinbase_advanced`, `freqtrade_paper`, `freqtrade_live`

These labels should travel with downstream artifacts to avoid conflating:
- pretraining substrate abundance (`binance`)
- target execution truth (`coinbase_advanced`)

## Muxer-Level Invariants (Minimum)

The muxer contract should preserve these identifiers/columns (names may vary if a mapping layer exists, but semantics must remain stable):

- instrument identity
  - canonical symbol/pair (for example `BTC/USDT`)
  - venue symbol if needed (for example `BTCUSDT`)
- timestamp(s)
  - event/observation timestamp
  - timezone/UTC normalization
- target/label correlation IDs
  - `signal_id` (or muxer field that deterministically maps to it)
  - row/run identifiers where needed for joins
- direction semantics
  - long/short (or equivalent sign convention)
- price fields
  - entry reference price / observed price
  - optional exit/realized fields when available
- model metadata passthrough
  - confidence / score / calibration flags (when attached downstream)

## Adapter Boundary Rules

- Exchange adapters may transform payload schemas, but must not destroy muxer semantics.
- `signal_id` is a cross-layer join key and must survive adapter boundaries.
- Venue-specific order fields belong in adapter payloads, not in muxer core columns.
- Reconciliation artifacts may be named "fidelity", but they are execution-abstraction diagnostics, not the final objective.

## Current Artifact Tagging (Implemented)

The pipeline/reconciliation tooling now supports:

- `exchange_target`
- `data_source`

These tags are recorded in reconciliation inputs and pipeline summaries so Binance/Coinbase roles remain explicit.

Default local runbook values:

- `EXCHANGE_TARGET=coinbase_advanced`
- `DATA_SOURCE=binance`

## Next Steps (Muxer-Centric)

- Add a concrete muxer dataframe schema checklist (columns + dtypes + nullability).
- Add adapter conformance tests that validate preservation of muxer semantics across:
  - Binance ingest -> muxer
  - Coinbase Advanced target eval -> muxer
  - execution offload/reconciliation joins
