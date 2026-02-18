# moneyfan — Project Goals

## What this is

A crypto trading system for 30 Coinbase Advanced spot pairs, selected by currency-graph route depth (not BTC favoritism). The system trains 24 signal codecs under a fiduciary HRM meta-allocator and executes via Coinbase Advanced Trade API.

## Architecture

```
Coinbase WS → PANDAS candles → 30 instruments → {24 tradebots/codecs} → HRM IO → fiduciary overlay → execution
```

### Signal codec layer (fast, regime-specific)
- 24 codecs, each a small model (2-layer, regime-typed)
- Each codec trains with its own context-shaped reward surface:
  - Regime accuracy (was the signal right in its claimed regime?)
  - Calibration (Brier score on confidence)
  - Route-depth penalty (discount on thin pairs)
- Each codec outputs: `[confidence, direction, regime_fit]`

### HRM meta-allocator (slow, portfolio-level)
- Consumes codec outputs, not raw signals
- Learns composition: which codecs to trust and when to blend
- Convergence detection: nonzero when >= 2 codecs agree direction + confidence
- Convergence/perturb/repeat optimization loop:
  1. Detect convergence across codec outputs
  2. Apply bounded perturbation to weighting, allocation, thresholds
  3. Evaluate delta on alpha / PnL / KD ratio
  4. Retain or revert

### Fiduciary governance
- Exposure, turnover, concentration, confidence floors gate all execution
- Oversight diagnostics emitted from the same tensors used for execution

## Primary metrics

| Metric | Definition |
|--------|-----------|
| **Alpha** | Excess return vs buy-and-hold baseline |
| **PnL** | Net profit after maker/taker fees (0.4%/0.6%) + spread (0.25%) |
| **KD ratio** | Kills (positive momentum captures) / Deaths (failed captures) |

## Pair selection

30 pairs chosen by `currency_graph.py` route depth — number of alternative routing paths through the currency graph, not volume on a single pair. This selects for structural liquidity and arbitrage surface.

## Tech stack (active)

- **Python 3.14** — HRM training, signal codecs, fiduciary RL, Coinbase pipeline
- **PyTorch** — model training and inference
- **Node.js (ESM)** — backtesting (SQLite + DuckDB), web dashboard
- **Coinbase Advanced Trade API** — primary exchange (REST + WebSocket)

## Current state

- Coinbase JWT auth: working (`coinbase_auth.py`)
- Fiduciary controller: implemented (`hrm/fiduciary_controller.py`, `hrm/fiduciary_rl.py`)
- Signal convergence: 16 signals implemented (`hrm/signal_hrm.py`), scaling to 24 codecs
- Currency graph routing: implemented (`hrm/currency_graph.py`)
- Continuous training loop: implemented (`hrm/continuous_trainer.py`)
- Backtest: SQLite + DuckDB over 699K+ hourly candles
- Live execution: simulation mode working, pending live API credentials

## What this is NOT

- Not an AGI or general reasoning system
- Not options, futures, fixed income, or multi-exchange
- Not a UI/dashboard product — the dashboard exists for monitoring only
- Not Kotlin/XChange (legacy path, museumified)
