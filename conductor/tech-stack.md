# Tech Stack

## Purpose
This document records the current (brownfield) technology stack used by `moneyfan`.

It is descriptive, not prescriptive, but the framing is explicit:
- the system is being built to make money with the HRM
- the stack exists to support efficient world-space representation, measurable edge estimation, and safe execution/promotion loops

## Project Type
- Brownfield, local-first quantitative trading / ML research monorepo
- Primary workflow: local training, evaluation, paper trading, governance automation
- Profit objective: improve risk-adjusted trading performance through disciplined model + execution + governance iteration

## Stack Selection Principle (Current)
The practical selection rule in this repo is:
- keep the representation/data path efficient enough to iterate quickly
- keep evaluation/governance strong enough to avoid fake alpha
- prefer local, inspectable tools over opaque managed systems while the edge is still being discovered

This means the stack is not just "what runs" but "what best supports":
- world-model quality
- calibration quality
- execution quality
- promotion discipline

## Core Language
- **Python 3** (primary application, training, backtesting, orchestration, dashboard, daemon)

## ML / Modeling Stack
- **MLX** (`mlx.core`) for HRM model execution/training (when available)
- Project-local HRM implementation modules under `hrm/`
- Stochastic episode training implemented in `train.py`
- MLX function transforms / autograd (`mx.grad`, `mx.value_and_grad`) as the path for updating shared HRM weights

### Why this matters for profit
- MLX + local HRM code enables fast iteration on the model’s internal representation of market state
- Stochastic training supports regime robustness (reducing brittle overfit behavior that destroys live performance)
- Autograd is the mechanism that turns better objectives into actual weight updates (without this, "world understanding" does not become trading behavior)

## Autograd Strategy (Profit-Oriented, GOALS-Aligned)
Autograd is the implementation bridge between:
- the HRM's efficient representation of the market world-space
- and the downstream goal of profitable, risk-controlled execution

### How autograd moves the system forward
1. **Define a single scalar training objective**
   - world-model / representation terms (next-bar feature prediction, indicator kernels, codec targets)
   - trade-head terms (direction / confidence / sizing / risk-head realism)
   - cost-aware penalties (turnover, fee/slippage sensitivity, overtrading)
   - regime robustness weighting (so one easy slice does not dominate gradient updates)

2. **Backpropagate through shared HRM components**
   - Use MLX autograd to send gradient signal through the shared encoder / TemporalOrderBook and into macro/tactical heads.
   - This is the practical mechanism for improving "understanding" rather than just changing execution rules.

3. **Keep non-differentiable truth outside the graph**
   - Exact walk-forward PnL, veto policy outcomes, and promotion decisions remain evaluation/governance signals.
   - Those are used to choose objectives and promote/reject artifacts, not directly differentiated.

4. **Iterate with governance**
   - Autograd proposes weight changes.
   - Backtests / calibration governor / simmer gates decide whether those changes are allowed to survive.

### Practical rule
- If an autograd-driven objective change improves training loss but fails OOS calibration or regime-aware validation, it is not treated as progress toward profit.

## Numeric / DataFrame Stack
- **NumPy**
- **pandas**

### Why this matters for profit
- Fast feature engineering and simulation loops increase experiment throughput
- Efficient vectorized transforms improve consistency between training, backtest, and runtime features

## Market Data / Storage / Query Stack
- Local parquet files (primarily under `data/binance/` and related generated stores)
- **DuckDB** for efficient local analytical reads and parquet querying
- In-memory + file-backed caches for candle/feature workflows

### Why this matters for profit
- Cheap local query/scan cycles reduce iteration latency on regime slices and validation windows
- Data-path efficiency supports broader regime coverage (which is more valuable than over-optimizing one slice)

## Runtime / Trading Execution Stack
- Python CLI runtime (`run.py`) for paper and live-preview modes
- Local execution adapters under `execution/`
- Coinbase order preview integration path (adapter-based)
- Local candle data primarily sourced from Binance historical files

### Why this matters for profit
- Runtime simplicity improves auditability of trading decisions and risk controls
- Adapter boundaries make execution behavior measurable without forcing premature broker abstraction work

## Evaluation / Governance Stack
- Walk-forward backtesting (`hrm_walkforward_backtest.py`)
- Parameter sweeps (`hrm_walkforward_sweep.py`)
- Simmer manager / gated promotion loop (`hrm_simmer_manager.py`)
- Calibration governance agent (`hrm_trade_head_calibration_governor.py`)
- Metrics + JSON artifact reporting for auditability

### Why this matters for profit
- This is the profit-protection layer: it converts experiments into governed decisions
- Backtests/sweeps/calibration-governor/simmer gating prevent accidental promotion of regressions
- OOS calibration and multi-slice validation directly improve trust in executable edge estimates

## UI / Monitoring Stack
- **Streamlit** dashboard (`dashboard.py`)
- **Plotly** charts (`plotly.graph_objects`, `plotly.express`)
- Python HTTP daemon (`trainerd.py`) using `http.server` for a vanilla web console
- Static console assets under `console/`

## Interface / API Artifacts
- OpenAPI spec artifact present: `trainerd.openapi.yaml`

## Development / Runtime Environment (Observed)
- Git repository (active brownfield with local experimentation artifacts)
- Local virtual environment present (`venv/`)
- macOS-style local development environment indicators present

## Data / Artifact Conventions (Observed)
- Model artifacts under `models/trained/`
- Walk-forward outputs under `walkforward_results/` and `walkforward_sweeps/`
- Simmer cycle outputs under `simmer_runs*/`
- Reports under `reports/`

## Architecture Notes (Observed)
- Monorepo-style organization with shared Python modules for:
  - training
  - runtime execution
  - backtesting/evaluation
  - calibration/veto analysis
  - governance/promotion loops
- GOALS.md describes a draw-thru architecture (source -> duckdb/cache -> pandas -> codecs -> HRM)

## World-Space Representation Alignment (GOALS-Oriented)
The stack aligns with the project goal that the HRM’s value depends on efficiently representing the market world-space:
- **draw-thru data path** supports consistent feature generation from local market data
- **shared HRM/world-model code** supports iterative improvements to representation quality
- **evaluation/governance tooling** tests whether better representation actually improves trading outcomes

Practical rule:
- representation improvements that do not survive calibration + regime-aware validation are not treated as profit improvements
- autograd is the only reliable way to push those representation improvements into the shared HRM weights at scale

## Known Constraints / Compatibility Notes
- MLX paths degrade gracefully when MLX is unavailable in some scripts
- No standard Python package manifest was detected (`pyproject.toml`, `requirements.txt`, etc. absent)
- Tooling and dependencies are currently inferred from imports and runtime scripts

## Near-Term Tech Stack Priorities (Profit-Driven)
- Preserve local iteration speed on training/backtest/governance loops
- Strengthen regime-manifest and promotion governance before adding platform complexity
- Improve calibration and representation-evaluation linkage before broadening infrastructure surface area
