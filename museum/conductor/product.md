# Product Guide

## Product Name
moneyfan

## Product Summary
moneyfan is a local-first HRM (Hierarchical Reasoning Model) crypto trading system for a solo quant operator. It combines stochastic training, regime-aware signal allocation/execution, and operational safety gates (backtests, calibration governance, validation/promotion) to improve trading performance without blind model drift.

## Primary User (Priority 1)
### Solo Quant Operator (Local Runtime + Paper/Live Preview)
A technically capable individual who wants to:
- train and evaluate an HRM trading model on local market data
- run paper trading and live-preview execution safely
- improve the system continuously with measurable promotion gates
- inspect performance, risk, and model behavior without enterprise infrastructure

## User Goals
- Generate risk-controlled alpha in crypto markets using a regime-aware HRM stack
- Improve model performance iteratively without overfitting to a single slice/regime
- Maintain operational safety with rollback, validation gating, and audit trails
- Run the full workflow locally (training, backtest, paper trading, governance)

## Core Product Capabilities
### 1. Data + Feature Pipeline (Draw-thru)
- Ingest local exchange candle data (primarily Binance parquet/DuckDB-backed workflows)
- Compute normalized indicator/features efficiently for HRM input
- Maintain a reusable feature schema for checkpoint compatibility

### 2. HRM Training + Checkpointing
- Stochastic episode-based training for regime robustness
- HRM checkpoint save/load (weights, model config, feature schema)
- Low-rate continual learning ("simmer") support with guarded promotion

### 3. Trading Runtime
- Paper trading and live-preview execution loop
- Ranked signal selection (`top-k`) with confidence/risk controls
- Position management (hold/cooldown) and risk-head sanity repair
- Mechanical veto and configurable veto override policies (auditable)

### 4. Evaluation + Profit Governance
- Walk-forward backtests with fees/slippage
- Parameter sweeps for trading profile optimization
- Calibration fitting and OOS calibration governance (fit/sweep/promote)
- Multi-slice validation gating before promoting model changes

### 5. Operator Observability
- Metrics outputs (`metrics.json`, `trades.json`, `equity_curve.json`)
- Reports for calibration governance and veto impact attribution
- Training dashboard / daemon interfaces for monitoring

## Product Principles
- Profit-first, but safety-gated
- Local reproducibility over hidden magic
- World-model quality matters only if it improves trading decisions
- Every promotion must be measurable, reversible, and logged
- Regime coverage beats single-slice wins

## Success Criteria (Near-Term)
- Stable paper-trading operation with reproducible backtest/paper alignment
- Promotion pipeline prevents obvious regressions in equity/DD/PF across validation slices
- Calibration governance improves trade-head magnitude realism OOS
- Solo operator can run train -> validate -> paper loops with minimal manual patching

## Non-Goals (Current Phase)
- Enterprise multi-user platform
- Fully automated live trading at scale without operator supervision
- Broker/exchange abstraction completeness across all venues
- UI polish over core model/evaluation/profit infrastructure

## Constraints
- Brownfield codebase with active experimentation and dirty working tree risk
- Local compute and local data availability shape throughput and evaluation cadence
- Market regimes change; optimization must prioritize robustness and rollback paths

## Product Direction (Current)
The immediate focus is building a robust agent-managed profit improvement loop around the HRM:
- safe calibration governance
- regime-aware validation and promotion
- continuous simmering with rollback
- better alignment between world-model outputs and executable trading edge
