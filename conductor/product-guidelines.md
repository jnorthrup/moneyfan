# Product Guidelines

## Purpose
These guidelines define how moneyfan should communicate, present information, and structure operator-facing experiences so the system remains useful for a solo quant focused on profitable, risk-controlled execution.

The project is a trading/research/operations tool first. Clarity, auditability, and decision speed take priority over polish for its own sake.

## Primary Communication Principle
**Direct, quantitative, and operationally useful.**

Every interface (CLI, reports, dashboards, docs) should help the operator answer:
1. What changed?
2. Did it improve expected profit/risk behavior?
3. Is it safe to promote/run?
4. What should I do next?

## Audience and Context
### Primary Audience
- Solo quant operator / builder running local training, backtests, paper trading, and promotion workflows

### Secondary Audience
- Research engineer contributing to model/training/governance improvements
- Future collaborators reviewing experiments and promotion decisions

## Voice and Tone
### Default Tone
- Direct
- Factual
- Quantitative
- Operational
- Unambiguous

### Tone Rules
- Prefer measured statements over hype
- Prefer evidence over claims
- Prefer explicit risk caveats over implied safety
- Avoid marketing language in internal/operator tooling
- Avoid vague success language like "looks good" without metrics

## Writing Style
### General
- Lead with outcome, then method, then evidence
- Use short sections and scannable bullets
- Use consistent metric names across docs/reports/CLI
- Include exact paths and artifact names when relevant
- Use absolute dates/timestamps for runs and reports when ambiguity is possible

### Required Behavior for Claims
If claiming improvement, include at least:
- comparison baseline
- metric delta
- evaluation slice/scope
- whether result is in-sample or OOS
- key caveats / limitations

### Examples (Preferred)
- "OOS calibration governor improved held-out weighted MAE from 43.42 bps to 25.21 bps on 18 validation trade files."
- "Candidate rejected: failed mandatory trend slice and weighted slice-win fraction gate."

### Examples (Avoid)
- "Calibration is much better now."
- "The model is smarter."
- "Looks profitable."

## Metric and Reporting Conventions
### Core Metrics (Trading)
Always prefer consistent names and units:
- `final_equity` (currency)
- `return_pct` (%)
- `max_drawdown` (fraction, negative)
- `profit_factor`
- `sharpe_ratio`
- `total_trades`
- `win_rate`
- `total_commission`
- `total_slippage`

### Calibration Metrics
Use explicit OOS language when applicable:
- `weighted_mae_bps`
- `weighted_rmse_bps`
- `weighted_mape`
- improvement vs raw baseline
- improvement vs current calibration

### Governance Metrics
Use clear pass/fail semantics:
- `pass_eq`, `pass_dd`, `pass_trades`, `pass_pf`
- `pass_mandatory_slices`
- `weighted_slice_win_fraction`
- `weighted_mean_equity_delta`

### Units
- Price move magnitudes: `bps`
- Percent returns: `%` (human-readable) or fraction (JSON) but not mixed without labels
- Money: `$` in human-facing summaries, numeric float in machine-readable files

## CLI / Console UX Guidelines
### Output Priorities
1. Current action (what is running)
2. Key result (success/failure + decision)
3. Metrics summary (small, high-signal)
4. Paths to artifacts/reports
5. Next action (when needed)

### CLI Formatting
- Prefer compact, stable line formats
- Keep logs append-only and audit-friendly
- Print promotion decisions explicitly
- Print rollback/restore actions explicitly
- Include command-line settings only when materially relevant to interpretation

### Failure Messaging
- State the failed step and exact reason
- Include file path / command path when applicable
- Do not hide stderr if it contains root-cause information
- Distinguish:
  - hard failure (cycle abort)
  - soft failure (sub-agent skipped, loop continues)

## Dashboard / Visualization Guidelines
### Purpose of Visuals
Visuals exist to support decisions, not decoration.

### Default Dashboard Priorities
- equity / drawdown / PnL progression
- validation gate outcomes
- calibration drift / calibration quality
- trade distribution and costs
- regime-slice comparison

### Visual Style
- High contrast, readable labels
- Minimal chart clutter
- Consistent colors for:
  - positive / pass
  - negative / fail
  - neutral / baseline
- Avoid ambiguous color semantics across screens

## Risk Communication Guidelines
### Non-Negotiable
- Risk warnings should be explicit and local to the action they affect
- Promotion gates should never be described as guarantees
- Paper performance must not be described as live performance

### Required Distinctions
Always distinguish:
- in-sample vs OOS
- backtest vs paper vs live-preview
- candidate vs promoted artifact
- model improvement vs execution/config improvement

## Experimentation and Evidence Hygiene
### Experiment Outputs
Each experiment/report should make it easy to recover:
- configuration used
- artifacts loaded
- evaluation scope
- summary metrics
- decision (promote/reject/observe)

### Naming
Prefer descriptive names over novelty:
- good: `scoremode_cons_cal_on`
- good: `veto_reason_shadow_report`
- avoid names that hide purpose or evaluation scope

## Product Behavior Priorities (When Tradeoffs Conflict)
1. Safety and reversibility
2. Measurement quality / auditability
3. Profit relevance
4. Operator speed
5. Convenience / polish

## Documentation Priorities
When updating docs, prioritize:
1. operational runbooks (how to run safely)
2. evaluation/promotion governance
3. model/training rationale
4. developer ergonomics
5. polish and presentation

## Non-Goals for Product Guidelines (Current Phase)
- Consumer-facing branding system
- Marketing copy standards
- Cross-platform design system completeness
- Visual identity optimization over operator utility

## Guideline Review Trigger
Update this document when any of the following changes materially:
- promotion gate policy
- evaluation metric definitions
- calibration governance policy
- runtime execution modes (paper/live-preview/live)
- target operator persona
