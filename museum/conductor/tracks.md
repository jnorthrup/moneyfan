# Project Tracks

This file tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

---

## [~] Track: Implement a profit-driven MLX HRM training loop (autograd objective telemetry + controls + traceable artifacts) - Freqtrade Integration Priority
**URGENT:** This track is blocking Freqtrade alpha release. HRM models must be deployable and servable via Freqtrade ring agent.

**Priority:** URGENT - Model development robustness for Freqtrade alpha
**Impact:** 
- ✅ **Model Development:** Completes HRM training pipeline
- ✅ **Freqtrade Integration:** Provides models for ring agent evaluation
- ✅ **Agent Harness:** Requires model serving capability

*Link: [./conductor/tracks/hrm_autograd_profit_governance_20260223/](./conductor/tracks/hrm_autograd_profit_governance_20260223/)*

## [~] Track: Freqtrade offload + HRM fidelity audit loop (runtime offload, reconciliation, reports, runbook)
*Link: [./conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/](./conductor/tracks/freqtrade_offload_hrm_fidelity_20260225/)*

## [ ] Track: Pretesting + Paper Testing Drawdown Source Artifacts (Freqtrade Insights Sink)

**Objective:** Produce deterministic drawdown source artifacts from moneyfan and hand them off to `freqtrade`, where all insights are published.

*Link: [./conductor/tracks/pretesting-paper-drawdown-kotlingrad-dsel_20260302/](./conductor/tracks/pretesting-paper-drawdown-kotlingrad-dsel_20260302/)*

## [ ] Track: Runtime Drawdown Guardrails and Rollback-Safe Execution

**Objective:** Add deterministic runtime drawdown protection (`warn`/`de-risk`/`halt`) with auditable artifacts and explicit resume controls for safer paper/live-preview operation.

*Link: [./conductor/tracks/runtime_drawdown_guardrails_rollback_20260303/](./conductor/tracks/runtime_drawdown_guardrails_rollback_20260303/)*
