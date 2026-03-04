# Spec: Moneyfan Drawdown Source Artifacts for Freqtrade Insights

## Objectives

1. Generate deterministic drawdown-oriented pretesting/paper source artifacts in moneyfan.
2. Keep all operator-facing insights centralized in `freqtrade`.
3. Preserve stable schema contracts for cross-repo ingestion and reconciliation.

## Acceptance Criteria

- Pretesting source artifacts are deterministic and schema-validated.
- Paper testing telemetry includes threshold and `signal_id` linkage fields.
- Source manifests include Kotlingrad expression references where available.
- Freqtrade ingestion smoke check can consume moneyfan source artifacts without schema drift.
