# Profit Debt TODO (Working List)

This is the current debt list from shipping the HRM trading runner/backtests toward net-positive behavior.

## Done (low debt, high ROI)

- [x] Walk-forward backtest + sweep harness (fees/slippage, top-k, hold/cooldown)
- [x] Simmer manager with gated promote/restore loop
- [x] Veto displacement audit (`VetoCFTopK`, `VetoDisplacedSlots`, shadow PnL)
- [x] Reason-whitelisted confidence override for soft veto reasons
- [x] Pre-veto risk-head repair layer (stop/target sanity repair)
- [x] Post-hoc trade-head calibration artifact fitter + loader (magnitude shrink by confidence bins)

## Next (least-debt money-first)

- [ ] Sweep calibration sensitivity (`--min-scale`, confidence bins, sample windows)
  - Current fitted median realized/predicted move ratio is very low (~0.025), so calibration floor matters.
- [x] Rewrite ranking score around calibrated expected net edge (not raw `pred_fwd_return`)
  - `score_mode` now uses calibrated net predicted edge after estimated costs.
- [x] Add per-reason veto shadow PnL summary
  - Walk-forward metrics now emit per-reason displaced shadow PnL / gross profit / gross loss / PF dicts.
- [x] Add per-reason veto shadow PnL slices report
  - `hrm_veto_reason_shadow_report.py` aggregates reason counts + shadow PnL across walk-forward/simmer results.
- [ ] Add veto reason report automation / regression watch
  - Periodically rerun the veto reason report and alert when `CFTopK`/displacement becomes non-zero.
- [x] Add multi-slice simmer promotion gate (trend + chop + high-vol slices)
  - `hrm_simmer_manager.py` now supports `--validation-slices-json` and per-slice promotion aggregation.
- [x] Add weighted multi-slice promotion gate
  - `hrm_simmer_manager.py` now supports per-slice `weight` and `mandatory` fields with weighted win/mean gates.
- [ ] Add regime manifest + mandatory coverage policy
  - Promote against a canonical GOALS-aligned regime manifest (trend/chop/high-vol/shock), not ad hoc slice JSON.
- [x] Add OOS calibration governor (fit/sweep/promote on held-out trades files)
  - `hrm_trade_head_calibration_governor.py` now does file-split holdout validation and gated promotion.
- [x] Integrate calibration governor into simmer agent chain
  - `hrm_simmer_manager.py` now runs calibration governor as a pre-validation substep and logs decision/report in cycle summary.
- [ ] Strengthen OOS calibration governance split policy
  - Current holdout is by trades-file split (newest files); tighten to explicit regime/time-window governance and non-overlapping evaluation corpora.
- [ ] Add calibration governor cadence/trigger policy in simmer
  - Run every N cycles or only after enough new trades accumulate, instead of every cycle.

## Medium Debt (important for stability)

- [ ] Confidence calibration (reliability mapping), not just move-magnitude calibration
- [ ] Regime-aware threshold scheduling (higher threshold in chop, lower in trend)
- [ ] Better cooldown/hold policy tuning by symbol and volatility bucket
- [ ] Calibration drift monitoring (auto-expire stale calibration artifacts)

## High Debt (largest gap)

- [ ] Cost-aware trade-head training objective (turnover penalty, cost penalty)
- [ ] Stronger trade-step usage vs mostly world-model optimization
- [ ] Trade-head target redesign / label calibration for TP/SL realism
- [ ] Multi-regime validation dataset curation and OOS governance
- [ ] Execution realism upgrades (latency/impact assumptions)

## Notes from recent evidence

- Risk-head repair removed `stop_too_tight` vetoes on tested slices without loosening safety veto policy.
- Trade-head calibration materially improved tested walk-forward slices by reducing overconfident magnitude assumptions.
- Calibration governor found and promoted a better OOS calibration candidate (`wMAE 43.42 -> 25.21 bps` on held-out trades-file split).
- Veto was often blamed, but measured top-k displacement was zero on several tested slices; ranking/calibration debt is bigger.
- Cross-run veto reason report currently shows `stop_too_tight` dominates raw vetoes, with zero observed counterfactual top-k displacement in the scanned corpus.
