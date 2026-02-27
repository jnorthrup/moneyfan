# Pair-Context Sampler Trace Report

- Generated: `2026-02-26T00:24:19.210554+00:00`
- Trace JSONL: `/Users/jim/work/moneyfan/runtime/pair_context_sampler_audit_smoke/pair_context_sampler_trace.jsonl`

## Summary

- rows_total=2 rows_valid=2 parse_errors=0 schema_filtered=0
- pair_width: count=2 min=2.0 max=3.0 mean=2.5 p50=2.5 p95=2.9499999999999997
- focal_pair_inclusion_failures=0

## Pair Width Histogram

| value | count |
|---|---:|
| `2` | 1 |
| `3` | 1 |

## Exchange Targets

| value | count |
|---|---:|
| `coinbase_advanced` | 2 |

## Data Sources

| value | count |
|---|---:|
| `binance` | 2 |

## Sampler Versions

| value | count |
|---|---:|
| `smoke_v1` | 2 |

## Sampler Policies

| value | count |
|---|---:|
| `ranked_stochastic_topk` | 2 |

## Ranker Names

| value | count |
|---|---:|
| `exchange_pair_ranker` | 2 |

## Ranker Versions

| value | count |
|---|---:|
| `smoke_ranker_v1` | 2 |

## Top Focal Pairs

| pair | count |
|---|---:|
| `BTC/USD` | 1 |
| `ETH/USD` | 1 |
