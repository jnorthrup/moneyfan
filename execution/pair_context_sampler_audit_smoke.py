#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from execution.pair_context_sampler_conformance import validate_muxer_sampler_conformance, write_json as write_json_report
from execution.pair_context_sampler_trace import build_pair_context_sampler_trace, write_pair_context_sampler_traces
from execution.pair_context_sampler_trace_report import (
    build_markdown_report,
    build_pair_context_sampler_trace_report,
    write_text,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _write_jsonl(path: Path, rows: List[Dict[str, Any]], reset_output: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset_output and path.exists():
        path.unlink()
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row)) + "\n")


def build_sample_muxer_rows(*, exchange_target: str, data_source: str) -> List[Dict[str, Any]]:
    _ = (exchange_target, data_source)  # reserved for future fixture variation
    return [
        {
            "pair": "BTC/USD",
            "symbol": "BTCUSD",
            "ts_utc": "2026-02-26T15:00:00Z",
            "price": 64000.0,
            "side": "long",
            "signal_id": "sig-sampler-smoke-001",
        },
        {
            "pair": "ETH/USD",
            "symbol": "ETHUSD",
            "ts_utc": "2026-02-26T15:00:00Z",
            "price": 3500.0,
            "side": "short",
            "signal_id": "sig-sampler-smoke-002",
        },
    ]


def build_sample_traces(*, exchange_target: str, data_source: str) -> List[Dict[str, Any]]:
    md = {
        "sampler_schema": "moneyfan.pair_context_sampler.v1",
        "sampler_version": "smoke_v1",
        "sampler_policy": "ranked_stochastic_topk",
        "ranker_name": "exchange_pair_ranker",
        "ranker_version": "smoke_ranker_v1",
        "ranker_score_timestamp_policy": "point_in_time",
        "exchange_target": exchange_target,
        "data_source": data_source,
        "universe_filter_version": "smoke_universe_v1",
        "candidate_universe_size": 12,
    }
    return [
        build_pair_context_sampler_trace(
            frame_id="frame-smoke-001",
            frame_ts_utc="2026-02-26T15:00:00Z",
            focal_pair="BTC/USD",
            slot_pairs=["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"],
            slot_mask=[1, 1, 1, 0],
            slot_features=[{"z": 1.1}, {"z": 0.8}, {"z": 0.4}, None],
            sampling_metadata=md,
            max_pair_width=8,
            slot_ordering="rank_desc",
            model_slot_order_invariant=False,
        ),
        build_pair_context_sampler_trace(
            frame_id="frame-smoke-002",
            frame_ts_utc="2026-02-26T15:01:00Z",
            focal_pair="ETH/USD",
            slot_pairs=["ETH/USD", "BTC/USD", "LINK/USD", "ADA/USD"],
            slot_mask=[1, 1, 0, 0],
            slot_features=[{"z": 0.9}, {"z": 1.2}, None, None],
            sampling_metadata=md,
            max_pair_width=8,
            slot_ordering="rank_desc",
            model_slot_order_invariant=False,
        ),
    ]


def run_sampler_audit_smoke(
    *,
    runtime_dir: Path,
    exchange_target: str = "coinbase_advanced",
    data_source: str = "binance",
) -> Dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    muxer_rows_path = runtime_dir / "pair_context_muxer_rows.jsonl"
    trace_jsonl_path = runtime_dir / "pair_context_sampler_trace.jsonl"
    conformance_json_path = runtime_dir / "pair_context_sampler_conformance.json"
    trace_report_json_path = runtime_dir / "pair_context_sampler_trace_report.json"
    trace_report_md_path = runtime_dir / "pair_context_sampler_trace_report.md"
    summary_json_path = runtime_dir / "pair_context_sampler_audit_smoke_summary.json"

    muxer_rows = build_sample_muxer_rows(exchange_target=exchange_target, data_source=data_source)
    _write_jsonl(muxer_rows_path, muxer_rows, reset_output=True)

    conformance = validate_muxer_sampler_conformance(
        muxer_rows,
        exchange_target=exchange_target,
        data_source=data_source,
    )
    write_json_report(conformance_json_path, conformance)
    if conformance.get("result") != "pass":
        raise RuntimeError("sampler audit smoke conformance failed unexpectedly")

    traces = build_sample_traces(exchange_target=exchange_target, data_source=data_source)
    trace_write_summary = write_pair_context_sampler_traces(trace_jsonl_path, traces, reset_output=True)

    trace_report = build_pair_context_sampler_trace_report(traces)
    with open(trace_report_json_path, "w") as f:
        json.dump(_json_safe(trace_report), f, indent=2)
    write_text(trace_report_md_path, build_markdown_report(trace_report, trace_path=str(trace_jsonl_path)))

    summary = {
        "schema": "moneyfan.pair_context_sampler_audit_smoke.v1",
        "generated_at_utc": utc_now_iso(),
        "context": {
            "exchange_target": exchange_target,
            "data_source": data_source,
        },
        "paths": {
            "runtime_dir": str(runtime_dir),
            "muxer_rows_jsonl": str(muxer_rows_path),
            "trace_jsonl": str(trace_jsonl_path),
            "conformance_json": str(conformance_json_path),
            "trace_report_json": str(trace_report_json_path),
            "trace_report_md": str(trace_report_md_path),
        },
        "results": {
            "conformance_result": conformance.get("result"),
            "trace_rows_written": int(trace_write_summary.get("rows_written", 0) or 0),
            "trace_report_rows_valid": int(
                ((trace_report.get("summary") or {}).get("rows_valid", 0))  # type: ignore[union-attr]
            ),
            "pair_width_p95": ((trace_report.get("summary") or {}).get("pair_width_stats") or {}).get("p95"),
            "focal_pair_inclusion_failures": int(
                ((trace_report.get("summary") or {}).get("focal_pair_inclusion_failures", 0))  # type: ignore[union-attr]
            ),
        },
    }
    with open(summary_json_path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    summary["paths"]["summary_json"] = str(summary_json_path)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end smoke test for sampler audit workflow (muxer conformance + trace report)")
    p.add_argument("--runtime-dir", type=str, default="runtime/pair_context_sampler_audit_smoke")
    p.add_argument("--exchange-target", type=str, default="coinbase_advanced")
    p.add_argument("--data-source", type=str, default="binance")
    p.add_argument("--print-summary", action="store_true")
    args = p.parse_args()

    summary = run_sampler_audit_smoke(
        runtime_dir=Path(args.runtime_dir),
        exchange_target=str(args.exchange_target or ""),
        data_source=str(args.data_source or ""),
    )
    if bool(args.print_summary):
        r = summary["results"]
        print(
            "✅ Sampler audit smoke "
            f"conformance={r['conformance_result']} "
            f"trace_rows={r['trace_rows_written']} "
            f"rows_valid={r['trace_report_rows_valid']} "
            f"p95={r['pair_width_p95']}"
        )
        print(f"📝 {summary['paths']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
