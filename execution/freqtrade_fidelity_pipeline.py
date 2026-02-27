#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from execution.freqtrade_handoff_bridge import process_handoff_batch
from execution.freqtrade_fill_event_normalizer import normalize_fill_jsonl
from execution.freqtrade_fidelity_reconcile import reconcile_from_paths, write_csv, write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_fidelity_pipeline(
    handoff_path: Path,
    bridge_state_path: Path,
    ack_log_path: Path,
    dispatch_log_path: Path,
    raw_fill_updates_path: Path,
    canonical_fill_events_path: Path,
    reconciliation_json_path: Path,
    reconciliation_csv_path: Path,
    webhook_url: Optional[str] = None,
    bridge_timeout_seconds: float = 5.0,
    bridge_max_records: int = 0,
    normalizer_reject_log_path: Optional[Path] = None,
    normalizer_dedupe: bool = True,
    normalizer_reset_output: bool = False,
    require_success_ack_for_fill_match: bool = False,
    print_payloads: bool = False,
    skip_bridge: bool = False,
    exchange_target: str = "",
    data_source: str = "",
) -> Dict[str, Any]:
    if bool(skip_bridge):
        bridge_summary = {
            "skipped": True,
            "reason": "skip_bridge",
            "dry_run": bool(not (webhook_url or "").strip()) if isinstance(webhook_url, str) else (webhook_url is None),
            "processed": 0,
            "forwarded": 0,
            "failed": 0,
        }
    else:
        bridge_summary = process_handoff_batch(
            handoff_path=handoff_path,
            state_path=bridge_state_path,
            ack_log_path=ack_log_path,
            webhook_url=webhook_url,
            timeout_seconds=float(bridge_timeout_seconds),
            max_records=int(bridge_max_records or 0),
            print_payloads=bool(print_payloads),
        )

    normalize_summary = normalize_fill_jsonl(
        input_path=raw_fill_updates_path,
        output_path=canonical_fill_events_path,
        reject_log_path=normalizer_reject_log_path,
        dedupe=bool(normalizer_dedupe),
        reset_output=bool(normalizer_reset_output),
    )

    report = reconcile_from_paths(
        dispatch_log_path=dispatch_log_path,
        ack_log_path=ack_log_path,
        fill_log_path=canonical_fill_events_path,
        require_success_ack_for_fill_match=bool(require_success_ack_for_fill_match),
        exchange_target=str(exchange_target or ""),
        data_source=str(data_source or ""),
    )
    write_json(reconciliation_json_path, report)
    write_csv(reconciliation_csv_path, list(report.get("records", [])))

    pipeline_summary = {
        "schema": "moneyfan.freqtrade.fidelity_pipeline_run.v1",
        "generated_at_utc": utc_now_iso(),
        "bridge": bridge_summary,
        "normalize": normalize_summary,
        "reconcile_summary": report.get("summary", {}),
        "context": {
            "exchange_target": str(exchange_target or ""),
            "data_source": str(data_source or ""),
        },
        "artifacts": {
            "handoff_path": str(handoff_path),
            "bridge_state_path": str(bridge_state_path),
            "ack_log_path": str(ack_log_path),
            "dispatch_log_path": str(dispatch_log_path),
            "raw_fill_updates_path": str(raw_fill_updates_path),
            "canonical_fill_events_path": str(canonical_fill_events_path),
            "reconciliation_json_path": str(reconciliation_json_path),
            "reconciliation_csv_path": str(reconciliation_csv_path),
            "normalizer_reject_log_path": str(normalizer_reject_log_path) if normalizer_reject_log_path else None,
        },
    }
    return pipeline_summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run local Freqtrade offload fidelity pipeline (bridge -> normalize -> reconcile)")
    p.add_argument("--handoff-path", type=str, default="runtime/freqtrade_handoff.jsonl")
    p.add_argument("--bridge-state-path", type=str, default="runtime/freqtrade_handoff_bridge_state.json")
    p.add_argument("--ack-log-path", type=str, default="runtime/freqtrade_dispatch_ack.jsonl")
    p.add_argument("--raw-fill-updates-path", type=str, default="runtime/freqtrade_trade_updates_raw.jsonl",
                   help="Raw trade update ingest JSONL (e.g., from freqtrade_fill_event_receiver)")
    p.add_argument("--canonical-fill-events-path", type=str, default="runtime/freqtrade_fill_events.jsonl")
    p.add_argument("--normalizer-reject-log-path", type=str, default="runtime/freqtrade_fill_event_rejects.jsonl")
    p.add_argument("--dispatch-log-path", type=str, default="runtime/hrm_fidelity_dispatch.jsonl")
    p.add_argument("--reconciliation-json-path", type=str, default="runtime/hrm_freqtrade_fidelity_reconciliation.json")
    p.add_argument("--reconciliation-csv-path", type=str, default="runtime/hrm_freqtrade_fidelity_reconciliation.csv")
    p.add_argument("--webhook-url", type=str, default="", help="Bridge webhook target (empty = dry-run)")
    p.add_argument("--bridge-timeout-seconds", type=float, default=5.0)
    p.add_argument("--bridge-max-records", type=int, default=0)
    p.add_argument("--normalizer-dedupe", action="store_true", default=True)
    p.add_argument("--no-normalizer-dedupe", action="store_false", dest="normalizer_dedupe")
    p.add_argument("--normalizer-reset-output", action="store_true")
    p.add_argument("--require-success-ack-for-fill-match", action="store_true")
    p.add_argument("--print-payloads", action="store_true")
    p.add_argument("--skip-bridge", action="store_true",
                   help="Replay mode: skip bridge dispatch and only normalize + reconcile existing artifacts")
    p.add_argument("--exchange-target", type=str, default="",
                   help="Target execution venue / exchange abstraction label (e.g. coinbase_advanced)")
    p.add_argument("--data-source", type=str, default="",
                   help="Primary data source label for this evaluation context (e.g. binance)")
    p.add_argument("--print-summary", action="store_true")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_fidelity_pipeline(
        handoff_path=Path(args.handoff_path),
        bridge_state_path=Path(args.bridge_state_path),
        ack_log_path=Path(args.ack_log_path),
        dispatch_log_path=Path(args.dispatch_log_path),
        raw_fill_updates_path=Path(args.raw_fill_updates_path),
        canonical_fill_events_path=Path(args.canonical_fill_events_path),
        reconciliation_json_path=Path(args.reconciliation_json_path),
        reconciliation_csv_path=Path(args.reconciliation_csv_path),
        webhook_url=(str(args.webhook_url).strip() or None),
        bridge_timeout_seconds=float(args.bridge_timeout_seconds),
        bridge_max_records=int(args.bridge_max_records or 0),
        normalizer_reject_log_path=(Path(args.normalizer_reject_log_path) if str(args.normalizer_reject_log_path or "").strip() else None),
        normalizer_dedupe=bool(args.normalizer_dedupe),
        normalizer_reset_output=bool(args.normalizer_reset_output),
        require_success_ack_for_fill_match=bool(args.require_success_ack_for_fill_match),
        print_payloads=bool(args.print_payloads),
        skip_bridge=bool(args.skip_bridge),
        exchange_target=str(args.exchange_target or ""),
        data_source=str(args.data_source or ""),
    )
    if bool(args.print_summary):
        print(json.dumps(summary, indent=2))
    else:
        rs = summary.get("reconcile_summary", {}) if isinstance(summary.get("reconcile_summary"), dict) else {}
        print(
            "✅ Fidelity pipeline "
            f"bridge_processed={summary.get('bridge', {}).get('processed', 0)} "
            f"fill_events_written={summary.get('normalize', {}).get('events_written', 0)} "
            f"reconciled={rs.get('dispatch_fully_reconciled', 0)} "
            f"dispatch_total={rs.get('dispatch_total', 0)}"
        )
        print(f"📝 {summary['artifacts']['reconciliation_json_path']}")
        print(f"🧾 {summary['artifacts']['reconciliation_csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
