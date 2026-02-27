#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from execution.freqtrade_fidelity_reconcile import extract_fill_view


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception as e:
                rows.append(
                    {
                        "_parse_error": str(e),
                        "_line_no": i,
                        "_raw_line": raw[:2000],
                    }
                )
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                rows.append({"value": parsed, "_line_no": i})
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(row)) + "\n")


def canonicalize_fill_event(
    raw_row: Dict[str, Any],
    include_raw: bool = False,
) -> Dict[str, Any]:
    source_row = raw_row
    if str(raw_row.get("schema", "") or "") == "moneyfan.freqtrade.trade_update_ingest.v1":
        payload = raw_row.get("payload")
        if isinstance(payload, dict):
            source_row = payload

    view = extract_fill_view(source_row)
    signal_id = str(view.get("signal_id", "") or "").strip()
    if not signal_id:
        raise ValueError("Missing signal_id in fill/trade row")

    event = {
        "schema": "moneyfan.freqtrade.fill_event.v1",
        "ts_utc": utc_now_iso(),
        "signal_id": signal_id,
        "pair": view.get("pair"),
        "side": view.get("side"),
        "status": view.get("status"),
        "entry_price": view.get("entry_price"),
        "exit_price": view.get("exit_price"),
        "pnl_abs": view.get("pnl_abs"),
        "pnl_pct": view.get("pnl_pct"),
        "fees_abs": view.get("fees_abs"),
        "fill_ts_utc": view.get("fill_ts_utc"),
        "exchange_trade_id": view.get("exchange_trade_id"),
        "source_schema": view.get("fill_schema"),
    }
    if include_raw:
        event["raw"] = raw_row
    if source_row is not raw_row:
        event["receiver_ingest"] = {
            "received_at_utc": raw_row.get("received_at_utc"),
            "source_path": raw_row.get("source_path"),
            "client_ip": raw_row.get("client_ip"),
        }
    return event


def _dedupe_key(event: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(event.get("signal_id", "") or ""),
        str(event.get("status", "") or ""),
        str(event.get("exchange_trade_id", "") or ""),
        str(event.get("fill_ts_utc", "") or ""),
        event.get("entry_price"),
        event.get("exit_price"),
        event.get("pnl_abs"),
        event.get("pnl_pct"),
    )


def normalize_fill_jsonl(
    input_path: Path,
    output_path: Path,
    reject_log_path: Optional[Path] = None,
    include_raw: bool = False,
    dedupe: bool = False,
    reset_output: bool = False,
) -> Dict[str, Any]:
    rows = read_jsonl(input_path)
    if reset_output and output_path.exists():
        output_path.unlink()
    if reset_output and reject_log_path and reject_log_path.exists():
        reject_log_path.unlink()

    total = 0
    forwarded = 0
    rejected = 0
    parse_errors = 0
    missing_signal_id = 0
    duplicates_skipped = 0
    seen = set()

    for row in rows:
        total += 1
        if row.get("_parse_error"):
            parse_errors += 1
            rejected += 1
            if reject_log_path is not None:
                append_jsonl(
                    reject_log_path,
                    {
                        "schema": "moneyfan.freqtrade.fill_event_reject.v1",
                        "ts_utc": utc_now_iso(),
                        "reason": "invalid_jsonl_record",
                        "error": row.get("_parse_error"),
                        "line_no": row.get("_line_no"),
                        "raw_line": row.get("_raw_line"),
                    },
                )
            continue
        try:
            event = canonicalize_fill_event(row, include_raw=include_raw)
        except Exception as e:
            if "signal_id" in str(e):
                missing_signal_id += 1
            rejected += 1
            if reject_log_path is not None:
                append_jsonl(
                    reject_log_path,
                    {
                        "schema": "moneyfan.freqtrade.fill_event_reject.v1",
                        "ts_utc": utc_now_iso(),
                        "reason": "normalization_error",
                        "error": str(e),
                        "source_schema": row.get("schema"),
                        "signal_id": row.get("signal_id"),
                    },
                )
            continue

        if dedupe:
            k = _dedupe_key(event)
            if k in seen:
                duplicates_skipped += 1
                continue
            seen.add(k)

        append_jsonl(output_path, event)
        forwarded += 1

    return {
        "schema": "moneyfan.freqtrade.fill_event_normalize_summary.v1",
        "generated_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "reject_log_path": str(reject_log_path) if reject_log_path else None,
        "total_rows_seen": int(total),
        "events_written": int(forwarded),
        "rejected_rows": int(rejected),
        "parse_errors": int(parse_errors),
        "missing_signal_id": int(missing_signal_id),
        "duplicates_skipped": int(duplicates_skipped),
        "include_raw": bool(include_raw),
        "dedupe": bool(dedupe),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize raw Freqtrade trade/fill JSONL into moneyfan canonical fill-event JSONL")
    p.add_argument("--in-jsonl", type=str, required=True,
                   help="Input raw JSONL (Freqtrade trade updates/exports) containing signal_id")
    p.add_argument("--out-jsonl", type=str, default="runtime/freqtrade_fill_events.jsonl",
                   help="Output canonical fill-event JSONL for fidelity reconciliation")
    p.add_argument("--reject-log-path", type=str, default="runtime/freqtrade_fill_event_rejects.jsonl",
                   help="Optional JSONL reject log (set empty string to disable)")
    p.add_argument("--include-raw", action="store_true",
                   help="Embed raw source row in normalized output (larger files)")
    p.add_argument("--dedupe", action="store_true",
                   help="Skip duplicate normalized events within this run")
    p.add_argument("--reset-output", action="store_true",
                   help="Delete output/reject files before writing this run")
    p.add_argument("--print-summary", action="store_true",
                   help="Print normalization summary JSON")
    return p


def run_cli(args: argparse.Namespace) -> int:
    reject_path = str(args.reject_log_path or "").strip()
    summary = normalize_fill_jsonl(
        input_path=Path(args.in_jsonl),
        output_path=Path(args.out_jsonl),
        reject_log_path=(Path(reject_path) if reject_path else None),
        include_raw=bool(args.include_raw),
        dedupe=bool(args.dedupe),
        reset_output=bool(args.reset_output),
    )
    if bool(args.print_summary):
        print(json.dumps(summary, indent=2))
    else:
        print(
            "✅ Fill-event normalize "
            f"rows={summary['total_rows_seen']} written={summary['events_written']} "
            f"rejected={summary['rejected_rows']} missing_signal_id={summary['missing_signal_id']} "
            f"dupes={summary['duplicates_skipped']}"
        )
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
