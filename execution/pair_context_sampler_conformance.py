#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


REQUIRED_COLUMNS_DEFAULT = [
    "pair",
    "symbol",
    "ts_utc",
]

RECOMMENDED_COLUMNS_DEFAULT = [
    "price",
    "side",
    "signal_id",
]


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                rows.append({"_parse_error": str(e), "_line_no": i, "_raw_line": raw[:500]})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({"value": obj, "_line_no": i})
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _normalize_records(records: Any) -> List[Dict[str, Any]]:
    # Optional pandas support without hard dependency.
    if hasattr(records, "to_dict"):
        try:
            rows = records.to_dict(orient="records")  # type: ignore[call-arg]
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            pass
    if isinstance(records, list):
        return [r for r in records if isinstance(r, dict)]
    raise TypeError("records must be a list[dict] or pandas DataFrame-like object")


def _missing_columns(columns: Sequence[str], required: Sequence[str]) -> List[str]:
    colset = set(columns)
    return [c for c in required if c not in colset]


def _bad_ts(rows: Iterable[Dict[str, Any]], ts_col: str) -> List[Dict[str, Any]]:
    out = []
    for idx, r in enumerate(rows):
        v = r.get(ts_col)
        s = str(v or "").strip()
        if not s:
            out.append({"row_index": idx, "reason": "missing_timestamp"})
            continue
        try:
            # tolerate Z suffix
            datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            out.append({"row_index": idx, "reason": "invalid_timestamp", "value": s})
    return out


def _null_violations(rows: Iterable[Dict[str, Any]], columns: Sequence[str]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        for c in columns:
            if c not in r or r.get(c) is None or str(r.get(c)).strip() == "":
                violations.append({"row_index": idx, "column": c})
    return violations


def validate_muxer_sampler_conformance(
    records: Any,
    *,
    required_columns: Optional[Sequence[str]] = None,
    recommended_columns: Optional[Sequence[str]] = None,
    non_nullable_columns: Optional[Sequence[str]] = None,
    timestamp_column: str = "ts_utc",
    require_monotonic_ts: bool = False,
    exchange_target: str = "",
    data_source: str = "",
) -> Dict[str, Any]:
    rows = _normalize_records(records)
    required = list(required_columns or REQUIRED_COLUMNS_DEFAULT)
    recommended = list(recommended_columns or RECOMMENDED_COLUMNS_DEFAULT)
    non_nullable = list(non_nullable_columns or ["pair", "ts_utc"])

    parsed_rows = [r for r in rows if not r.get("_parse_error")]
    parse_error_rows = [r for r in rows if r.get("_parse_error")]
    columns: List[str] = []
    seen = set()
    for r in parsed_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    missing_required = _missing_columns(columns, required)
    missing_recommended = _missing_columns(columns, recommended)
    null_violations = _null_violations(parsed_rows, non_nullable)
    ts_violations = _bad_ts(parsed_rows, timestamp_column)

    monotonic_violations: List[Dict[str, Any]] = []
    if require_monotonic_ts and not ts_violations and timestamp_column in columns:
        prev_dt = None
        for idx, r in enumerate(parsed_rows):
            cur = datetime.fromisoformat(str(r.get(timestamp_column)).replace("Z", "+00:00"))
            if prev_dt is not None and cur < prev_dt:
                monotonic_violations.append({"row_index": idx, "reason": "timestamp_decreased"})
            prev_dt = cur

    result = "pass"
    if parse_error_rows or missing_required or null_violations or ts_violations or monotonic_violations:
        result = "fail"

    return {
        "schema": "moneyfan.pair_context_sampler_conformance.v1",
        "generated_at_utc": utc_now_iso(),
        "result": result,
        "context": {
            "exchange_target": str(exchange_target or ""),
            "data_source": str(data_source or ""),
        },
        "summary": {
            "rows_total": len(rows),
            "rows_parsed": len(parsed_rows),
            "parse_error_rows": len(parse_error_rows),
            "columns_seen": columns,
            "missing_required_columns": missing_required,
            "missing_recommended_columns": missing_recommended,
            "null_violations": len(null_violations),
            "timestamp_violations": len(ts_violations),
            "monotonic_timestamp_violations": len(monotonic_violations),
        },
        "details": {
            "parse_errors": parse_error_rows[:50],
            "null_violations": null_violations[:200],
            "timestamp_violations": ts_violations[:200],
            "monotonic_timestamp_violations": monotonic_violations[:200],
        },
        "contract": {
            "required_columns": required,
            "recommended_columns": recommended,
            "non_nullable_columns": non_nullable,
            "timestamp_column": str(timestamp_column),
            "require_monotonic_ts": bool(require_monotonic_ts),
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate pandas muxer output conformance for pair-context sampler consumption")
    p.add_argument("--in-jsonl", type=str, default="", help="Input muxer rows as JSONL")
    p.add_argument("--in-csv", type=str, default="", help="Input muxer rows as CSV")
    p.add_argument("--out-json", type=str, default="runtime/pair_context_sampler_conformance.json")
    p.add_argument("--required-columns", type=str, default="pair,symbol,ts_utc")
    p.add_argument("--recommended-columns", type=str, default="price,side,signal_id")
    p.add_argument("--non-nullable-columns", type=str, default="pair,ts_utc")
    p.add_argument("--timestamp-column", type=str, default="ts_utc")
    p.add_argument("--require-monotonic-ts", action="store_true")
    p.add_argument("--exchange-target", type=str, default="coinbase_advanced")
    p.add_argument("--data-source", type=str, default="binance")
    p.add_argument("--print-summary", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    in_jsonl = str(args.in_jsonl or "").strip()
    in_csv = str(args.in_csv or "").strip()
    if not in_jsonl and not in_csv:
        raise SystemExit("Provide --in-jsonl or --in-csv")
    if in_jsonl and in_csv:
        raise SystemExit("Provide only one of --in-jsonl or --in-csv")

    rows = _read_jsonl(Path(in_jsonl)) if in_jsonl else _read_csv(Path(in_csv))
    report = validate_muxer_sampler_conformance(
        rows,
        required_columns=[c for c in str(args.required_columns).split(",") if c],
        recommended_columns=[c for c in str(args.recommended_columns).split(",") if c],
        non_nullable_columns=[c for c in str(args.non_nullable_columns).split(",") if c],
        timestamp_column=str(args.timestamp_column),
        require_monotonic_ts=bool(args.require_monotonic_ts),
        exchange_target=str(args.exchange_target or ""),
        data_source=str(args.data_source or ""),
    )
    out_json = Path(args.out_json)
    write_json(out_json, report)
    if bool(args.print_summary):
        s = report["summary"]
        print(
            "✅" if report["result"] == "pass" else "❌",
            "Pair-context sampler conformance",
            f"result={report['result']}",
            f"rows={s['rows_parsed']}/{s['rows_total']}",
            f"missing_required={len(s['missing_required_columns'])}",
            f"nulls={s['null_violations']}",
            f"ts_violations={s['timestamp_violations']}",
        )
        print(f"📝 {out_json}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
