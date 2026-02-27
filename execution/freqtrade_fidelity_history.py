#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_TS_RE = re.compile(r"_(\d{8}_\d{6})")


def _parse_stamp(name: str) -> Optional[str]:
    m = _TS_RE.search(name)
    return m.group(1) if m else None


def _parse_dt(stamp: str) -> Optional[datetime]:
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _collect(runtime_dir: Path, prefix: str, kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in runtime_dir.glob(f"{prefix}_*.md"):
        if not p.is_file():
            continue
        stamp = _parse_stamp(p.name)
        dt = _parse_dt(stamp) if stamp else None
        rows.append(
            {
                "kind": kind,
                "path": str(p),
                "name": p.name,
                "stamp": stamp,
                "ts_utc": dt.isoformat() if dt else None,
                "size_bytes": p.stat().st_size,
            }
        )
    rows.sort(key=lambda r: (r.get("stamp") or "", r["name"]), reverse=True)
    return rows


def build_history_index(runtime_dir: Path, limit: int = 20) -> Dict[str, Any]:
    runtime_dir = Path(runtime_dir)
    reports = _collect(runtime_dir, "hrm_freqtrade_fidelity_report", "report")
    compares = _collect(runtime_dir, "hrm_freqtrade_fidelity_compare_report", "compare")
    all_rows = sorted(reports + compares, key=lambda r: (r.get("stamp") or "", r["name"]), reverse=True)
    if int(limit) > 0:
        all_rows = all_rows[: int(limit)]
    return {
        "schema": "moneyfan.freqtrade.fidelity_history_index.v1",
        "runtime_dir": str(runtime_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "report_snapshots": len(reports),
            "compare_snapshots": len(compares),
            "returned": len(all_rows),
        },
        "rows": all_rows,
    }


def _print_table(payload: Dict[str, Any]) -> None:
    print(f"History index for {payload['runtime_dir']}")
    c = payload["counts"]
    print(f"report_snapshots={c['report_snapshots']} compare_snapshots={c['compare_snapshots']} returned={c['returned']}")
    for row in payload.get("rows", []):
        print(f"- [{row['kind']}] {row.get('stamp') or 'n/a'} {row['name']} ({row['size_bytes']} bytes)")


def main() -> int:
    p = argparse.ArgumentParser(description="List timestamped HRM/Freqtrade fidelity report snapshot history")
    p.add_argument("--runtime-dir", type=str, default="runtime", help="Runtime artifact directory")
    p.add_argument("--limit", type=int, default=20, help="Max rows to return (0 = all)")
    p.add_argument("--print-json", action="store_true", help="Print JSON instead of table")
    args = p.parse_args()
    payload = build_history_index(Path(args.runtime_dir), limit=int(args.limit))
    if bool(args.print_json):
        print(json.dumps(payload, indent=2))
    else:
        _print_table(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
