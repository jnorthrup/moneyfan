#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


_TS_RE = re.compile(r"_(\d{8}_\d{6})")


@dataclass
class PruneDecision:
    path: Path
    reason: str


def _parse_ts_from_name(path: Path) -> datetime | None:
    m = _TS_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _sorted_snapshot_files(runtime_dir: Path, prefix: str) -> List[Path]:
    rows: List[tuple[datetime, Path]] = []
    for path in runtime_dir.glob(f"{prefix}_*.md"):
        if not path.is_file():
            continue
        ts = _parse_ts_from_name(path) or _mtime_utc(path)
        rows.append((ts, path))
    rows.sort(key=lambda x: (x[0], x[1].name), reverse=True)
    return [p for _, p in rows]


def select_snapshot_prunes(runtime_dir: Path, keep_report_snapshots: int, keep_compare_snapshots: int) -> List[PruneDecision]:
    decisions: List[PruneDecision] = []
    report_files = _sorted_snapshot_files(runtime_dir, "hrm_freqtrade_fidelity_report")
    compare_files = _sorted_snapshot_files(runtime_dir, "hrm_freqtrade_fidelity_compare_report")

    for path in report_files[max(0, int(keep_report_snapshots)) :]:
        decisions.append(PruneDecision(path=path, reason="report_snapshot_retention"))
    for path in compare_files[max(0, int(keep_compare_snapshots)) :]:
        decisions.append(PruneDecision(path=path, reason="compare_snapshot_retention"))
    return decisions


def select_stale_runtime_prunes(
    runtime_dir: Path,
    stale_days: int,
    preserve_names: Sequence[str] | None = None,
) -> List[PruneDecision]:
    preserve = set(preserve_names or [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(stale_days)))
    stale_patterns = (
        "*.jsonl",
        "*.csv",
        "*.json",
        "*.log",
    )
    decisions: List[PruneDecision] = []
    seen: set[Path] = set()
    for pattern in stale_patterns:
        for path in runtime_dir.glob(pattern):
            if not path.is_file():
                continue
            if path.name in preserve:
                continue
            if path in seen:
                continue
            seen.add(path)
            if _mtime_utc(path) < cutoff:
                decisions.append(PruneDecision(path=path, reason=f"stale_runtime_file>{int(stale_days)}d"))
    decisions.sort(key=lambda d: str(d.path))
    return decisions


def execute_prune(decisions: Iterable[PruneDecision], dry_run: bool = False) -> Dict[str, Any]:
    deleted: List[str] = []
    failed: List[Dict[str, str]] = []
    for d in decisions:
        if dry_run:
            deleted.append(str(d.path))
            continue
        try:
            os.remove(d.path)
            deleted.append(str(d.path))
        except FileNotFoundError:
            continue
        except Exception as e:
            failed.append({"path": str(d.path), "reason": d.reason, "error": str(e)})
    return {"deleted": deleted, "failed": failed}


def prune_runtime_artifacts(
    runtime_dir: Path,
    keep_report_snapshots: int = 14,
    keep_compare_snapshots: int = 14,
    stale_days: int = 14,
    dry_run: bool = False,
) -> Dict[str, Any]:
    runtime_dir = Path(runtime_dir)
    preserve_names = {
        "hrm_freqtrade_fidelity_reconciliation.json",
        "hrm_freqtrade_fidelity_reconciliation.csv",
        "hrm_freqtrade_fidelity_report.md",
        "hrm_freqtrade_fidelity_compare_report.md",
    }
    snapshot_decisions = select_snapshot_prunes(runtime_dir, keep_report_snapshots, keep_compare_snapshots)
    stale_decisions = select_stale_runtime_prunes(runtime_dir, stale_days=stale_days, preserve_names=tuple(preserve_names))

    by_path = {str(d.path): d for d in snapshot_decisions}
    for d in stale_decisions:
        by_path.setdefault(str(d.path), d)
    decisions = sorted(by_path.values(), key=lambda d: str(d.path))

    result = execute_prune(decisions, dry_run=dry_run)
    payload: Dict[str, Any] = {
        "schema": "moneyfan.freqtrade.fidelity_retention_prune.v1",
        "runtime_dir": str(runtime_dir),
        "dry_run": bool(dry_run),
        "settings": {
            "keep_report_snapshots": int(keep_report_snapshots),
            "keep_compare_snapshots": int(keep_compare_snapshots),
            "stale_days": int(stale_days),
        },
        "counts": {
            "planned": len(decisions),
            "deleted": len(result["deleted"]),
            "failed": len(result["failed"]),
        },
        "planned": [{"path": str(d.path), "reason": d.reason} for d in decisions],
        "deleted": result["deleted"],
        "failed": result["failed"],
    }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune old Freqtrade/HRM fidelity runtime artifacts and report snapshots")
    p.add_argument("--runtime-dir", type=str, default="runtime", help="Runtime artifact directory")
    p.add_argument("--keep-report-snapshots", type=int, default=14, help="Keep newest N single-run report snapshots")
    p.add_argument("--keep-compare-snapshots", type=int, default=14, help="Keep newest N compare report snapshots")
    p.add_argument("--stale-days", type=int, default=14, help="Prune runtime JSON/JSONL/CSV/log files older than N days")
    p.add_argument("--dry-run", action="store_true", help="Print planned deletions only")
    p.add_argument("--print-json", action="store_true", help="Print full JSON summary")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = prune_runtime_artifacts(
        runtime_dir=Path(args.runtime_dir),
        keep_report_snapshots=int(args.keep_report_snapshots),
        keep_compare_snapshots=int(args.keep_compare_snapshots),
        stale_days=int(args.stale_days),
        dry_run=bool(args.dry_run),
    )
    if bool(args.print_json):
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        mode = "DRY-RUN" if summary["dry_run"] else "PRUNE"
        counts = summary["counts"]
        print(
            f"{mode} runtime={summary['runtime_dir']} planned={counts['planned']} "
            f"deleted={counts['deleted']} failed={counts['failed']}"
        )
        for row in summary["planned"][:20]:
            print(f"- {row['reason']}: {row['path']}")
        if len(summary["planned"]) > 20:
            print(f"- ... and {len(summary['planned']) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
