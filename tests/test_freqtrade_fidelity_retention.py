from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_retention import prune_runtime_artifacts


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _set_age_days(path: Path, days: int) -> None:
    old_ts = time.time() - (days * 86400)
    path.touch()
    path.write_text(path.read_text())
    path.stat()
    import os

    os.utime(path, (old_ts, old_ts))


def test_prune_runtime_artifacts_selects_old_snapshots_and_stale_files(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    # Stable latest artifacts should be preserved from stale pruning.
    _touch(runtime / "hrm_freqtrade_fidelity_report.md")
    _touch(runtime / "hrm_freqtrade_fidelity_compare_report.md")
    _touch(runtime / "hrm_freqtrade_fidelity_reconciliation.json")
    _touch(runtime / "hrm_freqtrade_fidelity_reconciliation.csv")

    # Timestamped snapshots (keep 1 each).
    _touch(runtime / "hrm_freqtrade_fidelity_report_20260225_010000.md")
    _touch(runtime / "hrm_freqtrade_fidelity_report_20260225_020000.md")
    _touch(runtime / "hrm_freqtrade_fidelity_compare_report_20260225_010000.md")
    _touch(runtime / "hrm_freqtrade_fidelity_compare_report_20260225_020000.md")

    stale_jsonl = runtime / "freqtrade_dispatch_ack.jsonl"
    _touch(stale_jsonl)
    _set_age_days(stale_jsonl, 30)

    fresh_jsonl = runtime / "freqtrade_handoff.jsonl"
    _touch(fresh_jsonl)

    summary = prune_runtime_artifacts(
        runtime_dir=runtime,
        keep_report_snapshots=1,
        keep_compare_snapshots=1,
        stale_days=14,
        dry_run=True,
    )

    planned_paths = {row["path"] for row in summary["planned"]}
    assert str(runtime / "hrm_freqtrade_fidelity_report_20260225_010000.md") in planned_paths
    assert str(runtime / "hrm_freqtrade_fidelity_compare_report_20260225_010000.md") in planned_paths
    assert str(stale_jsonl) in planned_paths
    assert str(fresh_jsonl) not in planned_paths
    assert str(runtime / "hrm_freqtrade_fidelity_report.md") not in planned_paths
    assert summary["counts"]["planned"] == 3


def test_prune_runtime_artifacts_executes_delete(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    old_snapshot = runtime / "hrm_freqtrade_fidelity_report_20260225_000000.md"
    _touch(old_snapshot)

    summary = prune_runtime_artifacts(
        runtime_dir=runtime,
        keep_report_snapshots=0,
        keep_compare_snapshots=0,
        stale_days=999,
        dry_run=False,
    )

    assert str(old_snapshot) in summary["deleted"]
    assert not old_snapshot.exists()
