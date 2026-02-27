from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_history import build_history_index


def test_build_history_index_collects_and_sorts_snapshots(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "hrm_freqtrade_fidelity_report_20260225_010000.md").write_text("r1")
    (runtime / "hrm_freqtrade_fidelity_report_20260225_020000.md").write_text("r2")
    (runtime / "hrm_freqtrade_fidelity_compare_report_20260225_015000.md").write_text("c1")

    payload = build_history_index(runtime, limit=10)

    assert payload["counts"]["report_snapshots"] == 2
    assert payload["counts"]["compare_snapshots"] == 1
    rows = payload["rows"]
    assert rows[0]["name"] == "hrm_freqtrade_fidelity_report_20260225_020000.md"
    assert any(r["kind"] == "compare" for r in rows)
