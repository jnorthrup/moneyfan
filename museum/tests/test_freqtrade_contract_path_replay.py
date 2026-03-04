from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_contract_path_replay import run_contract_path_replay


def test_run_contract_path_replay_multiple_batches_propagates_all_signal_ids(tmp_path: Path):
    summary = run_contract_path_replay(tmp_path / "runtime", batches=2, batch_size=3)
    counts = summary["counts"]
    checks = summary["checks"]

    assert counts["emitted_handoffs"] == 6
    assert counts["ack_rows"] == 6
    assert counts["proxy_dispatch_rows"] == 6
    assert counts["fill_event_rows"] == 6
    assert checks["all_acks_forwarded"] is True
    assert checks["all_ack_signal_ids_seen"] is True
    assert checks["all_proxy_signal_ids_seen"] is True
    assert checks["all_fill_signal_ids_seen"] is True
    assert checks["fill_receiver_reject_count"] == 0
    assert checks["proxy_reject_count"] == 0
