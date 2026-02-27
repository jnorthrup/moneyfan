from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_contract_path_smoke import run_contract_path_smoke


def test_run_contract_path_smoke_propagates_signal_id_and_writes_bridge_proxy_logs(tmp_path: Path):
    summary = run_contract_path_smoke(tmp_path / "runtime")
    checks = summary["checks"]

    assert checks["ack_signal_id_matches"] is True
    assert checks["ack_status"] == "webhook_forwarded"
    assert checks["proxy_ingest_seen"] is True
    assert checks["proxy_dispatch_seen"] is True
    assert checks["proxy_dispatch_status"] in {"downstream_forwarded", "downstream_forward_failed"}
    assert checks["fill_receiver_raw_seen"] is True
