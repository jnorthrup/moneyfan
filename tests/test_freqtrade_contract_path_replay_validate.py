from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_contract_path_replay_validate import (
    build_replay_validation_markdown,
    resolve_threshold_profile,
    validate_replay_summary,
)


def _summary_template():
    return {
        "schema": "moneyfan.freqtrade.contract_path_replay.v1",
        "exchange_target": "coinbase_advanced",
        "data_source": "binance",
        "params": {"batches": 2, "batch_size": 2, "bridge_max_records": 0},
        "counts": {
            "emitted_handoffs": 4,
            "ack_rows": 4,
            "proxy_dispatch_rows": 4,
            "fill_event_rows": 4,
            "fill_reject_rows": 0,
            "proxy_reject_rows": 0,
        },
        "checks": {
            "all_acks_forwarded": True,
            "all_ack_signal_ids_seen": True,
            "all_proxy_signal_ids_seen": True,
            "all_fill_signal_ids_seen": True,
            "fill_receiver_reject_count": 0,
            "proxy_reject_count": 0,
        },
    }


def test_validate_replay_summary_passes_with_zero_rejects_and_full_signal_visibility():
    validation = validate_replay_summary(_summary_template())
    assert validation["result"] == "pass"
    assert validation["counts"]["forward_rate"] == 1.0
    assert validation["context"]["exchange_target"] == "coinbase_advanced"
    assert validation["context"]["data_source"] == "binance"


def test_validate_replay_summary_fails_on_reject_threshold():
    s = _summary_template()
    s["counts"]["fill_reject_rows"] = 2
    s["checks"]["fill_receiver_reject_count"] = 2
    validation = validate_replay_summary(s, max_fill_rejects=0)
    assert validation["result"] == "fail"
    assert any(f["rule"] == "max_fill_rejects" for f in validation["failures"])


def test_build_replay_validation_markdown_includes_context_and_result():
    validation = validate_replay_summary(_summary_template())
    md = build_replay_validation_markdown(validation, replay_json_path="/tmp/replay.json")
    assert "# Contract Path Replay Validation Report" in md
    assert "`exchange_target`: `coinbase_advanced`" in md
    assert "`data_source`: `binance`" in md
    assert "`pass`" in md
    assert "/tmp/replay.json" in md


def test_resolve_threshold_profile_uses_exchange_target_data_source_combo():
    profile = resolve_threshold_profile(exchange_target="coinbase_advanced", data_source="binance")
    assert profile["min_forward_rate"] == 1.0
    assert profile["max_fill_rejects"] == 0
    assert profile["max_proxy_rejects"] == 0


def test_resolve_threshold_profile_supports_relaxed_named_profile():
    profile = resolve_threshold_profile(profile_name="coinbase_advanced__binance_relaxed")
    assert profile["min_forward_rate"] == 0.95
    assert profile["max_fill_rejects"] == 1
    assert profile["max_proxy_rejects"] == 0
