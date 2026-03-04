from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_handoff_bridge import (
    handoff_to_freqtrade_webhook_payload,
    load_bridge_state,
    process_handoff_batch,
    read_jsonl_batch_from_offset,
    validate_receiver_response_contract_v1,
)


def _sample_handoff(signal_id: str = "hrm-123") -> dict:
    return {
        "schema": "moneyfan.freqtrade.handoff.v1",
        "signal_id": signal_id,
        "pair": "BTC/USDT",
        "symbol": "BTCUSDT",
        "side": "long",
        "enter_long": 1,
        "enter_short": 0,
        "stake_fraction": 0.4,
        "stoploss": -0.02,
        "take_profit_pct": 0.03,
        "risk": {"risk_tier": "caution"},
        "model": {
            "confidence": 0.82,
            "pred_fwd_return": 0.011,
            "score": 13.2,
            "score_mode": "net_effective_predicted_edge_bps",
            "passes_edge_gate": True,
            "net_effective_predicted_edge_bps": 24.5,
            "trade_head_calibration_loaded": True,
            "raw_vetoed": False,
            "raw_veto_reason": None,
            "veto_overridden": False,
        },
        "dispatch": {
            "iteration": 7,
            "source_mode": "paper",
            "source_broker_label": "freqtrade",
        },
    }


def test_handoff_to_freqtrade_webhook_payload_preserves_signal_and_hrm_metadata():
    payload = handoff_to_freqtrade_webhook_payload(_sample_handoff("hrm-xyz"))

    assert payload["schema"] == "moneyfan.freqtrade.bridge.webhook.v1"
    assert payload["signal_id"] == "hrm-xyz"
    assert payload["pair"] == "BTC/USDT"
    assert payload["side"] == "long"
    assert payload["action"] == "enter_long"
    assert payload["enter_long"] == 1
    assert payload["enter_short"] == 0
    assert payload["stake_fraction"] == 0.4
    assert payload["stoploss"] == -0.02
    assert payload["take_profit_pct"] == 0.03
    assert payload["metadata"]["hrm"]["confidence"] == 0.82
    assert payload["metadata"]["hrm"]["trade_head_calibration_loaded"] is True
    assert payload["metadata"]["source_dispatch"]["iteration"] == 7


def test_handoff_to_freqtrade_webhook_payload_requires_signal_id_for_fidelity():
    row = _sample_handoff("hrm-temp")
    row["signal_id"] = ""

    try:
        handoff_to_freqtrade_webhook_payload(row)
    except ValueError as e:
        assert "signal_id" in str(e)
    else:
        raise AssertionError("expected ValueError for missing signal_id")


def test_read_jsonl_batch_from_offset_tracks_offsets_and_handles_truncate(tmp_path):
    handoff_path = tmp_path / "handoff.jsonl"
    rows = [_sample_handoff("hrm-a"), _sample_handoff("hrm-b")]
    handoff_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    batch1, offset1 = read_jsonl_batch_from_offset(handoff_path, 0, max_records=1)
    assert len(batch1) == 1
    assert batch1[0]["signal_id"] == "hrm-a"
    assert offset1 > 0

    batch2, offset2 = read_jsonl_batch_from_offset(handoff_path, offset1)
    assert len(batch2) == 1
    assert batch2[0]["signal_id"] == "hrm-b"
    assert offset2 >= offset1

    # Simulate rotation/truncate.
    handoff_path.write_text(json.dumps(_sample_handoff("hrm-c")) + "\n")
    batch3, offset3 = read_jsonl_batch_from_offset(handoff_path, offset2)
    assert len(batch3) == 1
    assert batch3[0]["signal_id"] == "hrm-c"
    assert offset3 > 0


def test_process_handoff_batch_dry_run_writes_ack_and_state(tmp_path):
    handoff_path = tmp_path / "handoff.jsonl"
    state_path = tmp_path / "bridge_state.json"
    ack_log_path = tmp_path / "dispatch_ack.jsonl"
    handoff_path.write_text(json.dumps(_sample_handoff("hrm-dry")) + "\n")

    summary = process_handoff_batch(
        handoff_path=handoff_path,
        state_path=state_path,
        ack_log_path=ack_log_path,
        webhook_url=None,
    )

    assert summary["processed"] == 1
    assert summary["forwarded"] == 1
    assert summary["failed"] == 0
    assert summary["dry_run"] is True
    assert summary["next_offset"] > 0

    ack_lines = ack_log_path.read_text().strip().splitlines()
    assert len(ack_lines) == 1
    ack = json.loads(ack_lines[0])
    assert ack["schema"] == "moneyfan.freqtrade.dispatch_ack.v1"
    assert ack["status"] == "dry_run_forwarded"
    assert ack["signal_id"] == "hrm-dry"
    assert ack["action"] == "enter_long"

    state = load_bridge_state(state_path)
    assert state["schema"] == "moneyfan.freqtrade.bridge_state.v1"
    assert state["offset"] == summary["next_offset"]


def test_validate_receiver_response_contract_v1_accepts_matching_signal_id():
    body = json.dumps(
        {
            "ok": True,
            "accepted": True,
            "signal_id": "hrm-123",
            "receiver_schema": "your.freqtrade.receiver.accept.v1",
            "freqtrade_request_id": "ftreq-1",
        }
    )
    parsed = validate_receiver_response_contract_v1(body, expected_signal_id="hrm-123")
    assert parsed["accepted"] is True
    assert parsed["signal_id"] == "hrm-123"


def test_process_handoff_batch_production_v1_marks_invalid_receiver_response(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoff.jsonl"
    state_path = tmp_path / "bridge_state.json"
    ack_log_path = tmp_path / "dispatch_ack.jsonl"
    handoff_path.write_text(json.dumps(_sample_handoff("hrm-prod")) + "\n")

    def _fake_post(url, payload, timeout_seconds=5.0):
        return {
            "ok": True,
            "http_status": 200,
            "response_body": json.dumps({"ok": True, "accepted": True, "signal_id": "wrong-id"}),
        }

    import execution.freqtrade_handoff_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "post_webhook_json", _fake_post)
    summary = process_handoff_batch(
        handoff_path=handoff_path,
        state_path=state_path,
        ack_log_path=ack_log_path,
        webhook_url="http://127.0.0.1:9999/trade-update",
        receiver_profile="production_v1",
    )

    assert summary["processed"] == 1
    assert summary["forwarded"] == 0
    assert summary["failed"] == 1
    assert summary["receiver_profile"] == "production_v1"

    ack = json.loads(ack_log_path.read_text().strip())
    assert ack["status"] == "webhook_contract_invalid_response"
    assert "signal_id mismatch" in ack["error"]
