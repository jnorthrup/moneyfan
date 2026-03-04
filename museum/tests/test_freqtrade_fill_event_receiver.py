from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fill_event_receiver import FillEventReceiverConfig, FreqtradeFillEventReceiver


def _receiver(tmp_path: Path, **cfg_overrides) -> FreqtradeFillEventReceiver:
    cfg = FillEventReceiverConfig(
        raw_log_path=str(tmp_path / "raw_ingest.jsonl"),
        fill_event_log_path=str(tmp_path / "fill_events.jsonl"),
        reject_log_path=str(tmp_path / "rejects.jsonl"),
        **cfg_overrides,
    )
    return FreqtradeFillEventReceiver(cfg)


def test_process_trade_update_payload_writes_raw_and_canonical_fill_event(tmp_path):
    receiver = _receiver(tmp_path)
    payload = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "signal_id": "sig-rx-1",
            "pair": "BTC/USDT",
            "is_short": False,
            "is_open": False,
            "open_rate": 50000.0,
            "close_rate": 50100.0,
            "id": "trade-1",
        },
    }

    status, resp = receiver.process_trade_update_payload(payload, source_path="/trade-update", client_ip="127.0.0.1")

    assert status == 200
    assert resp["ok"] is True
    assert resp["signal_id"] == "sig-rx-1"
    assert resp["schema"] == "moneyfan.freqtrade.fill_event.v1"

    raw_rows = [json.loads(x) for x in (tmp_path / "raw_ingest.jsonl").read_text().splitlines() if x.strip()]
    assert len(raw_rows) == 1
    assert raw_rows[0]["schema"] == "moneyfan.freqtrade.trade_update_ingest.v1"
    assert raw_rows[0]["payload"]["trade"]["signal_id"] == "sig-rx-1"

    fill_rows = [json.loads(x) for x in (tmp_path / "fill_events.jsonl").read_text().splitlines() if x.strip()]
    assert len(fill_rows) == 1
    assert fill_rows[0]["schema"] == "moneyfan.freqtrade.fill_event.v1"
    assert fill_rows[0]["signal_id"] == "sig-rx-1"
    assert fill_rows[0]["receiver_ingest"]["source_path"] == "/trade-update"
    assert fill_rows[0]["receiver_ingest"]["client_ip"] == "127.0.0.1"

    stats = receiver.stats_snapshot()["stats"]
    assert stats["requests_total"] == 1
    assert stats["canonical_events_written"] == 1
    assert stats["rejects_written"] == 0
    assert stats["last_signal_id"] == "sig-rx-1"


def test_process_trade_update_payload_rejects_missing_signal_id(tmp_path):
    receiver = _receiver(tmp_path)
    payload = {
        "schema": "freqtrade.trade_update",
        "trade": {"pair": "ETH/USDT", "open_rate": 100.0, "close_rate": 101.0},
    }

    status, resp = receiver.process_trade_update_payload(payload, source_path="/fill", client_ip="10.0.0.1")

    assert status == 400
    assert resp["ok"] is False
    assert "signal_id" in resp["error"]

    # Raw ingest is still captured for audit/debugging.
    raw_rows = [json.loads(x) for x in (tmp_path / "raw_ingest.jsonl").read_text().splitlines() if x.strip()]
    assert len(raw_rows) == 1

    reject_rows = [json.loads(x) for x in (tmp_path / "rejects.jsonl").read_text().splitlines() if x.strip()]
    assert len(reject_rows) == 1
    assert reject_rows[0]["schema"] == "moneyfan.freqtrade.fill_event_receiver_reject.v1"
    assert reject_rows[0]["reason"] == "canonicalize_error"

    stats = receiver.stats_snapshot()["stats"]
    assert stats["requests_total"] == 1
    assert stats["canonical_events_written"] == 0
    assert stats["rejects_written"] == 1


def test_process_http_json_body_rejects_invalid_json_and_non_object_payload(tmp_path):
    receiver = _receiver(tmp_path)

    status_bad_json, resp_bad_json = receiver.process_http_json_body(
        body_bytes=b"{invalid json}",
        source_path="/trade-update",
        client_ip="127.0.0.1",
    )
    assert status_bad_json == 400
    assert resp_bad_json["ok"] is False
    assert "invalid JSON" in resp_bad_json["error"]

    status_list, resp_list = receiver.process_http_json_body(
        body_bytes=b"[1,2,3]",
        source_path="/trade-update",
        client_ip="127.0.0.1",
    )
    assert status_list == 400
    assert resp_list["ok"] is False
    assert "JSON object" in resp_list["error"]

    reject_rows = [json.loads(x) for x in (tmp_path / "rejects.jsonl").read_text().splitlines() if x.strip()]
    assert len(reject_rows) == 2
    reasons = {r["reason"] for r in reject_rows}
    assert reasons == {"json_parse_error", "payload_type_error"}

    stats = receiver.stats_snapshot()["stats"]
    assert stats["requests_total"] == 1  # only non-object payload reaches request counter
    assert stats["json_parse_errors"] == 1
    assert stats["payload_type_errors"] == 1
