from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_contract_receiver_proxy import (
    ContractProxyConfig,
    FreqtradeContractReceiverProxy,
    build_downstream_payload,
)


def _proxy(tmp_path: Path, **overrides) -> FreqtradeContractReceiverProxy:
    cfg = ContractProxyConfig(
        ingest_log_path=str(tmp_path / "ingest.jsonl"),
        dispatch_log_path=str(tmp_path / "dispatch.jsonl"),
        reject_log_path=str(tmp_path / "rejects.jsonl"),
        **overrides,
    )
    return FreqtradeContractReceiverProxy(cfg)


def _bridge_payload(signal_id: str = "sig-1") -> dict:
    return {
        "schema": "moneyfan.freqtrade.bridge.webhook.v1",
        "ts_utc": "2026-02-25T22:10:00Z",
        "signal_id": signal_id,
        "pair": "BTC/USDT",
        "side": "long",
        "action": "enter_long",
        "enter_long": 1,
        "enter_short": 0,
        "stake_fraction": 0.1,
        "stoploss": -0.02,
        "take_profit_pct": 0.03,
        "metadata": {"source_schema": "moneyfan.freqtrade.handoff.v1", "hrm": {}},
    }


def test_process_bridge_payload_accepts_and_returns_contract_response(tmp_path: Path):
    proxy = _proxy(tmp_path)
    status, resp = proxy.process_bridge_payload(_bridge_payload("sig-accept"), source_path="/signal", client_ip="127.0.0.1")

    assert status == 200
    assert resp["ok"] is True
    assert resp["accepted"] is True
    assert resp["signal_id"] == "sig-accept"
    assert resp["receiver_schema"] == "moneyfan.freqtrade.receiver.accept.v1"
    assert resp["freqtrade_request_id"]

    ingest = [json.loads(x) for x in (tmp_path / "ingest.jsonl").read_text().splitlines() if x.strip()]
    dispatch = [json.loads(x) for x in (tmp_path / "dispatch.jsonl").read_text().splitlines() if x.strip()]
    assert ingest[0]["payload"]["signal_id"] == "sig-accept"
    assert dispatch[0]["signal_id"] == "sig-accept"
    assert dispatch[0]["status"] == "accepted_dry_run"


def test_process_bridge_payload_rejects_invalid_contract_payload(tmp_path: Path):
    proxy = _proxy(tmp_path)
    bad = {"schema": "moneyfan.freqtrade.bridge.webhook.v1", "pair": "BTC/USDT", "side": "long"}
    status, resp = proxy.process_bridge_payload(bad, source_path="/signal", client_ip="127.0.0.1")

    assert status == 400
    assert resp["ok"] is False
    assert resp["accepted"] is False
    assert "signal_id" in resp["error"]

    rejects = [json.loads(x) for x in (tmp_path / "rejects.jsonl").read_text().splitlines() if x.strip()]
    assert rejects[0]["reason"] == "contract_validation_error"


def test_build_downstream_payload_freqtrade_webhook_v1_preserves_signal_id():
    downstream = build_downstream_payload(
        bridge_payload=_bridge_payload("sig-map"),
        mode="freqtrade_webhook_v1",
        freqtrade_request_id="mfproxy-123",
    )
    assert downstream["schema"] == "moneyfan.freqtrade.proxy.forward.freqtrade_webhook_v1"
    assert downstream["pair"] == "BTC/USDT"
    assert downstream["action"] == "buy"
    assert downstream["metadata"]["signal_id"] == "sig-map"
    assert downstream["metadata"]["freqtrade_request_id"] == "mfproxy-123"


def test_process_bridge_payload_forwards_mapped_downstream_payload(monkeypatch, tmp_path: Path):
    sent = {}

    def _fake_post(url, payload, timeout_seconds=5.0):
        sent["url"] = url
        sent["payload"] = payload
        return {"ok": True, "http_status": 200, "response_body": '{"ok": true}'}

    import execution.freqtrade_contract_receiver_proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "post_json", _fake_post)
    proxy = _proxy(
        tmp_path,
        downstream_webhook_url="http://127.0.0.1:9999/ft",
        downstream_payload_mode="freqtrade_webhook_v1",
    )

    status, resp = proxy.process_bridge_payload(_bridge_payload("sig-fwd"), source_path="/signal", client_ip="127.0.0.1")
    assert status == 200
    assert resp["accepted"] is True
    assert sent["url"] == "http://127.0.0.1:9999/ft"
    assert sent["payload"]["schema"] == "moneyfan.freqtrade.proxy.forward.freqtrade_webhook_v1"
    assert sent["payload"]["metadata"]["signal_id"] == "sig-fwd"

    dispatch = [json.loads(x) for x in (tmp_path / "dispatch.jsonl").read_text().splitlines() if x.strip()]
    assert dispatch[0]["downstream_payload_mode"] == "freqtrade_webhook_v1"
    assert dispatch[0]["downstream_payload_schema"] == "moneyfan.freqtrade.proxy.forward.freqtrade_webhook_v1"
