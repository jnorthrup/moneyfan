#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

from execution.freqtrade_contract_receiver_proxy import (
    ContractProxyConfig,
    FreqtradeContractReceiverProxy,
    make_server as make_contract_proxy_server,
)
from execution.freqtrade_fill_event_receiver import (
    FillEventReceiverConfig,
    FreqtradeFillEventReceiver,
    make_server as make_fill_receiver_server,
)
from execution.freqtrade_handoff_bridge import process_handoff_batch


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"_parse_error": True, "_raw_line": line})
    return rows


def _sample_handoff(signal_id: str) -> Dict[str, Any]:
    return {
        "schema": "moneyfan.freqtrade.handoff.v1",
        "signal_id": signal_id,
        "pair": "BTC/USDT",
        "symbol": "BTCUSDT",
        "side": "long",
        "stake_fraction": 0.15,
        "stoploss": -0.02,
        "take_profit_pct": 0.03,
        "risk": {"risk_tier": "normal"},
        "model": {
            "confidence": 0.87,
            "pred_fwd_return": 0.0042,
            "score": 1.54,
            "score_mode": "calibrated",
            "passes_edge_gate": True,
            "net_effective_predicted_edge_bps": 28.0,
            "trade_head_calibration_loaded": True,
            "raw_vetoed": False,
            "raw_veto_reason": None,
            "veto_overridden": False,
        },
        "dispatch": {"iteration": 1, "source_mode": "paper", "source_broker_label": "freqtrade"},
    }


def run_contract_path_smoke(runtime_dir: Path) -> Dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    signal_id = "smoke-contract-path-001"

    handoff_path = runtime_dir / "handoff.jsonl"
    state_path = runtime_dir / "bridge_state.json"
    ack_log_path = runtime_dir / "dispatch_ack.jsonl"

    fill_raw = runtime_dir / "fill_receiver_raw.jsonl"
    fill_events = runtime_dir / "fill_receiver_events.jsonl"
    fill_rejects = runtime_dir / "fill_receiver_rejects.jsonl"

    proxy_ingest = runtime_dir / "contract_proxy_ingest.jsonl"
    proxy_dispatch = runtime_dir / "contract_proxy_dispatch.jsonl"
    proxy_rejects = runtime_dir / "contract_proxy_rejects.jsonl"

    handoff_path.write_text(json.dumps(_sample_handoff(signal_id)) + "\n")

    fill_port = _pick_free_port()
    proxy_port = _pick_free_port()

    fill_receiver = FreqtradeFillEventReceiver(
        FillEventReceiverConfig(
            raw_log_path=str(fill_raw),
            fill_event_log_path=str(fill_events),
            reject_log_path=str(fill_rejects),
        )
    )
    fill_server = make_fill_receiver_server("127.0.0.1", fill_port, fill_receiver)

    contract_proxy = FreqtradeContractReceiverProxy(
        ContractProxyConfig(
            ingest_log_path=str(proxy_ingest),
            dispatch_log_path=str(proxy_dispatch),
            reject_log_path=str(proxy_rejects),
            downstream_webhook_url=f"http://127.0.0.1:{fill_port}/trade-update",
            downstream_payload_mode="passthrough",
        )
    )
    proxy_server = make_contract_proxy_server("127.0.0.1", proxy_port, contract_proxy)

    threads: List[threading.Thread] = []
    for server in (fill_server, proxy_server):
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(0.15)
    try:
        bridge_summary = process_handoff_batch(
            handoff_path=handoff_path,
            state_path=state_path,
            ack_log_path=ack_log_path,
            webhook_url=f"http://127.0.0.1:{proxy_port}/signal",
            receiver_profile="production_v1",
        )
    finally:
        proxy_server.shutdown()
        fill_server.shutdown()
        proxy_server.server_close()
        fill_server.server_close()
        for t in threads:
            t.join(timeout=1.0)

    ack_rows = _read_jsonl(ack_log_path)
    proxy_ingest_rows = _read_jsonl(proxy_ingest)
    proxy_dispatch_rows = _read_jsonl(proxy_dispatch)
    proxy_reject_rows = _read_jsonl(proxy_rejects)
    fill_raw_rows = _read_jsonl(fill_raw)
    fill_event_rows = _read_jsonl(fill_events)
    fill_reject_rows = _read_jsonl(fill_rejects)

    ack = ack_rows[0] if ack_rows else {}
    proxy_dispatch_row = proxy_dispatch_rows[0] if proxy_dispatch_rows else {}

    summary = {
        "schema": "moneyfan.freqtrade.contract_path_smoke.v1",
        "signal_id": signal_id,
        "bridge": bridge_summary,
        "checks": {
            "ack_status": ack.get("status"),
            "ack_signal_id_matches": ack.get("signal_id") == signal_id,
            "proxy_ingest_seen": any((r.get("payload") or {}).get("signal_id") == signal_id for r in proxy_ingest_rows if isinstance(r, dict)),
            "proxy_dispatch_seen": proxy_dispatch_row.get("signal_id") == signal_id,
            "proxy_dispatch_status": proxy_dispatch_row.get("status"),
            "fill_receiver_raw_seen": any((r.get("payload") or {}).get("signal_id") == signal_id for r in fill_raw_rows if isinstance(r, dict)),
            "fill_receiver_canonical_count": len([r for r in fill_event_rows if isinstance(r, dict) and r.get("signal_id") == signal_id]),
            "fill_receiver_reject_count": len(fill_reject_rows),
        },
        "artifacts": {
            "ack_log_path": str(ack_log_path),
            "proxy_ingest_log_path": str(proxy_ingest),
            "proxy_dispatch_log_path": str(proxy_dispatch),
            "proxy_reject_log_path": str(proxy_rejects),
            "fill_receiver_raw_log_path": str(fill_raw),
            "fill_receiver_events_log_path": str(fill_events),
            "fill_receiver_reject_log_path": str(fill_rejects),
        },
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test bridge -> contract-proxy -> fill-receiver local contract path")
    p.add_argument("--runtime-dir", type=str, default="", help="Optional runtime dir for artifacts (default: temp dir)")
    p.add_argument("--print-json", action="store_true")
    args = p.parse_args()

    if str(args.runtime_dir or "").strip():
        summary = run_contract_path_smoke(Path(args.runtime_dir))
    else:
        with TemporaryDirectory() as d:
            summary = run_contract_path_smoke(Path(d))

    if bool(args.print_json):
        print(json.dumps(summary, indent=2))
    else:
        c = summary["checks"]
        print(
            "✅ Contract path smoke "
            f"ack={c['ack_status']} proxy_dispatch={c['proxy_dispatch_status']} "
            f"fill_raw_seen={c['fill_receiver_raw_seen']} "
            f"fill_events={c['fill_receiver_canonical_count']} rejects={c['fill_receiver_reject_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
