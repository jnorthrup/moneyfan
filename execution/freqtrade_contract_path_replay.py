#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

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
        rows.append(json.loads(line))
    return rows


def _handoff(signal_id: str, pair: str, side: str, iteration: int) -> Dict[str, Any]:
    is_long = str(side).lower() == "long"
    return {
        "schema": "moneyfan.freqtrade.handoff.v1",
        "signal_id": signal_id,
        "pair": pair,
        "symbol": pair.replace("/", ""),
        "side": side,
        "stake_fraction": 0.10 + (0.01 * (iteration % 3)),
        "stoploss": -0.02,
        "take_profit_pct": 0.03,
        "risk": {"risk_tier": "normal" if is_long else "caution"},
        "model": {
            "confidence": 0.70 + (0.02 * (iteration % 4)),
            "pred_fwd_return": 0.004 if is_long else -0.003,
            "score": 1.0 + iteration,
            "score_mode": "calibrated",
            "passes_edge_gate": True,
            "net_effective_predicted_edge_bps": 20.0 + iteration,
            "trade_head_calibration_loaded": True,
            "raw_vetoed": False,
            "raw_veto_reason": None,
            "veto_overridden": False,
        },
        "dispatch": {"iteration": iteration, "source_mode": "paper", "source_broker_label": "freqtrade"},
    }


def _sample_batch(batch_idx: int, batch_size: int) -> List[Dict[str, Any]]:
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    out: List[Dict[str, Any]] = []
    for i in range(batch_size):
        n = (batch_idx * batch_size) + i
        out.append(
            _handoff(
                signal_id=f"replay-sig-{n:04d}",
                pair=pairs[n % len(pairs)],
                side="long" if (n % 2 == 0) else "short",
                iteration=n + 1,
            )
        )
    return out


def run_contract_path_replay(
    runtime_dir: Path,
    batches: int = 3,
    batch_size: int = 4,
    bridge_max_records: int = 0,
    exchange_target: str = "",
    data_source: str = "",
) -> Dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = runtime_dir / "handoff.jsonl"
    state_path = runtime_dir / "bridge_state.json"
    ack_log_path = runtime_dir / "dispatch_ack.jsonl"
    fill_raw = runtime_dir / "fill_receiver_raw.jsonl"
    fill_events = runtime_dir / "fill_receiver_events.jsonl"
    fill_rejects = runtime_dir / "fill_receiver_rejects.jsonl"
    proxy_ingest = runtime_dir / "contract_proxy_ingest.jsonl"
    proxy_dispatch = runtime_dir / "contract_proxy_dispatch.jsonl"
    proxy_rejects = runtime_dir / "contract_proxy_rejects.jsonl"

    # Ensure each replay run starts from a clean artifact set so counts/thresholds are comparable.
    for p in (
        handoff_path,
        state_path,
        ack_log_path,
        fill_raw,
        fill_events,
        fill_rejects,
        proxy_ingest,
        proxy_dispatch,
        proxy_rejects,
    ):
        if p.exists():
            p.unlink()

    fill_port = _pick_free_port()
    proxy_port = _pick_free_port()

    fill_receiver = FreqtradeFillEventReceiver(
        FillEventReceiverConfig(
            raw_log_path=str(fill_raw),
            fill_event_log_path=str(fill_events),
            reject_log_path=str(fill_rejects),
        )
    )
    contract_proxy = FreqtradeContractReceiverProxy(
        ContractProxyConfig(
            ingest_log_path=str(proxy_ingest),
            dispatch_log_path=str(proxy_dispatch),
            reject_log_path=str(proxy_rejects),
            downstream_webhook_url=f"http://127.0.0.1:{fill_port}/trade-update",
            downstream_payload_mode="passthrough",
        )
    )
    fill_server = make_fill_receiver_server("127.0.0.1", fill_port, fill_receiver)
    proxy_server = make_contract_proxy_server("127.0.0.1", proxy_port, contract_proxy)

    threads: List[threading.Thread] = []
    for server in (fill_server, proxy_server):
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        t.start()
        threads.append(t)

    bridge_summaries: List[Dict[str, Any]] = []
    emitted_signal_ids: List[str] = []
    time.sleep(0.15)
    try:
        for b in range(int(batches)):
            rows = _sample_batch(b, int(batch_size))
            emitted_signal_ids.extend([str(r["signal_id"]) for r in rows])
            with open(handoff_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            summary = process_handoff_batch(
                handoff_path=handoff_path,
                state_path=state_path,
                ack_log_path=ack_log_path,
                webhook_url=f"http://127.0.0.1:{proxy_port}/signal",
                receiver_profile="production_v1",
                max_records=int(bridge_max_records or 0),
            )
            bridge_summaries.append(summary)
    finally:
        proxy_server.shutdown()
        fill_server.shutdown()
        proxy_server.server_close()
        fill_server.server_close()
        for t in threads:
            t.join(timeout=1.0)

    ack_rows = _read_jsonl(ack_log_path)
    proxy_dispatch_rows = _read_jsonl(proxy_dispatch)
    fill_raw_rows = _read_jsonl(fill_raw)
    fill_event_rows = _read_jsonl(fill_events)
    fill_reject_rows = _read_jsonl(fill_rejects)
    proxy_reject_rows = _read_jsonl(proxy_rejects)

    ack_signal_ids = [str(r.get("signal_id", "")) for r in ack_rows if isinstance(r, dict)]
    proxy_signal_ids = [str(r.get("signal_id", "")) for r in proxy_dispatch_rows if isinstance(r, dict)]
    fill_signal_ids = [str(r.get("signal_id", "")) for r in fill_event_rows if isinstance(r, dict)]
    emitted_set = set(emitted_signal_ids)

    return {
        "schema": "moneyfan.freqtrade.contract_path_replay.v1",
        "exchange_target": str(exchange_target or ""),
        "data_source": str(data_source or ""),
        "params": {"batches": int(batches), "batch_size": int(batch_size), "bridge_max_records": int(bridge_max_records or 0)},
        "bridge_passes": bridge_summaries,
        "counts": {
            "emitted_handoffs": len(emitted_signal_ids),
            "ack_rows": len(ack_rows),
            "proxy_dispatch_rows": len(proxy_dispatch_rows),
            "fill_raw_rows": len(fill_raw_rows),
            "fill_event_rows": len(fill_event_rows),
            "fill_reject_rows": len(fill_reject_rows),
            "proxy_reject_rows": len(proxy_reject_rows),
        },
        "checks": {
            "all_acks_forwarded": all(r.get("status") == "webhook_forwarded" for r in ack_rows if isinstance(r, dict)),
            "all_ack_signal_ids_seen": emitted_set.issubset(set(ack_signal_ids)),
            "all_proxy_signal_ids_seen": emitted_set.issubset(set(proxy_signal_ids)),
            "all_fill_signal_ids_seen": emitted_set.issubset(set(fill_signal_ids)),
            "fill_receiver_reject_count": len(fill_reject_rows),
            "proxy_reject_count": len(proxy_reject_rows),
        },
        "artifacts": {
            "runtime_dir": str(runtime_dir),
            "handoff_path": str(handoff_path),
            "bridge_state_path": str(state_path),
            "ack_log_path": str(ack_log_path),
            "proxy_dispatch_log_path": str(proxy_dispatch),
            "fill_events_log_path": str(fill_events),
        },
        "sample_signal_ids": emitted_signal_ids[: min(10, len(emitted_signal_ids))],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Replay multiple sample handoff batches through bridge -> contract proxy -> fill receiver")
    p.add_argument("--runtime-dir", type=str, default="", help="Optional artifact directory (default temp)")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bridge-max-records", type=int, default=0)
    p.add_argument("--exchange-target", type=str, default="coinbase_advanced")
    p.add_argument("--data-source", type=str, default="binance")
    p.add_argument("--print-json", action="store_true")
    args = p.parse_args()

    if str(args.runtime_dir or "").strip():
        summary = run_contract_path_replay(
            Path(args.runtime_dir),
            batches=int(args.batches),
            batch_size=int(args.batch_size),
            bridge_max_records=int(args.bridge_max_records or 0),
            exchange_target=str(args.exchange_target or ""),
            data_source=str(args.data_source or ""),
        )
    else:
        with TemporaryDirectory() as d:
            summary = run_contract_path_replay(
                Path(d),
                batches=int(args.batches),
                batch_size=int(args.batch_size),
                bridge_max_records=int(args.bridge_max_records or 0),
                exchange_target=str(args.exchange_target or ""),
                data_source=str(args.data_source or ""),
            )
    if bool(args.print_json):
        print(json.dumps(summary, indent=2))
    else:
        c = summary["counts"]
        k = summary["checks"]
        print(
            "✅ Contract path replay "
            f"emitted={c['emitted_handoffs']} ack={c['ack_rows']} proxy={c['proxy_dispatch_rows']} fill_events={c['fill_event_rows']} "
            f"acks_forwarded={k['all_acks_forwarded']} rejects(fill/proxy)={k['fill_receiver_reject_count']}/{k['proxy_reject_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
