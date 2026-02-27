#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(row)) + "\n")


def post_json(url: str, payload: Dict[str, Any], timeout_seconds: float = 5.0) -> Dict[str, Any]:
    body = json.dumps(_json_safe(payload)).encode("utf-8")
    req = Request(
        url=str(url),
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "moneyfan-freqtrade-contract-proxy/1"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=float(timeout_seconds)) as resp:
            txt = resp.read(4096).decode("utf-8", errors="replace")
            return {"ok": True, "http_status": int(getattr(resp, "status", 200)), "response_body": txt}
    except HTTPError as e:
        txt = e.read(4096).decode("utf-8", errors="replace")
        return {"ok": False, "http_status": int(e.code), "response_body": txt, "error": f"HTTPError: {e}"}
    except URLError as e:
        return {"ok": False, "http_status": None, "response_body": "", "error": f"URLError: {e}"}


def validate_bridge_payload_contract_v1(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if str(payload.get("schema", "") or "") != "moneyfan.freqtrade.bridge.webhook.v1":
        raise ValueError("unsupported schema (expected moneyfan.freqtrade.bridge.webhook.v1)")
    signal_id = str(payload.get("signal_id", "") or "").strip()
    if not signal_id:
        raise ValueError("missing signal_id")
    pair = str(payload.get("pair", "") or "").strip()
    side = str(payload.get("side", "") or "").strip().lower()
    if not pair:
        raise ValueError("missing pair")
    if side not in {"long", "short"}:
        raise ValueError("side must be long|short")
    return signal_id, pair, side


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def build_downstream_payload(
    bridge_payload: Dict[str, Any],
    mode: str,
    freqtrade_request_id: str,
) -> Dict[str, Any]:
    payload_mode = str(mode or "passthrough").strip().lower()
    if payload_mode == "passthrough":
        return dict(bridge_payload)
    if payload_mode != "freqtrade_webhook_v1":
        raise ValueError(f"unsupported downstream payload mode: {mode}")

    signal_id = str(bridge_payload.get("signal_id", "") or "")
    pair = str(bridge_payload.get("pair", "") or "")
    side = str(bridge_payload.get("side", "") or "").lower()
    metadata = bridge_payload.get("metadata") if isinstance(bridge_payload.get("metadata"), dict) else {}
    hrm = metadata.get("hrm") if isinstance(metadata.get("hrm"), dict) else {}
    source_dispatch = metadata.get("source_dispatch") if isinstance(metadata.get("source_dispatch"), dict) else {}

    return {
        "schema": "moneyfan.freqtrade.proxy.forward.freqtrade_webhook_v1",
        "ts_utc": utc_now_iso(),
        "pair": pair,
        "action": "buy" if side == "long" else "sell",
        "side": side,
        "enter_long": 1 if side == "long" else 0,
        "enter_short": 1 if side == "short" else 0,
        "stake_fraction": _as_float(bridge_payload.get("stake_fraction"), 0.0),
        "stoploss": _as_float(bridge_payload.get("stoploss"), 0.0),
        "take_profit_pct": _as_float(bridge_payload.get("take_profit_pct"), 0.0),
        "metadata": {
            "signal_id": signal_id,
            "freqtrade_request_id": str(freqtrade_request_id),
            "source_schema": str(bridge_payload.get("schema", "") or ""),
            "source_dispatch": {
                "iteration": source_dispatch.get("iteration"),
                "source_mode": source_dispatch.get("source_mode"),
                "source_broker_label": source_dispatch.get("source_broker_label"),
            },
            "hrm": {
                "confidence": hrm.get("confidence"),
                "pred_fwd_return": hrm.get("pred_fwd_return"),
                "net_effective_predicted_edge_bps": hrm.get("net_effective_predicted_edge_bps"),
            },
        },
    }


@dataclass
class ContractProxyConfig:
    ingest_log_path: str = "runtime/freqtrade_contract_receiver_ingest.jsonl"
    dispatch_log_path: str = "runtime/freqtrade_contract_receiver_dispatch.jsonl"
    reject_log_path: str = "runtime/freqtrade_contract_receiver_rejects.jsonl"
    downstream_webhook_url: str = ""
    downstream_payload_mode: str = "passthrough"
    timeout_seconds: float = 5.0
    print_events: bool = False


class FreqtradeContractReceiverProxy:
    def __init__(self, config: ContractProxyConfig):
        self.config = config
        self._lock = threading.Lock()
        self._stats = {
            "requests_total": 0,
            "accepted_total": 0,
            "rejected_total": 0,
            "downstream_forwarded": 0,
            "downstream_failed": 0,
            "last_signal_id": None,
            "last_error": None,
            "last_event_ts_utc": None,
        }

    def stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema": "moneyfan.freqtrade.contract_receiver_proxy_stats.v1",
                "ts_utc": utc_now_iso(),
                "stats": dict(self._stats),
                "paths": {
                    "ingest_log_path": str(self.config.ingest_log_path),
                    "dispatch_log_path": str(self.config.dispatch_log_path),
                    "reject_log_path": str(self.config.reject_log_path),
                },
                "downstream_webhook_url": str(self.config.downstream_webhook_url or ""),
                "downstream_payload_mode": str(self.config.downstream_payload_mode or "passthrough"),
            }

    def _append_ingest(self, payload: Dict[str, Any], source_path: str, client_ip: Optional[str]) -> None:
        append_jsonl(
            Path(self.config.ingest_log_path),
            {
                "schema": "moneyfan.freqtrade.contract_receiver_ingest.v1",
                "received_at_utc": utc_now_iso(),
                "source_path": str(source_path),
                "client_ip": client_ip,
                "payload": payload,
            },
        )

    def _append_reject(self, reason: str, error: str, source_path: str, client_ip: Optional[str], payload: Optional[Any] = None) -> None:
        append_jsonl(
            Path(self.config.reject_log_path),
            {
                "schema": "moneyfan.freqtrade.contract_receiver_reject.v1",
                "ts_utc": utc_now_iso(),
                "reason": str(reason),
                "error": str(error),
                "source_path": str(source_path),
                "client_ip": client_ip,
                "payload": payload,
            },
        )

    def process_bridge_payload(
        self,
        payload: Dict[str, Any],
        source_path: str = "/signal",
        client_ip: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        with self._lock:
            self._stats["requests_total"] = int(self._stats["requests_total"]) + 1

        if not isinstance(payload, dict):
            with self._lock:
                self._stats["rejected_total"] = int(self._stats["rejected_total"]) + 1
                self._stats["last_error"] = "payload must be a JSON object"
            self._append_reject("payload_type_error", "payload must be a JSON object", source_path, client_ip, payload=payload)
            return 400, {"ok": False, "accepted": False, "error": "payload must be a JSON object"}

        self._append_ingest(payload, source_path, client_ip)

        try:
            signal_id, pair, side = validate_bridge_payload_contract_v1(payload)
        except Exception as e:
            err = str(e)
            with self._lock:
                self._stats["rejected_total"] = int(self._stats["rejected_total"]) + 1
                self._stats["last_error"] = err
                self._stats["last_event_ts_utc"] = utc_now_iso()
            self._append_reject("contract_validation_error", err, source_path, client_ip, payload=payload)
            return 400, {"ok": False, "accepted": False, "error": err}

        freqtrade_request_id = f"mfproxy-{uuid.uuid4().hex[:12]}"
        dispatch_row: Dict[str, Any] = {
            "schema": "moneyfan.freqtrade.contract_receiver_dispatch.v1",
            "ts_utc": utc_now_iso(),
            "signal_id": signal_id,
            "pair": pair,
            "side": side,
            "source_path": str(source_path),
            "client_ip": client_ip,
            "mode": "downstream_webhook" if str(self.config.downstream_webhook_url or "").strip() else "dry_run_accept",
            "freqtrade_request_id": freqtrade_request_id,
        }

        downstream_url = str(self.config.downstream_webhook_url or "").strip()
        if downstream_url:
            try:
                downstream_payload = build_downstream_payload(
                    bridge_payload=payload,
                    mode=str(self.config.downstream_payload_mode or "passthrough"),
                    freqtrade_request_id=freqtrade_request_id,
                )
            except Exception as e:
                dispatch_row["status"] = "downstream_payload_build_failed"
                dispatch_row["error"] = str(e)
                append_jsonl(Path(self.config.dispatch_log_path), dispatch_row)
                with self._lock:
                    self._stats["downstream_failed"] = int(self._stats["downstream_failed"]) + 1
                    self._stats["accepted_total"] = int(self._stats["accepted_total"]) + 1
                    self._stats["last_signal_id"] = signal_id
                    self._stats["last_error"] = str(e)
                    self._stats["last_event_ts_utc"] = utc_now_iso()
                return 200, {
                    "ok": True,
                    "accepted": True,
                    "signal_id": signal_id,
                    "receiver_schema": "moneyfan.freqtrade.receiver.accept.v1",
                    "freqtrade_request_id": freqtrade_request_id,
                }

            dispatch_row["downstream_payload_mode"] = str(self.config.downstream_payload_mode or "passthrough")
            dispatch_row["downstream_payload_schema"] = downstream_payload.get("schema")
            resp = post_json(downstream_url, downstream_payload, timeout_seconds=float(self.config.timeout_seconds))
            dispatch_row["downstream_http_status"] = resp.get("http_status")
            if resp.get("ok"):
                dispatch_row["status"] = "downstream_forwarded"
                with self._lock:
                    self._stats["downstream_forwarded"] = int(self._stats["downstream_forwarded"]) + 1
            else:
                dispatch_row["status"] = "downstream_forward_failed"
                dispatch_row["error"] = resp.get("error")
                with self._lock:
                    self._stats["downstream_failed"] = int(self._stats["downstream_failed"]) + 1
            body = str(resp.get("response_body", "") or "")
            if body:
                dispatch_row["downstream_response_body"] = body[:500]
        else:
            dispatch_row["status"] = "accepted_dry_run"

        append_jsonl(Path(self.config.dispatch_log_path), dispatch_row)
        with self._lock:
            self._stats["accepted_total"] = int(self._stats["accepted_total"]) + 1
            self._stats["last_signal_id"] = signal_id
            self._stats["last_error"] = None
            self._stats["last_event_ts_utc"] = utc_now_iso()

        if bool(self.config.print_events):
            print(f"✅ contract-proxy signal_id={signal_id} pair={pair} side={side} status={dispatch_row['status']}")

        return 200, {
            "ok": True,
            "accepted": True,
            "signal_id": signal_id,
            "receiver_schema": "moneyfan.freqtrade.receiver.accept.v1",
            "freqtrade_request_id": freqtrade_request_id,
        }

    def process_http_json_body(self, body_bytes: bytes, source_path: str, client_ip: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        try:
            decoded = body_bytes.decode("utf-8")
            payload = json.loads(decoded)
        except Exception as e:
            self._append_reject("json_parse_error", str(e), source_path, client_ip, payload=None)
            with self._lock:
                self._stats["rejected_total"] = int(self._stats["rejected_total"]) + 1
                self._stats["last_error"] = f"json parse error: {e}"
            return 400, {"ok": False, "accepted": False, "error": "invalid JSON body"}
        return self.process_bridge_payload(payload, source_path=source_path, client_ip=client_ip)


class _ProxyHandler(BaseHTTPRequestHandler):
    server_version = "MoneyfanFreqtradeContractReceiverProxy/1.0"

    @property
    def proxy(self) -> FreqtradeContractReceiverProxy:
        return self.server.proxy  # type: ignore[attr-defined]

    def _client_ip(self) -> Optional[str]:
        try:
            return str(self.client_address[0]) if self.client_address else None
        except Exception:
            return None

    def _write_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[contract-proxy] {self.address_string()} - " + (fmt % args))

    def do_GET(self) -> None:
        path = (urlparse(self.path).path.rstrip("/") or "/")
        if path in {"/", "/healthz"}:
            self._write_json(200, {"ok": True, "service": "freqtrade_contract_receiver_proxy", "ts_utc": utc_now_iso()})
            return
        if path == "/stats":
            self._write_json(200, self.proxy.stats_snapshot())
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = (urlparse(self.path).path.rstrip("/") or "/")
        if path not in {"/signal", "/trade-update", "/ingest"}:
            self._write_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except Exception:
            length = 0
        if length <= 0:
            self._write_json(400, {"ok": False, "accepted": False, "error": "empty body"})
            return
        body = self.rfile.read(length)
        status, payload = self.proxy.process_http_json_body(body, source_path=path, client_ip=self._client_ip())
        self._write_json(status, payload)


def make_server(host: str, port: int, proxy: FreqtradeContractReceiverProxy) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), _ProxyHandler)
    server.proxy = proxy  # type: ignore[attr-defined]
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contract-compliant receiver/proxy for moneyfan bridge payloads (production profile v1)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--ingest-log-path", type=str, default="runtime/freqtrade_contract_receiver_ingest.jsonl")
    p.add_argument("--dispatch-log-path", type=str, default="runtime/freqtrade_contract_receiver_dispatch.jsonl")
    p.add_argument("--reject-log-path", type=str, default="runtime/freqtrade_contract_receiver_rejects.jsonl")
    p.add_argument("--downstream-webhook-url", type=str, default="", help="Optional downstream Freqtrade/custom webhook target")
    p.add_argument("--downstream-payload-mode", type=str, default="passthrough",
                   choices=["passthrough", "freqtrade_webhook_v1"],
                   help="Payload mapping mode for downstream forwarding")
    p.add_argument("--timeout-seconds", type=float, default=5.0)
    p.add_argument("--print-events", action="store_true")
    return p


def run_cli(args: argparse.Namespace) -> int:
    proxy = FreqtradeContractReceiverProxy(
        ContractProxyConfig(
            ingest_log_path=str(args.ingest_log_path),
            dispatch_log_path=str(args.dispatch_log_path),
            reject_log_path=str(args.reject_log_path),
            downstream_webhook_url=str(args.downstream_webhook_url or ""),
            downstream_payload_mode=str(args.downstream_payload_mode or "passthrough"),
            timeout_seconds=float(args.timeout_seconds),
            print_events=bool(args.print_events),
        )
    )
    server = make_server(str(args.host), int(args.port), proxy)
    print(
        "🧩 Freqtrade contract receiver proxy listening "
        f"http://{args.host}:{int(args.port)} (POST /signal, /trade-update, /ingest | GET /healthz, /stats)"
    )
    print(
        "📝 Logs "
        f"ingest={args.ingest_log_path} dispatch={args.dispatch_log_path} rejects={args.reject_log_path}"
    )
    if str(args.downstream_webhook_url or "").strip():
        print(f"🌐 Downstream webhook: {args.downstream_webhook_url}")
        print(f"🔀 Downstream payload mode: {args.downstream_payload_mode}")
    else:
        print("🧪 Downstream forwarding disabled (accept-only dry-run)")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n🛑 Contract proxy stopped")
    finally:
        server.server_close()
    return 0


def main() -> int:
    return run_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
