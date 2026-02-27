#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from execution.freqtrade_fill_event_normalizer import canonicalize_fill_event


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(row)) + "\n")


@dataclass
class FillEventReceiverConfig:
    raw_log_path: str = "runtime/freqtrade_trade_updates_raw.jsonl"
    fill_event_log_path: str = "runtime/freqtrade_fill_events.jsonl"
    reject_log_path: str = "runtime/freqtrade_fill_event_rejects.jsonl"
    include_raw_in_canonical: bool = False
    print_events: bool = False


class FreqtradeFillEventReceiver:
    def __init__(self, config: FillEventReceiverConfig):
        self.config = config
        self._lock = threading.Lock()
        self._stats = {
            "requests_total": 0,
            "json_parse_errors": 0,
            "payload_type_errors": 0,
            "canonical_events_written": 0,
            "rejects_written": 0,
            "last_signal_id": None,
            "last_error": None,
            "last_event_ts_utc": None,
        }

    def stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema": "moneyfan.freqtrade.fill_event_receiver_stats.v1",
                "ts_utc": utc_now_iso(),
                "stats": dict(self._stats),
                "paths": {
                    "raw_log_path": str(self.config.raw_log_path),
                    "fill_event_log_path": str(self.config.fill_event_log_path),
                    "reject_log_path": str(self.config.reject_log_path),
                },
            }

    def _append_raw_ingest(self, payload: Dict[str, Any], source_path: str, client_ip: Optional[str]) -> None:
        raw_row = {
            "schema": "moneyfan.freqtrade.trade_update_ingest.v1",
            "received_at_utc": utc_now_iso(),
            "source_path": str(source_path),
            "client_ip": client_ip,
            "payload": payload,
        }
        append_jsonl(Path(self.config.raw_log_path), raw_row)

    def _append_reject(self, reason: str, error: str, source_path: str, client_ip: Optional[str], raw_body: Optional[str] = None, payload: Optional[Any] = None) -> None:
        reject = {
            "schema": "moneyfan.freqtrade.fill_event_receiver_reject.v1",
            "ts_utc": utc_now_iso(),
            "reason": str(reason),
            "error": str(error),
            "source_path": str(source_path),
            "client_ip": client_ip,
        }
        if raw_body is not None:
            reject["raw_body"] = str(raw_body)[:4000]
        if payload is not None:
            reject["payload"] = payload
        append_jsonl(Path(self.config.reject_log_path), reject)
        if bool(self.config.print_events):
            print(f"⚠️  fill-event receiver reject: {reason} error={error}")

    def process_trade_update_payload(
        self,
        payload: Dict[str, Any],
        source_path: str = "/trade-update",
        client_ip: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        with self._lock:
            self._stats["requests_total"] = int(self._stats["requests_total"]) + 1

        if not isinstance(payload, dict):
            with self._lock:
                self._stats["payload_type_errors"] = int(self._stats["payload_type_errors"]) + 1
                self._stats["last_error"] = "payload must be a JSON object"
            self._append_reject(
                reason="payload_type_error",
                error="payload must be a JSON object",
                source_path=source_path,
                client_ip=client_ip,
                payload=payload,
            )
            return 400, {
                "ok": False,
                "error": "payload must be a JSON object",
            }

        self._append_raw_ingest(payload, source_path=source_path, client_ip=client_ip)

        try:
            event = canonicalize_fill_event(payload, include_raw=bool(self.config.include_raw_in_canonical))
        except Exception as e:
            err = str(e)
            with self._lock:
                self._stats["rejects_written"] = int(self._stats["rejects_written"]) + 1
                self._stats["last_error"] = err
                self._stats["last_event_ts_utc"] = utc_now_iso()
            self._append_reject(
                reason="canonicalize_error",
                error=err,
                source_path=source_path,
                client_ip=client_ip,
                payload=payload,
            )
            return 400, {
                "ok": False,
                "error": err,
            }

        # Add ingest metadata without changing the canonical top-level contract.
        event["receiver_ingest"] = {
            "received_at_utc": utc_now_iso(),
            "source_path": str(source_path),
            "client_ip": client_ip,
        }
        append_jsonl(Path(self.config.fill_event_log_path), event)

        with self._lock:
            self._stats["canonical_events_written"] = int(self._stats["canonical_events_written"]) + 1
            self._stats["last_signal_id"] = event.get("signal_id")
            self._stats["last_error"] = None
            self._stats["last_event_ts_utc"] = utc_now_iso()

        if bool(self.config.print_events):
            print(
                "✅ fill-event receiver "
                f"signal_id={event.get('signal_id')} pair={event.get('pair')} side={event.get('side')} status={event.get('status')}"
            )

        return 200, {
            "ok": True,
            "signal_id": event.get("signal_id"),
            "pair": event.get("pair"),
            "side": event.get("side"),
            "status": event.get("status"),
            "schema": event.get("schema"),
        }

    def process_http_json_body(
        self,
        body_bytes: bytes,
        source_path: str,
        client_ip: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            decoded = body_bytes.decode("utf-8")
        except Exception as e:
            with self._lock:
                self._stats["json_parse_errors"] = int(self._stats["json_parse_errors"]) + 1
                self._stats["last_error"] = f"utf8 decode error: {e}"
            self._append_reject(
                reason="utf8_decode_error",
                error=str(e),
                source_path=source_path,
                client_ip=client_ip,
                raw_body=repr(body_bytes[:4000]),
            )
            return 400, {"ok": False, "error": "invalid utf-8 body"}

        try:
            payload = json.loads(decoded)
        except Exception as e:
            with self._lock:
                self._stats["json_parse_errors"] = int(self._stats["json_parse_errors"]) + 1
                self._stats["last_error"] = f"json parse error: {e}"
            self._append_reject(
                reason="json_parse_error",
                error=str(e),
                source_path=source_path,
                client_ip=client_ip,
                raw_body=decoded,
            )
            return 400, {"ok": False, "error": "invalid JSON body"}

        return self.process_trade_update_payload(payload, source_path=source_path, client_ip=client_ip)


class _ReceiverHandler(BaseHTTPRequestHandler):
    server_version = "MoneyfanFreqtradeFillEventReceiver/1.0"

    @property
    def receiver(self) -> FreqtradeFillEventReceiver:
        return self.server.receiver  # type: ignore[attr-defined]

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
        # Keep logs concise and operator-oriented.
        print(f"[fill-receiver] {self.address_string()} - " + (fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in {"/healthz", "/"}:
            self._write_json(200, {"ok": True, "service": "freqtrade_fill_event_receiver", "ts_utc": utc_now_iso()})
            return
        if path == "/stats":
            self._write_json(200, self.receiver.stats_snapshot())
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path not in {"/trade-update", "/fill", "/fill-event", "/ingest"}:
            self._write_json(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except Exception:
            length = 0
        if length <= 0:
            self._write_json(400, {"ok": False, "error": "empty body"})
            return

        body = self.rfile.read(length)
        status, payload = self.receiver.process_http_json_body(
            body_bytes=body,
            source_path=path,
            client_ip=self._client_ip(),
        )
        self._write_json(status, payload)


def make_server(host: str, port: int, receiver: FreqtradeFillEventReceiver) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), _ReceiverHandler)
    server.receiver = receiver  # type: ignore[attr-defined]
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Receive Freqtrade trade updates and write canonical moneyfan fill-event JSONL")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=8091, help="Bind port")
    p.add_argument("--raw-log-path", type=str, default="runtime/freqtrade_trade_updates_raw.jsonl",
                   help="Append-only raw incoming trade update JSONL")
    p.add_argument("--fill-event-log-path", type=str, default="runtime/freqtrade_fill_events.jsonl",
                   help="Canonical fill-event JSONL for reconciliation")
    p.add_argument("--reject-log-path", type=str, default="runtime/freqtrade_fill_event_rejects.jsonl",
                   help="Receiver reject/error JSONL")
    p.add_argument("--include-raw-in-canonical", action="store_true",
                   help="Embed full raw payload in canonical fill-event rows")
    p.add_argument("--print-events", action="store_true",
                   help="Print accepted/rejected event summaries")
    return p


def run_cli(args: argparse.Namespace) -> int:
    receiver = FreqtradeFillEventReceiver(
        FillEventReceiverConfig(
            raw_log_path=args.raw_log_path,
            fill_event_log_path=args.fill_event_log_path,
            reject_log_path=args.reject_log_path,
            include_raw_in_canonical=bool(args.include_raw_in_canonical),
            print_events=bool(args.print_events),
        )
    )
    server = make_server(str(args.host), int(args.port), receiver)
    print(
        "🚚 Freqtrade fill-event receiver listening "
        f"http://{args.host}:{int(args.port)} "
        f"(POST /trade-update, /fill, /fill-event, /ingest | GET /healthz, /stats)"
    )
    print(
        "📝 Logs "
        f"raw={args.raw_log_path} canonical={args.fill_event_log_path} rejects={args.reject_log_path}"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n🛑 Receiver stopped")
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
