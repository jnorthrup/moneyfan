#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


def load_bridge_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": "moneyfan.freqtrade.bridge_state.v1",
            "offset": 0,
            "updated_at": None,
        }
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return {
            "schema": "moneyfan.freqtrade.bridge_state.v1",
            "offset": 0,
            "updated_at": None,
        }
    if not isinstance(payload, dict):
        return {
            "schema": "moneyfan.freqtrade.bridge_state.v1",
            "offset": 0,
            "updated_at": None,
        }
    payload.setdefault("schema", "moneyfan.freqtrade.bridge_state.v1")
    payload["offset"] = int(payload.get("offset", 0) or 0)
    payload.setdefault("updated_at", None)
    return payload


def save_bridge_state(path: Path, state: Dict[str, Any]) -> None:
    payload = dict(state)
    payload["schema"] = "moneyfan.freqtrade.bridge_state.v1"
    payload["offset"] = int(payload.get("offset", 0) or 0)
    payload["updated_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    tmp_path.replace(path)


def read_jsonl_batch_from_offset(
    path: Path,
    offset: int,
    max_records: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        return [], int(offset or 0)

    file_size = path.stat().st_size
    start_offset = int(offset or 0)
    if file_size < start_offset:
        # File rotated/truncated; restart from the beginning.
        start_offset = 0

    rows: List[Dict[str, Any]] = []
    next_offset = start_offset
    max_n = int(max_records or 0)

    with open(path, "rb") as f:
        f.seek(start_offset)
        while True:
            offset_before = int(f.tell())
            line = f.readline()
            if not line:
                break
            offset_after = int(f.tell())
            next_offset = offset_after
            if not line.strip():
                continue

            record: Dict[str, Any]
            try:
                parsed = json.loads(line.decode("utf-8"))
                if isinstance(parsed, dict):
                    record = parsed
                else:
                    record = {"value": parsed}
            except Exception as e:
                record = {
                    "_parse_error": str(e),
                    "_raw_line": line.decode("utf-8", errors="replace").rstrip("\n"),
                }
            record["_source_offset_start"] = offset_before
            record["_source_offset_end"] = offset_after
            rows.append(record)

            if max_n > 0 and len(rows) >= max_n:
                break

    return rows, next_offset


def handoff_to_freqtrade_webhook_payload(handoff: Dict[str, Any]) -> Dict[str, Any]:
    schema = str(handoff.get("schema", "") or "")
    if schema != "moneyfan.freqtrade.handoff.v1":
        raise ValueError(f"Unsupported handoff schema: {schema or '<missing>'}")

    pair = str(handoff.get("pair", "") or "")
    side = str(handoff.get("side", "") or "").lower()
    if side not in {"long", "short"}:
        raise ValueError(f"Unsupported side: {side or '<missing>'}")
    signal_id = str(handoff.get("signal_id", "") or "").strip()
    if not signal_id:
        raise ValueError("Missing signal_id (required for HRM fidelity reconciliation)")
    action = "enter_long" if side == "long" else "enter_short"

    model = handoff.get("model") if isinstance(handoff.get("model"), dict) else {}
    risk = handoff.get("risk") if isinstance(handoff.get("risk"), dict) else {}
    dispatch = handoff.get("dispatch") if isinstance(handoff.get("dispatch"), dict) else {}

    # This is a bridge payload shaped for a generic Freqtrade webhook/custom receiver.
    # It preserves signal_id and HRM fidelity metadata so downstream execution/fill logs
    # can be reconciled to model predictions.
    return {
        "schema": "moneyfan.freqtrade.bridge.webhook.v1",
        "ts_utc": utc_now_iso(),
        "signal_id": signal_id,
        "pair": pair,
        "side": side,
        "action": action,
        "enter_long": 1 if side == "long" else 0,
        "enter_short": 1 if side == "short" else 0,
        "stake_fraction": float(handoff.get("stake_fraction", 0.0) or 0.0),
        "stoploss": float(handoff.get("stoploss", 0.0) or 0.0),
        "take_profit_pct": float(handoff.get("take_profit_pct", 0.0) or 0.0),
        "metadata": {
            "source_schema": schema,
            "source_dispatch": {
                "iteration": dispatch.get("iteration"),
                "source_mode": dispatch.get("source_mode"),
                "source_broker_label": dispatch.get("source_broker_label"),
            },
            "hrm": {
                "confidence": model.get("confidence"),
                "pred_fwd_return": model.get("pred_fwd_return"),
                "score": model.get("score"),
                "score_mode": model.get("score_mode"),
                "passes_edge_gate": model.get("passes_edge_gate"),
                "net_effective_predicted_edge_bps": model.get("net_effective_predicted_edge_bps"),
                "trade_head_calibration_loaded": model.get("trade_head_calibration_loaded"),
                "risk_tier": risk.get("risk_tier"),
                "raw_vetoed": model.get("raw_vetoed"),
                "raw_veto_reason": model.get("raw_veto_reason"),
                "veto_overridden": model.get("veto_overridden"),
            },
        },
    }


def validate_bridge_webhook_payload_v1(payload: Dict[str, Any]) -> None:
    required_top = (
        "schema", "ts_utc", "signal_id", "pair", "side", "action",
        "enter_long", "enter_short", "stake_fraction", "stoploss",
        "take_profit_pct", "metadata",
    )
    for key in required_top:
        if key not in payload:
            raise ValueError(f"Contract v1 payload missing required field: {key}")
    if str(payload.get("schema", "") or "") != "moneyfan.freqtrade.bridge.webhook.v1":
        raise ValueError("Contract v1 payload schema mismatch")
    if str(payload.get("side", "") or "").lower() not in {"long", "short"}:
        raise ValueError("Contract v1 payload side must be long|short")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Contract v1 payload metadata must be object")
    if "source_schema" not in metadata:
        raise ValueError("Contract v1 payload metadata.source_schema missing")
    hrm = metadata.get("hrm")
    if not isinstance(hrm, dict):
        raise ValueError("Contract v1 payload metadata.hrm must be object")
    for key in ("confidence", "pred_fwd_return", "net_effective_predicted_edge_bps"):
        if key not in hrm:
            raise ValueError(f"Contract v1 payload metadata.hrm.{key} missing")


def parse_json_object_or_none(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_receiver_response_contract_v1(response_body_text: str, expected_signal_id: str) -> Dict[str, Any]:
    body = parse_json_object_or_none(response_body_text)
    if body is None:
        raise ValueError("Contract v1 receiver response must be a JSON object body")
    if bool(body.get("ok")) is not True:
        raise ValueError("Contract v1 receiver response requires ok=true")
    if bool(body.get("accepted")) is not True:
        raise ValueError("Contract v1 receiver response requires accepted=true")
    got_signal_id = str(body.get("signal_id", "") or "").strip()
    if not got_signal_id:
        raise ValueError("Contract v1 receiver response missing signal_id")
    if got_signal_id != str(expected_signal_id or ""):
        raise ValueError("Contract v1 receiver response signal_id mismatch")
    return body


def post_webhook_json(url: str, payload: Dict[str, Any], timeout_seconds: float = 5.0) -> Dict[str, Any]:
    body = json.dumps(_json_safe(payload)).encode("utf-8")
    req = Request(
        url=str(url),
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "moneyfan-freqtrade-handoff-bridge/1",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=float(timeout_seconds)) as resp:
            resp_body = resp.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": True,
                "http_status": int(getattr(resp, "status", 200)),
                "response_body": resp_body,
            }
    except HTTPError as e:
        body_text = e.read(4096).decode("utf-8", errors="replace")
        return {
            "ok": False,
            "http_status": int(e.code),
            "response_body": body_text,
            "error": f"HTTPError: {e}",
        }
    except URLError as e:
        return {
            "ok": False,
            "http_status": None,
            "response_body": "",
            "error": f"URLError: {e}",
        }


def process_handoff_batch(
    handoff_path: Path,
    state_path: Path,
    ack_log_path: Path,
    webhook_url: Optional[str] = None,
    timeout_seconds: float = 5.0,
    max_records: int = 0,
    print_payloads: bool = False,
    receiver_profile: str = "generic",
) -> Dict[str, Any]:
    state = load_bridge_state(state_path)
    start_offset = int(state.get("offset", 0) or 0)
    rows, next_offset = read_jsonl_batch_from_offset(handoff_path, start_offset, max_records=max_records)

    processed = 0
    forwarded = 0
    failed = 0
    skipped = 0
    dry_run = not bool((webhook_url or "").strip())
    profile = str(receiver_profile or "generic").strip().lower()

    for row in rows:
        processed += 1
        signal_id = row.get("signal_id")
        ack: Dict[str, Any] = {
            "schema": "moneyfan.freqtrade.dispatch_ack.v1",
            "ts_utc": utc_now_iso(),
            "mode": "dry_run" if dry_run else "webhook",
            "handoff_path": str(handoff_path),
            "signal_id": signal_id,
            "handoff_schema": row.get("schema"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "handoff_offset_start": row.get("_source_offset_start"),
            "handoff_offset_end": row.get("_source_offset_end"),
        }

        if row.get("_parse_error"):
            ack.update(
                {
                    "status": "invalid_jsonl_record",
                    "error": row.get("_parse_error"),
                    "raw_line": row.get("_raw_line", "")[:500],
                }
            )
            failed += 1
            append_jsonl(ack_log_path, ack)
            continue

        try:
            payload = handoff_to_freqtrade_webhook_payload(row)
            if profile == "production_v1":
                validate_bridge_webhook_payload_v1(payload)
        except Exception as e:
            ack.update(
                {
                    "status": "invalid_handoff_record",
                    "error": str(e),
                }
            )
            failed += 1
            append_jsonl(ack_log_path, ack)
            continue

        ack["bridge_signal_id"] = payload.get("signal_id")
        ack["action"] = payload.get("action")
        ack["stake_fraction"] = payload.get("stake_fraction")

        if print_payloads:
            print(json.dumps(payload, indent=2))

        if dry_run:
            ack["status"] = "dry_run_forwarded"
            forwarded += 1
            append_jsonl(ack_log_path, ack)
            continue

        resp = post_webhook_json(str(webhook_url), payload, timeout_seconds=timeout_seconds)
        ack["http_status"] = resp.get("http_status")
        if resp.get("ok"):
            if profile == "production_v1":
                try:
                    parsed_receiver_resp = validate_receiver_response_contract_v1(
                        str(resp.get("response_body", "") or ""),
                        expected_signal_id=str(payload.get("signal_id", "") or ""),
                    )
                    ack["status"] = "webhook_forwarded"
                    ack["receiver_accepted"] = bool(parsed_receiver_resp.get("accepted"))
                    if parsed_receiver_resp.get("receiver_schema") is not None:
                        ack["receiver_schema"] = parsed_receiver_resp.get("receiver_schema")
                    if parsed_receiver_resp.get("freqtrade_request_id") is not None:
                        ack["freqtrade_request_id"] = parsed_receiver_resp.get("freqtrade_request_id")
                    forwarded += 1
                except Exception as e:
                    ack["status"] = "webhook_contract_invalid_response"
                    ack["error"] = str(e)
                    failed += 1
            else:
                ack["status"] = "webhook_forwarded"
                forwarded += 1
        else:
            ack["status"] = "webhook_forward_failed"
            ack["error"] = resp.get("error")
            failed += 1
        body_snippet = str(resp.get("response_body", "") or "")
        if body_snippet:
            ack["response_body"] = body_snippet[:500]
        append_jsonl(ack_log_path, ack)

    if not rows:
        skipped = 0
    save_bridge_state(
        state_path,
        {
            "offset": int(next_offset),
            "handoff_path": str(handoff_path),
            "ack_log_path": str(ack_log_path),
            "last_batch_processed": int(processed),
            "last_batch_forwarded": int(forwarded),
            "last_batch_failed": int(failed),
            "last_batch_skipped": int(skipped),
        },
    )

    return {
        "processed": int(processed),
        "forwarded": int(forwarded),
        "failed": int(failed),
        "skipped": int(skipped),
        "dry_run": bool(dry_run),
        "receiver_profile": profile,
        "start_offset": int(start_offset),
        "next_offset": int(next_offset),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bridge moneyfan Freqtrade handoff JSONL to a webhook/custom receiver")
    p.add_argument("--handoff-path", type=str, default="runtime/freqtrade_handoff.jsonl",
                   help="Input JSONL handoff file produced by run.py offload mode")
    p.add_argument("--state-path", type=str, default="runtime/freqtrade_handoff_bridge_state.json",
                   help="Bridge state JSON with byte offset tracking")
    p.add_argument("--ack-log-path", type=str, default="runtime/freqtrade_dispatch_ack.jsonl",
                   help="Output JSONL acknowledgments for forwarded/failed handoff records")
    p.add_argument("--webhook-url", type=str, default="",
                   help="Webhook endpoint to POST bridged payloads to (empty = dry-run)")
    p.add_argument("--timeout-seconds", type=float, default=5.0,
                   help="HTTP timeout for webhook POST")
    p.add_argument("--max-records", type=int, default=0,
                   help="Max records per pass (0 = all available)")
    p.add_argument("--follow", action="store_true",
                   help="Poll for new records until interrupted")
    p.add_argument("--poll-seconds", type=float, default=2.0,
                   help="Polling interval in follow mode")
    p.add_argument("--print-payloads", action="store_true",
                   help="Print bridged payload JSON before dispatch (debugging)")
    p.add_argument("--receiver-profile", type=str, default="generic",
                   choices=["generic", "production_v1"],
                   help="Receiver contract validation profile (generic or documented production_v1)")
    return p


def run_loop(args: argparse.Namespace) -> int:
    handoff_path = Path(args.handoff_path)
    state_path = Path(args.state_path)
    ack_log_path = Path(args.ack_log_path)
    webhook_url = str(args.webhook_url or "").strip() or None
    poll_seconds = max(0.1, float(args.poll_seconds))

    print(
        "🔁 Freqtrade handoff bridge "
        f"mode={'webhook' if webhook_url else 'dry-run'} "
        f"handoff={handoff_path} ack_log={ack_log_path} state={state_path}"
    )
    if webhook_url:
        print(f"🌐 Webhook target: {webhook_url}")
    else:
        print("🧪 Dry-run mode: no webhook URL configured")
    print(f"📦 Receiver profile: {str(args.receiver_profile)}")

    try:
        while True:
            summary = process_handoff_batch(
                handoff_path=handoff_path,
                state_path=state_path,
                ack_log_path=ack_log_path,
                webhook_url=webhook_url,
                timeout_seconds=float(args.timeout_seconds),
                max_records=int(args.max_records or 0),
                print_payloads=bool(args.print_payloads),
                receiver_profile=str(args.receiver_profile),
            )
            if summary["processed"] > 0:
                print(
                    "✅ Bridge pass "
                    f"processed={summary['processed']} forwarded={summary['forwarded']} failed={summary['failed']} "
                    f"offset={summary['start_offset']}->{summary['next_offset']}"
                )
            elif not bool(args.follow):
                print("ℹ️  No new handoff records")

            if not bool(args.follow):
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n🛑 Bridge stopped")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
