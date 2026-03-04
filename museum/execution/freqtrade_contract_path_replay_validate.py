#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.freqtrade_contract_path_replay import run_contract_path_replay


THRESHOLD_PROFILES: Dict[str, Dict[str, Any]] = {
    "default_strict": {
        "min_forward_rate": 1.0,
        "max_fill_rejects": 0,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
    "coinbase_advanced__binance": {
        "min_forward_rate": 1.0,
        "max_fill_rejects": 0,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
    "coinbase_advanced__binance_relaxed": {
        "min_forward_rate": 0.95,
        "max_fill_rejects": 1,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
    "coinbase_advanced__coinbase_advanced": {
        "min_forward_rate": 1.0,
        "max_fill_rejects": 0,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
    "coinbase_advanced__coinbase_advanced_relaxed": {
        "min_forward_rate": 0.98,
        "max_fill_rejects": 1,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
    "freqtrade_paper__binance": {
        "min_forward_rate": 1.0,
        "max_fill_rejects": 0,
        "max_proxy_rejects": 0,
        "require_all_signal_ids_seen": True,
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build_timestamped_path(base_path: Path, stamp: Optional[str] = None) -> Path:
    ts = str(stamp or _utc_stamp())
    suffix = "".join(base_path.suffixes) or ".md"
    stem = base_path.name[: -len(suffix)] if suffix and base_path.name.endswith(suffix) else base_path.stem
    return base_path.with_name(f"{stem}_{ts}{suffix}")


def validate_replay_summary(
    replay_summary: Dict[str, Any],
    min_forward_rate: float = 1.0,
    max_fill_rejects: int = 0,
    max_proxy_rejects: int = 0,
    require_all_signal_ids_seen: bool = True,
) -> Dict[str, Any]:
    counts = replay_summary.get("counts") if isinstance(replay_summary.get("counts"), dict) else {}
    checks = replay_summary.get("checks") if isinstance(replay_summary.get("checks"), dict) else {}
    params = replay_summary.get("params") if isinstance(replay_summary.get("params"), dict) else {}

    emitted = int(counts.get("emitted_handoffs", 0) or 0)
    ack_rows = int(counts.get("ack_rows", 0) or 0)
    proxy_rows = int(counts.get("proxy_dispatch_rows", 0) or 0)
    fill_rows = int(counts.get("fill_event_rows", 0) or 0)
    fill_rejects = int(checks.get("fill_receiver_reject_count", counts.get("fill_reject_rows", 0) or 0) or 0)
    proxy_rejects = int(checks.get("proxy_reject_count", counts.get("proxy_reject_rows", 0) or 0) or 0)

    forward_rate = (ack_rows / emitted) if emitted > 0 else 0.0

    failures: List[Dict[str, Any]] = []

    if forward_rate < float(min_forward_rate):
        failures.append({"rule": "min_forward_rate", "actual": forward_rate, "expected_min": float(min_forward_rate)})
    if fill_rejects > int(max_fill_rejects):
        failures.append({"rule": "max_fill_rejects", "actual": fill_rejects, "expected_max": int(max_fill_rejects)})
    if proxy_rejects > int(max_proxy_rejects):
        failures.append({"rule": "max_proxy_rejects", "actual": proxy_rejects, "expected_max": int(max_proxy_rejects)})
    if bool(require_all_signal_ids_seen):
        for key in ("all_ack_signal_ids_seen", "all_proxy_signal_ids_seen", "all_fill_signal_ids_seen"):
            if bool(checks.get(key)) is not True:
                failures.append({"rule": key, "actual": checks.get(key), "expected": True})
    if bool(checks.get("all_acks_forwarded")) is not True:
        failures.append({"rule": "all_acks_forwarded", "actual": checks.get("all_acks_forwarded"), "expected": True})

    validation = {
        "schema": "moneyfan.freqtrade.contract_path_replay_validation.v1",
        "generated_at_utc": utc_now_iso(),
        "result": "pass" if not failures else "fail",
        "context": {
            "exchange_target": str(replay_summary.get("exchange_target", "") or ""),
            "data_source": str(replay_summary.get("data_source", "") or ""),
        },
        "replay_params": dict(params),
        "counts": {
            "emitted_handoffs": emitted,
            "ack_rows": ack_rows,
            "proxy_dispatch_rows": proxy_rows,
            "fill_event_rows": fill_rows,
            "fill_reject_rows": fill_rejects,
            "proxy_reject_rows": proxy_rejects,
            "forward_rate": forward_rate,
        },
        "thresholds": {
            "min_forward_rate": float(min_forward_rate),
            "max_fill_rejects": int(max_fill_rejects),
            "max_proxy_rejects": int(max_proxy_rejects),
            "require_all_signal_ids_seen": bool(require_all_signal_ids_seen),
        },
        "checks": checks,
        "failures": failures,
    }
    return validation


def resolve_threshold_profile(
    profile_name: str = "",
    exchange_target: str = "",
    data_source: str = "",
) -> Dict[str, Any]:
    if str(profile_name or "").strip():
        key = str(profile_name).strip()
        if key not in THRESHOLD_PROFILES:
            raise ValueError(f"Unknown threshold profile: {key}")
        return dict(THRESHOLD_PROFILES[key])
    combo_key = f"{str(exchange_target or '').strip()}__{str(data_source or '').strip()}"
    if combo_key in THRESHOLD_PROFILES:
        return dict(THRESHOLD_PROFILES[combo_key])
    return dict(THRESHOLD_PROFILES["default_strict"])


def build_replay_validation_markdown(validation: Dict[str, Any], replay_json_path: Optional[str] = None) -> str:
    counts = validation.get("counts", {}) if isinstance(validation.get("counts"), dict) else {}
    thresholds = validation.get("thresholds", {}) if isinstance(validation.get("thresholds"), dict) else {}
    checks = validation.get("checks", {}) if isinstance(validation.get("checks"), dict) else {}
    ctx = validation.get("context", {}) if isinstance(validation.get("context"), dict) else {}
    failures = validation.get("failures", []) if isinstance(validation.get("failures"), list) else []
    profile = validation.get("profile", {}) if isinstance(validation.get("profile"), dict) else {}

    lines: List[str] = []
    lines.append("# Contract Path Replay Validation Report")
    lines.append("")
    lines.append(f"- Generated: `{validation.get('generated_at_utc')}`")
    lines.append(f"- Result: `{validation.get('result')}`")
    if replay_json_path:
        lines.append(f"- Replay Summary JSON: `{replay_json_path}`")
    if ctx.get("exchange_target"):
        lines.append(f"- `exchange_target`: `{ctx.get('exchange_target')}`")
    if ctx.get("data_source"):
        lines.append(f"- `data_source`: `{ctx.get('data_source')}`")
    if profile:
        lines.append(f"- Threshold Profile: `{profile.get('selected')}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(
        f"- emitted={counts.get('emitted_handoffs', 0)} ack={counts.get('ack_rows', 0)} "
        f"proxy_dispatch={counts.get('proxy_dispatch_rows', 0)} fill_events={counts.get('fill_event_rows', 0)}"
    )
    lines.append(
        f"- forward_rate={counts.get('forward_rate')} fill_rejects={counts.get('fill_reject_rows', 0)} "
        f"proxy_rejects={counts.get('proxy_reject_rows', 0)}"
    )
    lines.append("")

    lines.append("## Thresholds")
    lines.append("")
    if isinstance(profile.get("resolved"), dict):
        lines.append(f"- Resolved profile thresholds: `{json.dumps(profile.get('resolved'), sort_keys=True)}`")
    for k in ("min_forward_rate", "max_fill_rejects", "max_proxy_rejects", "require_all_signal_ids_seen"):
        lines.append(f"- `{k}`: {thresholds.get(k)}")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    for k in ("all_acks_forwarded", "all_ack_signal_ids_seen", "all_proxy_signal_ids_seen", "all_fill_signal_ids_seen"):
        lines.append(f"- `{k}`: {checks.get(k)}")
    lines.append("")

    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("_No failures._")
    else:
        for f in failures:
            lines.append(f"- `{f.get('rule')}` actual={f.get('actual')} expected={f.get('expected', f.get('expected_min', f.get('expected_max')))}")
    lines.append("")
    return "\n".join(lines)


def run_validation_cli(args: argparse.Namespace) -> int:
    exchange_target = str(args.exchange_target or "")
    data_source = str(args.data_source or "")
    profile = resolve_threshold_profile(
        profile_name=str(args.threshold_profile or ""),
        exchange_target=exchange_target,
        data_source=data_source,
    )

    if str(args.replay_json or "").strip():
        replay_summary = _load_json(Path(args.replay_json))
    else:
        replay_summary = run_contract_path_replay(
            runtime_dir=Path(args.runtime_dir),
            batches=int(args.batches),
            batch_size=int(args.batch_size),
            bridge_max_records=int(args.bridge_max_records or 0),
            exchange_target=exchange_target,
            data_source=data_source,
        )

    replay_summary["exchange_target"] = exchange_target
    replay_summary["data_source"] = data_source

    validation = validate_replay_summary(
        replay_summary=replay_summary,
        min_forward_rate=float(args.min_forward_rate if args.min_forward_rate is not None else profile["min_forward_rate"]),
        max_fill_rejects=int(args.max_fill_rejects if args.max_fill_rejects is not None else profile["max_fill_rejects"]),
        max_proxy_rejects=int(args.max_proxy_rejects if args.max_proxy_rejects is not None else profile["max_proxy_rejects"]),
        require_all_signal_ids_seen=bool(
            args.require_all_signal_ids_seen
            if args.require_all_signal_ids_seen is not None
            else profile["require_all_signal_ids_seen"]
        ),
    )
    validation["profile"] = {
        "selected": str(args.threshold_profile or "") or f"{exchange_target}__{data_source}" or "default_strict",
        "resolved": profile,
    }

    replay_json_out = Path(args.replay_json_out)
    validation_json_out = Path(args.validation_json_out)
    validation_md_out = Path(args.validation_md_out)
    write_json(replay_json_out, replay_summary)
    write_json(validation_json_out, validation)
    md = build_replay_validation_markdown(validation, replay_json_path=str(replay_json_out))
    write_text(validation_md_out, md)

    timestamped_md = None
    if bool(args.also_write_timestamped):
        timestamped_md = build_timestamped_path(validation_md_out, stamp=(str(args.timestamp_stamp).strip() or None))
        write_text(timestamped_md, md)

    if bool(args.print_json):
        print(json.dumps({"replay": replay_summary, "validation": validation}, indent=2))
    else:
        print(
            "✅ Contract path replay validation "
            f"result={validation['result']} "
            f"forward_rate={validation['counts']['forward_rate']} "
            f"fill_rejects={validation['counts']['fill_reject_rows']} "
            f"proxy_rejects={validation['counts']['proxy_reject_rows']}"
        )
        print(f"📝 Replay JSON: {replay_json_out}")
        print(f"🧾 Validation JSON: {validation_json_out}")
        print(f"📄 Validation MD: {validation_md_out}")
        if timestamped_md is not None:
            print(f"🗂️  Snapshot: {timestamped_md}")
    return 0 if validation["result"] == "pass" or bool(args.no_fail_exit) else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run and validate repeatable contract-path replay with pass/fail thresholds")
    p.add_argument("--runtime-dir", type=str, default="runtime/contract_path_replay_validation")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bridge-max-records", type=int, default=0)
    p.add_argument("--replay-json", type=str, default="", help="Existing replay summary JSON to validate (skip replay run)")
    p.add_argument("--replay-json-out", type=str, default="runtime/contract_path_replay_summary.json")
    p.add_argument("--validation-json-out", type=str, default="runtime/contract_path_replay_validation.json")
    p.add_argument("--validation-md-out", type=str, default="runtime/contract_path_replay_validation.md")
    p.add_argument("--threshold-profile", type=str, default="",
                   help="Named validation threshold profile (defaults to exchange_target__data_source if available)")
    p.add_argument("--min-forward-rate", type=float, default=None)
    p.add_argument("--max-fill-rejects", type=int, default=None)
    p.add_argument("--max-proxy-rejects", type=int, default=None)
    p.add_argument("--require-all-signal-ids-seen", action="store_true", dest="require_all_signal_ids_seen", default=None)
    p.add_argument("--no-require-all-signal-ids-seen", action="store_false", dest="require_all_signal_ids_seen")
    p.add_argument("--exchange-target", type=str, default="coinbase_advanced")
    p.add_argument("--data-source", type=str, default="binance")
    p.add_argument("--also-write-timestamped", action="store_true")
    p.add_argument("--timestamp-stamp", type=str, default="")
    p.add_argument("--no-fail-exit", action="store_true", help="Exit 0 even when validation result=fail")
    p.add_argument("--print-json", action="store_true")
    return p


def main() -> int:
    return run_validation_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
