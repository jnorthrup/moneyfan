#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.pair_context_sampler_audit_smoke import run_sampler_audit_smoke


THRESHOLD_PROFILES: Dict[str, Dict[str, Any]] = {
    "default_strict": {
        "min_trace_rows_valid": 1,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": None,
        "require_conformance_pass": True,
    },
    "coinbase_advanced__binance": {
        "min_trace_rows_valid": 2,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": 64.0,
        "require_conformance_pass": True,
    },
    "coinbase_advanced__binance_relaxed": {
        "min_trace_rows_valid": 1,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": 128.0,
        "require_conformance_pass": True,
    },
    "coinbase_advanced__coinbase_advanced": {
        "min_trace_rows_valid": 2,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": 64.0,
        "require_conformance_pass": True,
    },
    "coinbase_advanced__coinbase_advanced_relaxed": {
        "min_trace_rows_valid": 1,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": 96.0,
        "require_conformance_pass": True,
    },
    "freqtrade_paper__binance": {
        "min_trace_rows_valid": 1,
        "max_focal_pair_inclusion_failures": 0,
        "min_pair_width_p95": 1.0,
        "max_pair_width_p95": 128.0,
        "require_conformance_pass": True,
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


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


def resolve_threshold_profile(profile_name: str = "", exchange_target: str = "", data_source: str = "") -> Dict[str, Any]:
    if str(profile_name or "").strip():
        key = str(profile_name).strip()
        if key not in THRESHOLD_PROFILES:
            available = ", ".join(sorted(THRESHOLD_PROFILES.keys()))
            raise ValueError(f"Unknown threshold profile: {key}. Available: {available}")
        return dict(THRESHOLD_PROFILES[key])
    combo = f"{str(exchange_target or '').strip()}__{str(data_source or '').strip()}"
    if combo in THRESHOLD_PROFILES:
        return dict(THRESHOLD_PROFILES[combo])
    return dict(THRESHOLD_PROFILES["default_strict"])


def validate_sampler_audit_summary(
    summary: Dict[str, Any],
    *,
    min_trace_rows_valid: int = 1,
    max_focal_pair_inclusion_failures: int = 0,
    min_pair_width_p95: Optional[float] = 1.0,
    max_pair_width_p95: Optional[float] = None,
    require_conformance_pass: bool = True,
) -> Dict[str, Any]:
    results = summary.get("results") if isinstance(summary.get("results"), dict) else {}
    context = summary.get("context") if isinstance(summary.get("context"), dict) else {}

    conformance_result = str(results.get("conformance_result", "") or "")
    rows_valid = int(results.get("trace_report_rows_valid", 0) or 0)
    focal_failures = int(results.get("focal_pair_inclusion_failures", 0) or 0)
    pair_width_p95_raw = results.get("pair_width_p95")
    pair_width_p95 = None if pair_width_p95_raw is None else float(pair_width_p95_raw)

    failures: List[Dict[str, Any]] = []
    if bool(require_conformance_pass) and conformance_result != "pass":
        failures.append({"rule": "require_conformance_pass", "actual": conformance_result, "expected": "pass"})
    if rows_valid < int(min_trace_rows_valid):
        failures.append({"rule": "min_trace_rows_valid", "actual": rows_valid, "expected_min": int(min_trace_rows_valid)})
    if focal_failures > int(max_focal_pair_inclusion_failures):
        failures.append(
            {
                "rule": "max_focal_pair_inclusion_failures",
                "actual": focal_failures,
                "expected_max": int(max_focal_pair_inclusion_failures),
            }
        )
    if min_pair_width_p95 is not None:
        if pair_width_p95 is None or pair_width_p95 < float(min_pair_width_p95):
            failures.append(
                {"rule": "min_pair_width_p95", "actual": pair_width_p95, "expected_min": float(min_pair_width_p95)}
            )
    if max_pair_width_p95 is not None:
        if pair_width_p95 is None or pair_width_p95 > float(max_pair_width_p95):
            failures.append(
                {"rule": "max_pair_width_p95", "actual": pair_width_p95, "expected_max": float(max_pair_width_p95)}
            )

    return {
        "schema": "moneyfan.pair_context_sampler_audit_validation.v1",
        "generated_at_utc": utc_now_iso(),
        "result": "pass" if not failures else "fail",
        "context": {
            "exchange_target": str(context.get("exchange_target", "") or ""),
            "data_source": str(context.get("data_source", "") or ""),
        },
        "counts": {
            "trace_rows_written": int(results.get("trace_rows_written", 0) or 0),
            "trace_report_rows_valid": rows_valid,
            "focal_pair_inclusion_failures": focal_failures,
            "pair_width_p95": pair_width_p95,
        },
        "thresholds": {
            "min_trace_rows_valid": int(min_trace_rows_valid),
            "max_focal_pair_inclusion_failures": int(max_focal_pair_inclusion_failures),
            "min_pair_width_p95": None if min_pair_width_p95 is None else float(min_pair_width_p95),
            "max_pair_width_p95": None if max_pair_width_p95 is None else float(max_pair_width_p95),
            "require_conformance_pass": bool(require_conformance_pass),
        },
        "checks": {
            "conformance_passed": conformance_result == "pass",
        },
        "failures": failures,
    }


def build_validation_markdown(validation: Dict[str, Any], summary_json_path: Optional[str] = None) -> str:
    ctx = validation.get("context", {}) if isinstance(validation.get("context"), dict) else {}
    counts = validation.get("counts", {}) if isinstance(validation.get("counts"), dict) else {}
    thresholds = validation.get("thresholds", {}) if isinstance(validation.get("thresholds"), dict) else {}
    checks = validation.get("checks", {}) if isinstance(validation.get("checks"), dict) else {}
    failures = validation.get("failures", []) if isinstance(validation.get("failures"), list) else []
    profile = validation.get("profile", {}) if isinstance(validation.get("profile"), dict) else {}

    lines: List[str] = []
    lines.append("# Sampler Audit Validation Report")
    lines.append("")
    lines.append(f"- Generated: `{validation.get('generated_at_utc')}`")
    lines.append(f"- Result: `{validation.get('result')}`")
    if summary_json_path:
        lines.append(f"- Sampler Audit Summary JSON: `{summary_json_path}`")
    if ctx.get("exchange_target"):
        lines.append(f"- `exchange_target`: `{ctx.get('exchange_target')}`")
    if ctx.get("data_source"):
        lines.append(f"- `data_source`: `{ctx.get('data_source')}`")
    if profile:
        lines.append(f"- Threshold Profile: `{profile.get('selected')}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(f"- trace_rows_written={counts.get('trace_rows_written')} trace_report_rows_valid={counts.get('trace_report_rows_valid')}")
    lines.append(f"- focal_pair_inclusion_failures={counts.get('focal_pair_inclusion_failures')} pair_width_p95={counts.get('pair_width_p95')}")
    lines.append("")

    lines.append("## Thresholds")
    lines.append("")
    if isinstance(profile.get("resolved"), dict):
        lines.append(f"- Resolved profile thresholds: `{json.dumps(profile.get('resolved'), sort_keys=True)}`")
    for k in (
        "min_trace_rows_valid",
        "max_focal_pair_inclusion_failures",
        "min_pair_width_p95",
        "max_pair_width_p95",
        "require_conformance_pass",
    ):
        lines.append(f"- `{k}`: {thresholds.get(k)}")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append(f"- `conformance_passed`: {checks.get('conformance_passed')}")
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
    if bool(args.list_threshold_profiles):
        payload = {"profiles": {k: THRESHOLD_PROFILES[k] for k in sorted(THRESHOLD_PROFILES.keys())}}
        if bool(args.print_json):
            print(json.dumps(payload, indent=2))
        else:
            print("Sampler threshold profiles:")
            for name in sorted(payload["profiles"].keys()):
                print(f"- {name}: {json.dumps(payload['profiles'][name], sort_keys=True)}")
        return 0

    exchange_target = str(args.exchange_target or "")
    data_source = str(args.data_source or "")
    profile = resolve_threshold_profile(
        profile_name=str(args.threshold_profile or ""),
        exchange_target=exchange_target,
        data_source=data_source,
    )

    if str(args.summary_json or "").strip():
        summary = _load_json(Path(args.summary_json))
    else:
        summary = run_sampler_audit_smoke(
            runtime_dir=Path(args.runtime_dir),
            exchange_target=exchange_target,
            data_source=data_source,
        )

    summary["context"] = {"exchange_target": exchange_target, "data_source": data_source}

    validation = validate_sampler_audit_summary(
        summary,
        min_trace_rows_valid=int(args.min_trace_rows_valid if args.min_trace_rows_valid is not None else profile["min_trace_rows_valid"]),
        max_focal_pair_inclusion_failures=int(
            args.max_focal_pair_inclusion_failures
            if args.max_focal_pair_inclusion_failures is not None
            else profile["max_focal_pair_inclusion_failures"]
        ),
        min_pair_width_p95=(
            None
            if args.min_pair_width_p95 is None and profile["min_pair_width_p95"] is None
            else float(args.min_pair_width_p95 if args.min_pair_width_p95 is not None else profile["min_pair_width_p95"])
        ),
        max_pair_width_p95=(
            None
            if args.max_pair_width_p95 is None and profile["max_pair_width_p95"] is None
            else float(args.max_pair_width_p95 if args.max_pair_width_p95 is not None else profile["max_pair_width_p95"])
        ),
        require_conformance_pass=bool(
            args.require_conformance_pass
            if args.require_conformance_pass is not None
            else profile["require_conformance_pass"]
        ),
    )
    validation["profile"] = {
        "selected": str(args.threshold_profile or "") or f"{exchange_target}__{data_source}" or "default_strict",
        "resolved": profile,
    }

    summary_json_out = Path(args.summary_json_out)
    validation_json_out = Path(args.validation_json_out)
    validation_md_out = Path(args.validation_md_out)
    write_json(summary_json_out, summary)
    write_json(validation_json_out, validation)
    md = build_validation_markdown(validation, summary_json_path=str(summary_json_out))
    write_text(validation_md_out, md)
    timestamped_md = None
    if bool(args.also_write_timestamped):
        timestamped_md = build_timestamped_path(validation_md_out, stamp=(str(args.timestamp_stamp).strip() or None))
        write_text(timestamped_md, md)

    if bool(args.print_json):
        print(json.dumps({"summary": summary, "validation": validation}, indent=2))
    else:
        print(
            "✅ Sampler audit validation "
            f"result={validation['result']} "
            f"rows_valid={validation['counts']['trace_report_rows_valid']} "
            f"focal_failures={validation['counts']['focal_pair_inclusion_failures']} "
            f"pair_width_p95={validation['counts']['pair_width_p95']}"
        )
        print(f"📝 Summary JSON: {summary_json_out}")
        print(f"🧾 Validation JSON: {validation_json_out}")
        print(f"📄 Validation MD: {validation_md_out}")
        if timestamped_md is not None:
            print(f"🗂️  Snapshot: {timestamped_md}")
    return 0 if validation["result"] == "pass" or bool(args.no_fail_exit) else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run or validate sampler-audit smoke output with thresholded pass/fail checks")
    p.add_argument("--runtime-dir", type=str, default="runtime/pair_context_sampler_audit_validation")
    p.add_argument("--summary-json", type=str, default="", help="Existing sampler audit smoke summary JSON (skip smoke run)")
    p.add_argument("--summary-json-out", type=str, default="runtime/pair_context_sampler_audit_smoke_summary.json")
    p.add_argument("--validation-json-out", type=str, default="runtime/pair_context_sampler_audit_validation.json")
    p.add_argument("--validation-md-out", type=str, default="runtime/pair_context_sampler_audit_validation.md")
    p.add_argument("--threshold-profile", type=str, default="", help="Named threshold profile (defaults to exchange_target__data_source if available)")
    p.add_argument("--min-trace-rows-valid", type=int, default=None)
    p.add_argument("--max-focal-pair-inclusion-failures", type=int, default=None)
    p.add_argument("--min-pair-width-p95", type=float, default=None)
    p.add_argument("--max-pair-width-p95", type=float, default=None)
    p.add_argument("--require-conformance-pass", action="store_true", dest="require_conformance_pass", default=None)
    p.add_argument("--no-require-conformance-pass", action="store_false", dest="require_conformance_pass")
    p.add_argument("--exchange-target", type=str, default="coinbase_advanced")
    p.add_argument("--data-source", type=str, default="binance")
    p.add_argument("--also-write-timestamped", action="store_true")
    p.add_argument("--timestamp-stamp", type=str, default="")
    p.add_argument("--print-json", action="store_true")
    p.add_argument("--list-threshold-profiles", action="store_true")
    p.add_argument("--no-fail-exit", action="store_true")
    return p


def main() -> int:
    return run_validation_cli(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
