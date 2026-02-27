#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt_num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _fmt_pct(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{100.0 * float(v):.{digits}f}%"
    except Exception:
        return str(v)


def _fmt_delta(v: Optional[float], digits: int = 3, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    sign = "+" if float(v) > 0 else ""
    return f"{sign}{float(v):.{digits}f}{suffix}"


def _summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return report.get("summary") if isinstance(report.get("summary"), dict) else {}


def _fidelity_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    s = _summary(report)
    return s.get("fidelity_metrics") if isinstance(s.get("fidelity_metrics"), dict) else {}


def _records(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = report.get("records")
    return rows if isinstance(rows, list) else []


def _record_map(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _records(report):
        if not isinstance(row, dict):
            continue
        sid = str(row.get("signal_id", "") or "").strip()
        if sid:
            out[sid] = row
    return out


def _metric_delta(candidate: Dict[str, Any], baseline: Dict[str, Any], key: str) -> Optional[float]:
    cv = _as_float(candidate.get(key))
    bv = _as_float(baseline.get(key))
    if cv is None or bv is None:
        return None
    return float(cv - bv)


def _joined_matched_rows(
    baseline_report: Dict[str, Any],
    candidate_report: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    bmap = _record_map(baseline_report)
    cmap = _record_map(candidate_report)
    out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for sid in sorted(set(bmap.keys()) & set(cmap.keys())):
        br = bmap[sid]
        cr = cmap[sid]
        if br.get("reconcile_status") != "ack_fill_matched":
            continue
        if cr.get("reconcile_status") != "ack_fill_matched":
            continue
        out.append((sid, br, cr))
    return out


def _top_abs(rows: List[Dict[str, Any]], key: str, n: int = 10) -> List[Dict[str, Any]]:
    vals = [r for r in rows if _as_float(r.get(key)) is not None]
    vals.sort(key=lambda r: abs(float(r.get(key, 0.0) or 0.0)), reverse=True)
    return vals[: max(0, int(n))]


def build_fidelity_compare_markdown_report(
    baseline_report: Dict[str, Any],
    candidate_report: Dict[str, Any],
    baseline_path: Optional[str] = None,
    candidate_path: Optional[str] = None,
    top_n: int = 10,
) -> str:
    bs = _summary(baseline_report)
    cs = _summary(candidate_report)
    bfm = _fidelity_metrics(baseline_report)
    cfm = _fidelity_metrics(candidate_report)

    joined = _joined_matched_rows(baseline_report, candidate_report)
    per_signal_deltas: List[Dict[str, Any]] = []
    for sid, br, cr in joined:
        b_abs_err = _as_float(br.get("abs_pred_error_bps"))
        c_abs_err = _as_float(cr.get("abs_pred_error_bps"))
        b_slip = _as_float(br.get("adverse_entry_slippage_bps"))
        c_slip = _as_float(cr.get("adverse_entry_slippage_bps"))
        b_real = _as_float(br.get("realized_return_bps"))
        c_real = _as_float(cr.get("realized_return_bps"))
        b_pred_err = _as_float(br.get("pred_error_bps"))
        c_pred_err = _as_float(cr.get("pred_error_bps"))
        per_signal_deltas.append(
            {
                "signal_id": sid,
                "pair": cr.get("dispatch_pair") or br.get("dispatch_pair"),
                "side": cr.get("dispatch_side") or br.get("dispatch_side"),
                "baseline_abs_pred_error_bps": b_abs_err,
                "candidate_abs_pred_error_bps": c_abs_err,
                "delta_abs_pred_error_bps": (c_abs_err - b_abs_err) if (c_abs_err is not None and b_abs_err is not None) else None,
                "baseline_adverse_entry_slippage_bps": b_slip,
                "candidate_adverse_entry_slippage_bps": c_slip,
                "delta_adverse_entry_slippage_bps": (c_slip - b_slip) if (c_slip is not None and b_slip is not None) else None,
                "baseline_realized_return_bps": b_real,
                "candidate_realized_return_bps": c_real,
                "delta_realized_return_bps": (c_real - b_real) if (c_real is not None and b_real is not None) else None,
                "baseline_pred_error_bps": b_pred_err,
                "candidate_pred_error_bps": c_pred_err,
                "delta_pred_error_bps": (c_pred_err - b_pred_err) if (c_pred_err is not None and b_pred_err is not None) else None,
            }
        )

    worst_mae_regressions = _top_abs(
        [r for r in per_signal_deltas if _as_float(r.get("delta_abs_pred_error_bps")) is not None and float(r["delta_abs_pred_error_bps"]) > 0],
        "delta_abs_pred_error_bps",
        n=top_n,
    )
    best_mae_improvements = _top_abs(
        [r for r in per_signal_deltas if _as_float(r.get("delta_abs_pred_error_bps")) is not None and float(r["delta_abs_pred_error_bps"]) < 0],
        "delta_abs_pred_error_bps",
        n=top_n,
    )

    # Sort improvements most negative first
    best_mae_improvements.sort(key=lambda r: float(r.get("delta_abs_pred_error_bps", 0.0) or 0.0))

    delta = {
        "dispatch_total": _metric_delta(cs, bs, "dispatch_total"),
        "dispatch_fully_reconciled": _metric_delta(cs, bs, "dispatch_fully_reconciled"),
        "records_with_realized_return": _metric_delta(cs, bs, "records_with_realized_return"),
        "orphan_ack_count": _metric_delta(cs, bs, "orphan_ack_count"),
        "orphan_fill_count": _metric_delta(cs, bs, "orphan_fill_count"),
        "mean_abs_pred_error_bps": _metric_delta(cfm, bfm, "mean_abs_pred_error_bps"),
        "rmse_pred_error_bps": _metric_delta(cfm, bfm, "rmse_pred_error_bps"),
        "directional_accuracy": _metric_delta(cfm, bfm, "directional_accuracy"),
        "pearson_pred_vs_realized_bps": _metric_delta(cfm, bfm, "pearson_pred_vs_realized_bps"),
        "mean_adverse_entry_slippage_bps": _metric_delta(cfm, bfm, "mean_adverse_entry_slippage_bps"),
    }

    def _winner_for_error_metric(delta_value: Optional[float]) -> str:
        if delta_value is None:
            return "n/a"
        if delta_value < 0:
            return "candidate"
        if delta_value > 0:
            return "baseline"
        return "tie"

    def _winner_for_higher_better(delta_value: Optional[float]) -> str:
        if delta_value is None:
            return "n/a"
        if delta_value > 0:
            return "candidate"
        if delta_value < 0:
            return "baseline"
        return "tie"

    lines: List[str] = []
    lines.append("# HRM/Freqtrade Fidelity Compare Report")
    lines.append("")
    lines.append(f"- Generated: `{utc_now_iso()}`")
    if baseline_path:
        lines.append(f"- Baseline: `{baseline_path}`")
    if candidate_path:
        lines.append(f"- Candidate: `{candidate_path}`")
    lines.append("")

    lines.append("## Summary Delta")
    lines.append("")
    lines.append(
        f"- `dispatch_total`: baseline={int(bs.get('dispatch_total', 0) or 0)} "
        f"candidate={int(cs.get('dispatch_total', 0) or 0)} delta={_fmt_delta(delta['dispatch_total'], 0)}"
    )
    lines.append(
        f"- `dispatch_fully_reconciled`: baseline={int(bs.get('dispatch_fully_reconciled', 0) or 0)} "
        f"candidate={int(cs.get('dispatch_fully_reconciled', 0) or 0)} delta={_fmt_delta(delta['dispatch_fully_reconciled'], 0)}"
    )
    lines.append(
        f"- `records_with_realized_return`: baseline={int(bs.get('records_with_realized_return', 0) or 0)} "
        f"candidate={int(cs.get('records_with_realized_return', 0) or 0)} delta={_fmt_delta(delta['records_with_realized_return'], 0)}"
    )
    lines.append(
        f"- `orphan_ack_count`: baseline={int(bs.get('orphan_ack_count', 0) or 0)} "
        f"candidate={int(cs.get('orphan_ack_count', 0) or 0)} delta={_fmt_delta(delta['orphan_ack_count'], 0)}"
    )
    lines.append(
        f"- `orphan_fill_count`: baseline={int(bs.get('orphan_fill_count', 0) or 0)} "
        f"candidate={int(cs.get('orphan_fill_count', 0) or 0)} delta={_fmt_delta(delta['orphan_fill_count'], 0)}"
    )
    lines.append("")

    lines.append("## Fidelity Metric Delta")
    lines.append("")
    lines.append(
        f"- `mean_abs_pred_error_bps`: baseline={_fmt_num(bfm.get('mean_abs_pred_error_bps'))} "
        f"candidate={_fmt_num(cfm.get('mean_abs_pred_error_bps'))} "
        f"delta={_fmt_delta(delta['mean_abs_pred_error_bps'])} bps "
        f"(winner: `{_winner_for_error_metric(delta['mean_abs_pred_error_bps'])}`)"
    )
    lines.append(
        f"- `rmse_pred_error_bps`: baseline={_fmt_num(bfm.get('rmse_pred_error_bps'))} "
        f"candidate={_fmt_num(cfm.get('rmse_pred_error_bps'))} "
        f"delta={_fmt_delta(delta['rmse_pred_error_bps'])} bps "
        f"(winner: `{_winner_for_error_metric(delta['rmse_pred_error_bps'])}`)"
    )
    lines.append(
        f"- `directional_accuracy`: baseline={_fmt_pct(bfm.get('directional_accuracy'))} "
        f"candidate={_fmt_pct(cfm.get('directional_accuracy'))} "
        f"delta={_fmt_delta(delta['directional_accuracy'], 4)} "
        f"(winner: `{_winner_for_higher_better(delta['directional_accuracy'])}`)"
    )
    lines.append(
        f"- `pearson_pred_vs_realized_bps`: baseline={_fmt_num(bfm.get('pearson_pred_vs_realized_bps'))} "
        f"candidate={_fmt_num(cfm.get('pearson_pred_vs_realized_bps'))} "
        f"delta={_fmt_delta(delta['pearson_pred_vs_realized_bps'])} "
        f"(winner: `{_winner_for_higher_better(delta['pearson_pred_vs_realized_bps'])}`)"
    )
    lines.append(
        f"- `mean_adverse_entry_slippage_bps`: baseline={_fmt_num(bfm.get('mean_adverse_entry_slippage_bps'))} "
        f"candidate={_fmt_num(cfm.get('mean_adverse_entry_slippage_bps'))} "
        f"delta={_fmt_delta(delta['mean_adverse_entry_slippage_bps'])} bps "
        f"(winner: `{_winner_for_error_metric(delta['mean_adverse_entry_slippage_bps'])}`)"
    )
    lines.append("")

    lines.append("## Joined Matched Sample")
    lines.append("")
    lines.append(f"- Joined `ack_fill_matched` rows by `signal_id`: {len(per_signal_deltas)}")
    lines.append("")

    def render_table(title: str, rows: List[Dict[str, Any]], cols: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("_No rows._")
            lines.append("")
            return
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            vals: List[str] = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    vals.append(_fmt_num(v))
                elif v is None:
                    vals.append("n/a")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    render_table(
        f"Top {top_n} MAE Regressions (Candidate Worse)",
        worst_mae_regressions,
        [
            "signal_id",
            "pair",
            "side",
            "baseline_abs_pred_error_bps",
            "candidate_abs_pred_error_bps",
            "delta_abs_pred_error_bps",
            "baseline_adverse_entry_slippage_bps",
            "candidate_adverse_entry_slippage_bps",
            "delta_adverse_entry_slippage_bps",
        ],
    )

    render_table(
        f"Top {top_n} MAE Improvements (Candidate Better)",
        best_mae_improvements,
        [
            "signal_id",
            "pair",
            "side",
            "baseline_abs_pred_error_bps",
            "candidate_abs_pred_error_bps",
            "delta_abs_pred_error_bps",
            "baseline_realized_return_bps",
            "candidate_realized_return_bps",
            "delta_realized_return_bps",
        ],
    )

    lines.append("## Notes")
    lines.append("")
    lines.append("- Error/slippage metrics are lower-is-better; directional accuracy/correlation are higher-is-better.")
    lines.append("- `delta_*` values are `candidate - baseline`.")
    lines.append("- Per-signal tables only include rows matched in both reports with `reconcile_status = ack_fill_matched`.")
    lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build_timestamped_compare_report_path(base_path: Path, stamp: Optional[str] = None) -> Path:
    ts = str(stamp or _utc_stamp())
    suffix = "".join(base_path.suffixes) or ".md"
    stem = base_path.name[: -len(suffix)] if suffix and base_path.name.endswith(suffix) else base_path.stem
    return base_path.with_name(f"{stem}_{ts}{suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a markdown compare report for two HRM/Freqtrade fidelity reconciliation JSON files")
    p.add_argument("--baseline-json", type=str, required=True, help="Baseline reconciliation JSON path")
    p.add_argument("--candidate-json", type=str, required=True, help="Candidate reconciliation JSON path")
    p.add_argument("--out-md", type=str, default="runtime/hrm_freqtrade_fidelity_compare_report.md", help="Output markdown path")
    p.add_argument("--also-write-timestamped", action="store_true",
                   help="Also write a timestamped snapshot next to --out-md")
    p.add_argument("--timestamp-stamp", type=str, default="",
                   help="Override UTC timestamp token for --also-write-timestamped (testing/manual replay)")
    p.add_argument("--top-n", type=int, default=10, help="Rows for top regression/improvement tables")
    p.add_argument("--print-path", action="store_true", help="Print output path after writing")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    bpath = Path(args.baseline_json)
    cpath = Path(args.candidate_json)
    out = Path(args.out_md)
    b = _load_json(bpath)
    c = _load_json(cpath)
    md = build_fidelity_compare_markdown_report(
        baseline_report=b,
        candidate_report=c,
        baseline_path=str(bpath),
        candidate_path=str(cpath),
        top_n=int(args.top_n),
    )
    write_text(out, md)
    timestamped_path = None
    if bool(args.also_write_timestamped):
        timestamped_path = build_timestamped_compare_report_path(
            out,
            stamp=(str(args.timestamp_stamp).strip() or None),
        )
        write_text(timestamped_path, md)
    if bool(args.print_path):
        print(str(out))
        if timestamped_path is not None:
            print(str(timestamped_path))
    else:
        print(f"✅ Wrote fidelity compare report: {out}")
        if timestamped_path is not None:
            print(f"🗂️  Snapshot: {timestamped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
