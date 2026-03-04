#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _fmt_num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _fmt_pct(x: Any, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0 * float(x):.{digits}f}%"
    except Exception:
        return str(x)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("reconciliation JSON must be an object")
    return payload


def _top_records(records: List[Dict[str, Any]], key: str, n: int = 10, reverse: bool = True) -> List[Dict[str, Any]]:
    rows = [r for r in records if isinstance(r, dict) and r.get(key) is not None]
    rows.sort(key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=reverse)
    return rows[: max(0, int(n))]


def build_fidelity_markdown_report(
    reconciliation: Dict[str, Any],
    reconciliation_json_path: Optional[str] = None,
    top_n: int = 10,
) -> str:
    summary = reconciliation.get("summary") if isinstance(reconciliation.get("summary"), dict) else {}
    fm = summary.get("fidelity_metrics") if isinstance(summary.get("fidelity_metrics"), dict) else {}
    records = reconciliation.get("records") if isinstance(reconciliation.get("records"), list) else []
    inputs = reconciliation.get("inputs") if isinstance(reconciliation.get("inputs"), dict) else {}

    generated_at = reconciliation.get("generated_at_utc") or utc_now_iso()
    schema = reconciliation.get("schema", "unknown")

    matched = [r for r in records if isinstance(r, dict) and r.get("reconcile_status") == "ack_fill_matched"]
    orphans = [r for r in records if isinstance(r, dict) and str(r.get("reconcile_status", "")).startswith("orphan_")]
    worst_abs_err = _top_records(matched, "abs_pred_error_bps", n=top_n, reverse=True)
    worst_slippage = _top_records(matched, "adverse_entry_slippage_bps", n=top_n, reverse=True)

    lines: List[str] = []
    lines.append("# HRM/Freqtrade Fidelity Report")
    lines.append("")
    lines.append(f"- Generated: `{generated_at}`")
    lines.append(f"- Schema: `{schema}`")
    if reconciliation_json_path:
        lines.append(f"- Source Report: `{reconciliation_json_path}`")
    if inputs:
        if inputs.get("dispatch_log_path"):
            lines.append(f"- Dispatch Log: `{inputs.get('dispatch_log_path')}`")
        if inputs.get("ack_log_path"):
            lines.append(f"- Ack Log: `{inputs.get('ack_log_path')}`")
        if inputs.get("fill_log_path"):
            lines.append(f"- Fill Log: `{inputs.get('fill_log_path')}`")
    lines.append("")

    lines.append("## Outcome")
    lines.append("")
    lines.append(
        f"- `dispatch_total`: {int(summary.get('dispatch_total', 0) or 0)} | "
        f"`dispatch_fully_reconciled`: {int(summary.get('dispatch_fully_reconciled', 0) or 0)} | "
        f"`records_with_realized_return`: {int(summary.get('records_with_realized_return', 0) or 0)}"
    )
    lines.append(
        f"- `orphan_ack_count`: {int(summary.get('orphan_ack_count', 0) or 0)} | "
        f"`orphan_fill_count`: {int(summary.get('orphan_fill_count', 0) or 0)} | "
        f"`dispatch_with_fill`: {int(summary.get('dispatch_with_fill', 0) or 0)}"
    )
    lines.append("")

    lines.append("## Fidelity Metrics")
    lines.append("")
    lines.append(f"- `mean_abs_pred_error_bps`: {_fmt_num(fm.get('mean_abs_pred_error_bps'))} bps")
    lines.append(f"- `rmse_pred_error_bps`: {_fmt_num(fm.get('rmse_pred_error_bps'))} bps")
    lines.append(f"- `directional_accuracy`: {_fmt_pct(fm.get('directional_accuracy'))}")
    lines.append(f"- `pearson_pred_vs_realized_bps`: {_fmt_num(fm.get('pearson_pred_vs_realized_bps'))}")
    lines.append(f"- `mean_adverse_entry_slippage_bps`: {_fmt_num(fm.get('mean_adverse_entry_slippage_bps'))} bps")
    lines.append("")

    parse_errors = summary.get("parse_errors") if isinstance(summary.get("parse_errors"), dict) else {}
    lines.append("## Data Quality")
    lines.append("")
    lines.append(
        f"- Parse errors: dispatch={int(parse_errors.get('dispatch_log', 0) or 0)} "
        f"ack={int(parse_errors.get('ack_log', 0) or 0)} "
        f"fill={int(parse_errors.get('fill_log', 0) or 0)}"
    )
    lines.append(f"- Matched rows available for ranking sections: {len(matched)}")
    lines.append(f"- Orphan rows: {len(orphans)}")
    lines.append("")

    def _render_table(title: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("_No rows._")
            lines.append("")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for r in rows:
            vals: List[str] = []
            for c in columns:
                v = r.get(c)
                if isinstance(v, float):
                    if c.endswith("_pct"):
                        vals.append(_fmt_pct(v))
                    else:
                        vals.append(_fmt_num(v))
                elif v is None:
                    vals.append("n/a")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    _render_table(
        f"Top {top_n} Absolute Prediction Errors (Matched)",
        worst_abs_err,
        [
            "signal_id",
            "dispatch_pair",
            "dispatch_side",
            "confidence",
            "predicted_return_bps",
            "realized_return_bps",
            "pred_error_bps",
            "abs_pred_error_bps",
            "adverse_entry_slippage_bps",
            "ack_status",
        ],
    )

    _render_table(
        f"Top {top_n} Adverse Entry Slippage (Matched)",
        worst_slippage,
        [
            "signal_id",
            "dispatch_pair",
            "dispatch_side",
            "dispatch_price",
            "entry_price",
            "adverse_entry_slippage_bps",
            "predicted_return_bps",
            "realized_return_bps",
            "pred_error_bps",
        ],
    )

    orphan_rows = [r for r in records if isinstance(r, dict) and r.get("reconcile_status") in {"orphan_ack", "orphan_fill"}]
    orphan_rows = orphan_rows[: max(0, int(top_n))]
    _render_table(
        f"Sample Orphans (Top {top_n})",
        orphan_rows,
        ["signal_id", "reconcile_status", "ack_status", "fill_status", "fill_pair", "fill_side", "pnl_pct"],
    )

    lines.append("## Notes")
    lines.append("")
    lines.append("- `adverse_entry_slippage_bps` is signed so higher values are worse for the dispatched side.")
    lines.append("- `pred_error_bps = realized_return_bps - predicted_return_bps`.")
    lines.append("- `orphan_ack`/`orphan_fill` indicate join gaps by `signal_id` and should be investigated before trusting trend changes.")
    lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build_timestamped_report_path(base_path: Path, stamp: Optional[str] = None) -> Path:
    ts = str(stamp or _utc_stamp())
    suffix = "".join(base_path.suffixes) or ".md"
    stem = base_path.name[: -len(suffix)] if suffix and base_path.name.endswith(suffix) else base_path.stem
    filename = f"{stem}_{ts}{suffix}"
    return base_path.with_name(filename)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a markdown HRM/Freqtrade fidelity report from reconciliation JSON")
    p.add_argument("--reconciliation-json", type=str, required=True,
                   help="Input reconciliation JSON from execution.freqtrade_fidelity_reconcile or fidelity pipeline")
    p.add_argument("--out-md", type=str, default="runtime/hrm_freqtrade_fidelity_report.md",
                   help="Output markdown report path")
    p.add_argument("--also-write-timestamped", action="store_true",
                   help="Also write a timestamped snapshot next to --out-md")
    p.add_argument("--timestamp-stamp", type=str, default="",
                   help="Override UTC timestamp token for --also-write-timestamped (testing/manual replay)")
    p.add_argument("--top-n", type=int, default=10, help="Rows to include in top-error/slippage tables")
    p.add_argument("--print-path", action="store_true", help="Print output path after writing")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    src = Path(args.reconciliation_json)
    out = Path(args.out_md)
    payload = _load_json(src)
    md = build_fidelity_markdown_report(payload, reconciliation_json_path=str(src), top_n=int(args.top_n))
    write_text(out, md)
    timestamped_path = None
    if bool(args.also_write_timestamped):
        timestamped_path = build_timestamped_report_path(
            out,
            stamp=(str(args.timestamp_stamp).strip() or None),
        )
        write_text(timestamped_path, md)
    if bool(args.print_path):
        print(str(out))
        if timestamped_path is not None:
            print(str(timestamped_path))
    else:
        print(f"✅ Wrote fidelity report: {out}")
        if timestamped_path is not None:
            print(f"🗂️  Snapshot: {timestamped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
