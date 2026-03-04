#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                rows.append({"_parse_error": str(e), "_line_no": i, "_raw_line": raw[:500]})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({"value": obj, "_line_no": i})
    return rows


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, float(q))) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def build_pair_context_sampler_trace_report(trace_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    parse_errors = [r for r in trace_rows if r.get("_parse_error")]
    valid = [r for r in trace_rows if str(r.get("schema", "") or "") == "moneyfan.pair_context_sampler.v1"]

    widths: List[float] = []
    max_widths: List[float] = []
    focal_pairs: Counter[str] = Counter()
    exchange_targets: Counter[str] = Counter()
    data_sources: Counter[str] = Counter()
    sampler_versions: Counter[str] = Counter()
    sampler_policies: Counter[str] = Counter()
    ranker_versions: Counter[str] = Counter()
    ranker_names: Counter[str] = Counter()
    slot_orderings: Counter[str] = Counter()
    focal_inclusion_failures = 0

    for row in valid:
        try:
            widths.append(float(row.get("pair_width")))
        except Exception:
            pass
        try:
            max_widths.append(float(row.get("max_pair_width")))
        except Exception:
            pass
        focal_pair = str(row.get("focal_pair", "") or "").strip()
        if focal_pair:
            focal_pairs[focal_pair] += 1
        slot_pairs = [str(x) for x in (row.get("slot_pairs") or [])] if isinstance(row.get("slot_pairs"), list) else []
        if focal_pair and focal_pair not in slot_pairs:
            focal_inclusion_failures += 1
        slot_ordering = str(row.get("slot_ordering", "") or "").strip()
        if slot_ordering:
            slot_orderings[slot_ordering] += 1
        md = row.get("sampling_metadata") if isinstance(row.get("sampling_metadata"), dict) else {}
        if md:
            exchange_targets[str(md.get("exchange_target", "") or "")] += 1
            data_sources[str(md.get("data_source", "") or "")] += 1
            sampler_versions[str(md.get("sampler_version", "") or "")] += 1
            sampler_policies[str(md.get("sampler_policy", "") or "")] += 1
            ranker_versions[str(md.get("ranker_version", "") or "")] += 1
            ranker_names[str(md.get("ranker_name", "") or "")] += 1

    widths_sorted = sorted(widths)
    max_widths_sorted = sorted(max_widths)
    rows_total = len(trace_rows)
    rows_valid = len(valid)

    summary = {
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "parse_error_rows": len(parse_errors),
        "schema_filtered_out_rows": max(0, rows_total - rows_valid - len(parse_errors)),
        "pair_width_stats": {
            "count": len(widths_sorted),
            "min": min(widths_sorted) if widths_sorted else None,
            "max": max(widths_sorted) if widths_sorted else None,
            "mean": (sum(widths_sorted) / len(widths_sorted)) if widths_sorted else None,
            "p50": _percentile(widths_sorted, 0.5),
            "p95": _percentile(widths_sorted, 0.95),
        },
        "max_pair_width_stats": {
            "count": len(max_widths_sorted),
            "min": min(max_widths_sorted) if max_widths_sorted else None,
            "max": max(max_widths_sorted) if max_widths_sorted else None,
            "mean": (sum(max_widths_sorted) / len(max_widths_sorted)) if max_widths_sorted else None,
        },
        "focal_pair_inclusion_failures": int(focal_inclusion_failures),
    }

    return {
        "schema": "moneyfan.pair_context_sampler_trace_report.v1",
        "generated_at_utc": utc_now_iso(),
        "summary": summary,
        "distributions": {
            "pair_width_histogram": {str(int(k)): int(v) for k, v in sorted(Counter(int(x) for x in widths_sorted).items())},
            "exchange_targets": dict(exchange_targets),
            "data_sources": dict(data_sources),
            "sampler_versions": dict(sampler_versions),
            "sampler_policies": dict(sampler_policies),
            "ranker_names": dict(ranker_names),
            "ranker_versions": dict(ranker_versions),
            "slot_orderings": dict(slot_orderings),
        },
        "top_focal_pairs": [{"pair": k, "count": int(v)} for k, v in focal_pairs.most_common(20)],
        "parse_errors": parse_errors[:50],
    }


def build_markdown_report(report: Dict[str, Any], trace_path: Optional[str] = None) -> str:
    s = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    d = report.get("distributions", {}) if isinstance(report.get("distributions"), dict) else {}
    lines: List[str] = []
    lines.append("# Pair-Context Sampler Trace Report")
    lines.append("")
    lines.append(f"- Generated: `{report.get('generated_at_utc')}`")
    if trace_path:
        lines.append(f"- Trace JSONL: `{trace_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- rows_total={s.get('rows_total', 0)} rows_valid={s.get('rows_valid', 0)} "
        f"parse_errors={s.get('parse_error_rows', 0)} schema_filtered={s.get('schema_filtered_out_rows', 0)}"
    )
    pws = s.get("pair_width_stats", {}) if isinstance(s.get("pair_width_stats"), dict) else {}
    lines.append(
        f"- pair_width: count={pws.get('count')} min={pws.get('min')} max={pws.get('max')} "
        f"mean={pws.get('mean')} p50={pws.get('p50')} p95={pws.get('p95')}"
    )
    lines.append(f"- focal_pair_inclusion_failures={s.get('focal_pair_inclusion_failures', 0)}")
    lines.append("")

    def _render_counter(title: str, counter_obj: Dict[str, Any], limit: int = 20) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not counter_obj:
            lines.append("_No data._")
            lines.append("")
            return
        items = sorted(counter_obj.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:limit]
        lines.append("| value | count |")
        lines.append("|---|---:|")
        for k, v in items:
            lines.append(f"| `{k}` | {int(v)} |")
        lines.append("")

    _render_counter("Pair Width Histogram", d.get("pair_width_histogram", {}) if isinstance(d.get("pair_width_histogram"), dict) else {}, limit=50)
    _render_counter("Exchange Targets", d.get("exchange_targets", {}) if isinstance(d.get("exchange_targets"), dict) else {})
    _render_counter("Data Sources", d.get("data_sources", {}) if isinstance(d.get("data_sources"), dict) else {})
    _render_counter("Sampler Versions", d.get("sampler_versions", {}) if isinstance(d.get("sampler_versions"), dict) else {})
    _render_counter("Sampler Policies", d.get("sampler_policies", {}) if isinstance(d.get("sampler_policies"), dict) else {})
    _render_counter("Ranker Names", d.get("ranker_names", {}) if isinstance(d.get("ranker_names"), dict) else {})
    _render_counter("Ranker Versions", d.get("ranker_versions", {}) if isinstance(d.get("ranker_versions"), dict) else {})

    top_focal = report.get("top_focal_pairs", []) if isinstance(report.get("top_focal_pairs"), list) else []
    lines.append("## Top Focal Pairs")
    lines.append("")
    if not top_focal:
        lines.append("_No data._")
        lines.append("")
    else:
        lines.append("| pair | count |")
        lines.append("|---|---:|")
        for row in top_focal[:20]:
            lines.append(f"| `{row.get('pair')}` | {int(row.get('count', 0) or 0)} |")
        lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build_timestamped_path(base_path: Path, stamp: Optional[str] = None) -> Path:
    ts = str(stamp or _utc_stamp())
    suffix = "".join(base_path.suffixes) or ".md"
    stem = base_path.name[: -len(suffix)] if suffix and base_path.name.endswith(suffix) else base_path.stem
    return base_path.with_name(f"{stem}_{ts}{suffix}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build JSON/markdown audit report from pair-context sampler trace JSONL")
    p.add_argument("--trace-jsonl", type=str, required=True)
    p.add_argument("--out-json", type=str, default="runtime/pair_context_sampler_trace_report.json")
    p.add_argument("--out-md", type=str, default="runtime/pair_context_sampler_trace_report.md")
    p.add_argument("--also-write-timestamped", action="store_true")
    p.add_argument("--timestamp-stamp", type=str, default="")
    p.add_argument("--print-summary", action="store_true")
    args = p.parse_args()

    trace_path = Path(args.trace_jsonl)
    report = build_pair_context_sampler_trace_report(read_jsonl(trace_path))
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, report)
    md = build_markdown_report(report, trace_path=str(trace_path))
    write_text(out_md, md)
    ts_path = None
    if bool(args.also_write_timestamped):
        ts_path = build_timestamped_path(out_md, stamp=(str(args.timestamp_stamp).strip() or None))
        write_text(ts_path, md)

    if bool(args.print_summary):
        s = report["summary"]
        print(
            "✅ Pair-context sampler trace report "
            f"rows={s['rows_valid']}/{s['rows_total']} "
            f"pair_width_p95={s['pair_width_stats']['p95']} "
            f"focal_inclusion_failures={s['focal_pair_inclusion_failures']}"
        )
        print(f"📝 {out_json}")
        print(f"📄 {out_md}")
        if ts_path is not None:
            print(f"🗂️  {ts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
