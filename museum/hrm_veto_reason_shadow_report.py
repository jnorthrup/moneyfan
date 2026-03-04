#!/usr/bin/env python3
"""Aggregate per-reason veto shadow impact across walk-forward runs.

Scans walk-forward `metrics.json` outputs and summarizes which mechanical veto
reasons are actually costly (blocked profitable counterfactual trades) vs
protective (blocked losing trades) across slices/regimes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in {"inf", "+inf", "infinity", "+infinity"}:
                return float("inf")
            if xl in {"-inf", "-infinity"}:
                return float("-inf")
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _coerce_num_map(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in obj.items():
        out[str(k)] = _safe_float(v, 0.0)
    return out


def _coerce_int_map(obj: Any) -> Dict[str, int]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            out[str(k)] = int(_safe_float(v, 0.0))
    return out


def _iter_metric_files(inputs: List[Path], pattern: str) -> Iterable[Path]:
    seen: set[Path] = set()
    for p in inputs:
        if not p.exists():
            continue
        if p.is_file():
            candidates = [p] if p.name == "metrics.json" else []
        else:
            candidates = sorted(p.glob(pattern))
        for c in candidates:
            if c.name != "metrics.json":
                continue
            rc = c.resolve()
            if rc in seen:
                continue
            seen.add(rc)
            yield rc


def _is_walkforward_metrics(metrics: Dict[str, Any]) -> bool:
    # Distinguish from paper-trading metrics in the repo.
    required = ["raw_vetoed_candidates", "raw_veto_reason_counts", "top_k", "signal_threshold"]
    return all(k in metrics for k in required)


def _new_reason_record() -> Dict[str, Any]:
    return {
        "raw_vetoed_candidates": 0,
        "counterfactual_topk_candidates": 0,
        "displaced_shadow_trades": 0,
        "displaced_shadow_pnl": 0.0,  # >0 means veto blocked profit; <0 means veto prevented loss
        "displaced_shadow_gross_profit": 0.0,
        "displaced_shadow_gross_loss": 0.0,
        "runs_with_raw_veto": 0,
        "runs_with_cf_topk": 0,
        "runs_with_displacement": 0,
        "runs_with_positive_shadow_pnl": 0,
        "runs_with_negative_shadow_pnl": 0,
    }


def _calc_profit_factor(gross_profit: float, gross_loss: float) -> Any:
    gp = float(gross_profit)
    gl = float(gross_loss)
    if gl > 0.0:
        return float(gp / gl)
    if gp > 0.0:
        return float("inf")
    return 0.0


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(x) for x in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
    return obj


def _parse_inputs(raw: str) -> List[Path]:
    parts = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not parts:
        parts = ["walkforward_results", "walkforward_sweeps", "simmer_runs", "simmer_runs_multislice_smoke"]
    return [Path(p) for p in parts]


def _run_row(path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metrics_path": str(path),
        "run_dir": str(path.parent),
        "symbols": list(metrics.get("symbols", [])) if isinstance(metrics.get("symbols"), list) else metrics.get("symbols"),
        "use_mechanical_veto": bool(metrics.get("use_mechanical_veto", False)),
        "top_k": int(metrics.get("top_k", 0) or 0),
        "hold_bars": int(metrics.get("hold_bars", 0) or 0),
        "signal_threshold": _safe_float(metrics.get("signal_threshold"), 0.0),
        "final_equity": _safe_float(metrics.get("final_equity"), 0.0),
        "return_pct": _safe_float(metrics.get("return_pct"), 0.0),
        "max_drawdown": _safe_float(metrics.get("max_drawdown"), 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "raw_vetoed_candidates": int(metrics.get("raw_vetoed_candidates", 0) or 0),
        "raw_veto_reason_counts": dict(metrics.get("raw_veto_reason_counts", {})),
        "raw_veto_counterfactual_topk_candidates": int(metrics.get("raw_veto_counterfactual_topk_candidates", 0) or 0),
        "raw_veto_counterfactual_topk_reason_counts": dict(metrics.get("raw_veto_counterfactual_topk_reason_counts", {})),
        "raw_veto_displaced_topk_slots": int(metrics.get("raw_veto_displaced_topk_slots", 0) or 0),
        "raw_veto_displaced_steps": int(metrics.get("raw_veto_displaced_steps", 0) or 0),
        "raw_veto_displaced_shadow_trades": int(metrics.get("raw_veto_displaced_shadow_trades", 0) or 0),
        "raw_veto_displaced_reason_counts": dict(metrics.get("raw_veto_displaced_reason_counts", {})),
        "raw_veto_displaced_shadow_pnl": _safe_float(metrics.get("raw_veto_displaced_shadow_pnl"), 0.0),
        "raw_veto_displaced_shadow_pnl_by_reason": dict(metrics.get("raw_veto_displaced_shadow_pnl_by_reason", {})),
    }


def main():
    p = argparse.ArgumentParser(description="Aggregate veto reason shadow PnL across walk-forward runs")
    p.add_argument(
        "--inputs",
        type=str,
        default="walkforward_results,walkforward_sweeps,simmer_runs,simmer_runs_multislice_smoke",
        help="Comma-separated dirs/files to scan for metrics.json",
    )
    p.add_argument("--pattern", type=str, default="**/metrics.json", help="Glob pattern for metrics files within input dirs")
    p.add_argument("--out-dir", type=str, default="", help="Output dir (default: reports/veto_reason_shadow/<timestamp>)")
    p.add_argument("--top-runs", type=int, default=10, help="How many example runs to include in summary rankings")
    p.add_argument("--include-noveto", action="store_true", help="Include runs with mechanical veto disabled")
    p.add_argument("--require-per-reason-shadow", action="store_true",
                   help="Only use runs that have new per-reason shadow metrics fields")
    args = p.parse_args()

    inputs = _parse_inputs(args.inputs)
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / "veto_reason_shadow" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    loaded = 0
    used = 0
    skipped_non_walkforward = 0
    skipped_noveto = 0
    skipped_missing_per_reason = 0

    aggregate: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "inputs": [str(p) for p in inputs],
        "pattern": str(args.pattern),
        "scan_counts": {},
        "totals": {
            "runs_used": 0,
            "runs_with_raw_veto": 0,
            "runs_with_counterfactual_topk": 0,
            "runs_with_displacement": 0,
            "total_raw_vetoed_candidates": 0,
            "total_raw_veto_counterfactual_topk_candidates": 0,
            "total_raw_veto_displaced_topk_slots": 0,
            "total_raw_veto_displaced_steps": 0,
            "total_raw_veto_displaced_shadow_trades": 0,
            "total_raw_veto_displaced_shadow_pnl": 0.0,
            "total_raw_veto_displaced_shadow_gross_profit": 0.0,
            "total_raw_veto_displaced_shadow_gross_loss": 0.0,
        },
        "reason_summary": {},
        "run_examples": {
            "largest_positive_shadow_pnl_runs": [],
            "largest_negative_shadow_pnl_runs": [],
            "largest_raw_veto_runs": [],
            "largest_displacement_runs": [],
        },
    }

    reason_summary: Dict[str, Dict[str, Any]] = {}
    run_rows: List[Dict[str, Any]] = []

    for metrics_path in _iter_metric_files(inputs, args.pattern):
        scanned += 1
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            loaded += 1
        except Exception:
            continue

        if not isinstance(metrics, dict) or not _is_walkforward_metrics(metrics):
            skipped_non_walkforward += 1
            continue

        if (not args.include_noveto) and (not bool(metrics.get("use_mechanical_veto", False))):
            skipped_noveto += 1
            continue

        has_per_reason_shadow = (
            "raw_veto_displaced_shadow_pnl_by_reason" in metrics
            and "raw_veto_displaced_reason_counts" in metrics
            and "raw_veto_counterfactual_topk_reason_counts" in metrics
        )
        if args.require_per_reason_shadow and (not has_per_reason_shadow):
            skipped_missing_per_reason += 1
            continue

        used += 1
        aggregate["totals"]["runs_used"] = int(aggregate["totals"]["runs_used"]) + 1

        raw_reason_counts = _coerce_int_map(metrics.get("raw_veto_reason_counts"))
        cf_reason_counts = _coerce_int_map(metrics.get("raw_veto_counterfactual_topk_reason_counts"))
        displaced_reason_counts = _coerce_int_map(metrics.get("raw_veto_displaced_reason_counts"))
        pnl_by_reason = _coerce_num_map(metrics.get("raw_veto_displaced_shadow_pnl_by_reason"))
        gp_by_reason = _coerce_num_map(metrics.get("raw_veto_displaced_shadow_gross_profit_by_reason"))
        gl_by_reason = _coerce_num_map(metrics.get("raw_veto_displaced_shadow_gross_loss_by_reason"))

        total_raw_vetoed = int(metrics.get("raw_vetoed_candidates", 0) or 0)
        total_cf = int(metrics.get("raw_veto_counterfactual_topk_candidates", 0) or 0)
        total_disp_slots = int(metrics.get("raw_veto_displaced_topk_slots", 0) or 0)
        total_disp_steps = int(metrics.get("raw_veto_displaced_steps", 0) or 0)
        total_disp_trades = int(metrics.get("raw_veto_displaced_shadow_trades", 0) or 0)
        total_shadow_pnl = _safe_float(metrics.get("raw_veto_displaced_shadow_pnl"), 0.0)
        total_shadow_gp = _safe_float(metrics.get("raw_veto_displaced_shadow_gross_profit"), 0.0)
        total_shadow_gl = _safe_float(metrics.get("raw_veto_displaced_shadow_gross_loss"), 0.0)

        t = aggregate["totals"]
        t["total_raw_vetoed_candidates"] = int(t["total_raw_vetoed_candidates"]) + total_raw_vetoed
        t["total_raw_veto_counterfactual_topk_candidates"] = int(t["total_raw_veto_counterfactual_topk_candidates"]) + total_cf
        t["total_raw_veto_displaced_topk_slots"] = int(t["total_raw_veto_displaced_topk_slots"]) + total_disp_slots
        t["total_raw_veto_displaced_steps"] = int(t["total_raw_veto_displaced_steps"]) + total_disp_steps
        t["total_raw_veto_displaced_shadow_trades"] = int(t["total_raw_veto_displaced_shadow_trades"]) + total_disp_trades
        t["total_raw_veto_displaced_shadow_pnl"] = float(t["total_raw_veto_displaced_shadow_pnl"]) + total_shadow_pnl
        t["total_raw_veto_displaced_shadow_gross_profit"] = float(t["total_raw_veto_displaced_shadow_gross_profit"]) + total_shadow_gp
        t["total_raw_veto_displaced_shadow_gross_loss"] = float(t["total_raw_veto_displaced_shadow_gross_loss"]) + total_shadow_gl

        if total_raw_vetoed > 0:
            t["runs_with_raw_veto"] = int(t["runs_with_raw_veto"]) + 1
        if total_cf > 0:
            t["runs_with_counterfactual_topk"] = int(t["runs_with_counterfactual_topk"]) + 1
        if total_disp_trades > 0 or abs(total_shadow_pnl) > 0:
            t["runs_with_displacement"] = int(t["runs_with_displacement"]) + 1

        reasons = sorted(set(raw_reason_counts) | set(cf_reason_counts) | set(displaced_reason_counts) | set(pnl_by_reason))
        for reason in reasons:
            rec = reason_summary.setdefault(reason, _new_reason_record())
            raw_n = int(raw_reason_counts.get(reason, 0))
            cf_n = int(cf_reason_counts.get(reason, 0))
            disp_n = int(displaced_reason_counts.get(reason, 0))
            pnl = float(pnl_by_reason.get(reason, 0.0))
            gp = float(gp_by_reason.get(reason, 0.0))
            gl = float(gl_by_reason.get(reason, 0.0))

            rec["raw_vetoed_candidates"] = int(rec["raw_vetoed_candidates"]) + raw_n
            rec["counterfactual_topk_candidates"] = int(rec["counterfactual_topk_candidates"]) + cf_n
            rec["displaced_shadow_trades"] = int(rec["displaced_shadow_trades"]) + disp_n
            rec["displaced_shadow_pnl"] = float(rec["displaced_shadow_pnl"]) + pnl
            rec["displaced_shadow_gross_profit"] = float(rec["displaced_shadow_gross_profit"]) + gp
            rec["displaced_shadow_gross_loss"] = float(rec["displaced_shadow_gross_loss"]) + gl
            if raw_n > 0:
                rec["runs_with_raw_veto"] = int(rec["runs_with_raw_veto"]) + 1
            if cf_n > 0:
                rec["runs_with_cf_topk"] = int(rec["runs_with_cf_topk"]) + 1
            if disp_n > 0:
                rec["runs_with_displacement"] = int(rec["runs_with_displacement"]) + 1
            if pnl > 0:
                rec["runs_with_positive_shadow_pnl"] = int(rec["runs_with_positive_shadow_pnl"]) + 1
            elif pnl < 0:
                rec["runs_with_negative_shadow_pnl"] = int(rec["runs_with_negative_shadow_pnl"]) + 1

        run_rows.append(_run_row(metrics_path, metrics))

    for reason, rec in reason_summary.items():
        rec["displaced_shadow_profit_factor"] = _calc_profit_factor(
            _safe_float(rec.get("displaced_shadow_gross_profit"), 0.0),
            _safe_float(rec.get("displaced_shadow_gross_loss"), 0.0),
        )
        # Positive => veto may be hurting (blocked profits). Negative => veto helping (blocked losses).
        rec["veto_effect_bias"] = (
            "harmful_if_high"
            if _safe_float(rec.get("displaced_shadow_pnl"), 0.0) > 0.0
            else ("protective_if_low" if _safe_float(rec.get("displaced_shadow_pnl"), 0.0) < 0.0 else "neutral_or_unobserved")
        )

    total_gp = float(aggregate["totals"]["total_raw_veto_displaced_shadow_gross_profit"])
    total_gl = float(aggregate["totals"]["total_raw_veto_displaced_shadow_gross_loss"])
    aggregate["totals"]["total_raw_veto_displaced_shadow_profit_factor"] = _calc_profit_factor(total_gp, total_gl)

    aggregate["scan_counts"] = {
        "metrics_files_scanned": scanned,
        "metrics_files_loaded": loaded,
        "walkforward_runs_used": used,
        "skipped_non_walkforward_or_incompatible": skipped_non_walkforward,
        "skipped_noveto_runs": skipped_noveto,
        "skipped_missing_per_reason_fields": skipped_missing_per_reason,
    }

    # Ranked reason views.
    ranked_reasons = []
    for reason, rec in reason_summary.items():
        ranked_reasons.append({"reason": reason, **rec})
    ranked_harmful = sorted(ranked_reasons, key=lambda r: float(r.get("displaced_shadow_pnl", 0.0)), reverse=True)
    ranked_protective = sorted(ranked_reasons, key=lambda r: float(r.get("displaced_shadow_pnl", 0.0)))
    ranked_by_raw_count = sorted(ranked_reasons, key=lambda r: int(r.get("raw_vetoed_candidates", 0)), reverse=True)
    ranked_by_cf = sorted(ranked_reasons, key=lambda r: int(r.get("counterfactual_topk_candidates", 0)), reverse=True)
    ranked_by_disp = sorted(ranked_reasons, key=lambda r: int(r.get("displaced_shadow_trades", 0)), reverse=True)

    aggregate["reason_summary"] = {row["reason"]: {k: v for k, v in row.items() if k != "reason"} for row in ranked_reasons}
    aggregate["reason_rankings"] = {
        "by_shadow_pnl_harmful_first": ranked_harmful,
        "by_shadow_pnl_protective_first": ranked_protective,
        "by_raw_veto_count": ranked_by_raw_count,
        "by_counterfactual_topk_count": ranked_by_cf,
        "by_displaced_shadow_trades": ranked_by_disp,
    }

    top_n = max(1, int(args.top_runs))
    aggregate["run_examples"]["largest_positive_shadow_pnl_runs"] = sorted(
        run_rows, key=lambda r: float(r.get("raw_veto_displaced_shadow_pnl", 0.0)), reverse=True
    )[:top_n]
    aggregate["run_examples"]["largest_negative_shadow_pnl_runs"] = sorted(
        run_rows, key=lambda r: float(r.get("raw_veto_displaced_shadow_pnl", 0.0))
    )[:top_n]
    aggregate["run_examples"]["largest_raw_veto_runs"] = sorted(
        run_rows, key=lambda r: int(r.get("raw_vetoed_candidates", 0)), reverse=True
    )[:top_n]
    aggregate["run_examples"]["largest_displacement_runs"] = sorted(
        run_rows, key=lambda r: int(r.get("raw_veto_displaced_shadow_trades", 0)), reverse=True
    )[:top_n]

    with open(out_dir / "veto_reason_shadow_report.json", "w") as f:
        json.dump(_sanitize_json(aggregate), f, indent=2)

    # Lightweight CSV for reason rankings (easy to inspect quickly).
    csv_lines = [
        "reason,raw_vetoed_candidates,counterfactual_topk_candidates,displaced_shadow_trades,displaced_shadow_pnl,displaced_shadow_gross_profit,displaced_shadow_gross_loss,displaced_shadow_profit_factor,runs_with_raw_veto,runs_with_cf_topk,runs_with_displacement"
    ]
    for row in ranked_harmful:
        csv_lines.append(
            ",".join(
                [
                    json.dumps(str(row.get("reason", ""))),
                    str(int(row.get("raw_vetoed_candidates", 0) or 0)),
                    str(int(row.get("counterfactual_topk_candidates", 0) or 0)),
                    str(int(row.get("displaced_shadow_trades", 0) or 0)),
                    f"{float(row.get('displaced_shadow_pnl', 0.0) or 0.0):.10f}",
                    f"{float(row.get('displaced_shadow_gross_profit', 0.0) or 0.0):.10f}",
                    f"{float(row.get('displaced_shadow_gross_loss', 0.0) or 0.0):.10f}",
                    str(row.get("displaced_shadow_profit_factor", 0.0)),
                    str(int(row.get("runs_with_raw_veto", 0) or 0)),
                    str(int(row.get("runs_with_cf_topk", 0) or 0)),
                    str(int(row.get("runs_with_displacement", 0) or 0)),
                ]
            )
        )
    with open(out_dir / "veto_reason_shadow_reason_summary.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("\nVeto reason shadow report complete")
    print(f"Output dir: {out_dir.resolve()}")
    print(
        "Runs used: "
        f"{aggregate['scan_counts']['walkforward_runs_used']} "
        f"(scanned {aggregate['scan_counts']['metrics_files_scanned']}, "
        f"loaded {aggregate['scan_counts']['metrics_files_loaded']})"
    )
    print(
        "Totals: "
        f"RawVetoed={aggregate['totals']['total_raw_vetoed_candidates']} "
        f"CFTopK={aggregate['totals']['total_raw_veto_counterfactual_topk_candidates']} "
        f"DispTrades={aggregate['totals']['total_raw_veto_displaced_shadow_trades']} "
        f"ShadowPnL=${aggregate['totals']['total_raw_veto_displaced_shadow_pnl']:.2f}"
    )
    if ranked_by_raw_count:
        print("Top veto reasons by raw count:")
        for row in ranked_by_raw_count[: min(5, len(ranked_by_raw_count))]:
            print(
                f"  - {row['reason']}: raw={int(row.get('raw_vetoed_candidates', 0))} "
                f"cf={int(row.get('counterfactual_topk_candidates', 0))} "
                f"disp={int(row.get('displaced_shadow_trades', 0))} "
                f"shadow_pnl={float(row.get('displaced_shadow_pnl', 0.0)):+.4f}"
            )


if __name__ == "__main__":
    main()
