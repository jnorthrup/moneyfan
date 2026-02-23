#!/usr/bin/env python3
"""OOS calibration governor for HRM trade-head magnitude calibration.

Purpose (money-first):
- Fit multiple calibration candidates on a training subset of walk-forward trades.
- Score them on held-out trades (out-of-sample by file split).
- Promote only if validation error improves vs the current calibration artifact.

This is an "agent" component: it applies a concrete promotion policy rather than
just producing a one-off fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hrm.trade_head_calibration import (
    TradeHeadCalibrator,
    discover_trade_head_calibration_path,
    fit_trade_head_calibration_from_trade_rows,
)


TS_RE = re.compile(r"(20\d{6}_\d{6})")


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


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in _parse_csv(raw)]


def _parse_int_list(raw: str) -> List[int]:
    return [int(float(x)) for x in _parse_csv(raw)]


def _parse_edge_grid(raw: str) -> List[List[float]]:
    specs = [x.strip() for x in (raw or "").split(";") if x.strip()]
    if not specs:
        return [[0.0, 0.35, 0.50, 0.65, 0.80, 1.01]]
    out: List[List[float]] = []
    for spec in specs:
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) < 2:
            continue
        out.append(vals)
    return out or [[0.0, 0.35, 0.50, 0.65, 0.80, 1.01]]


def _discover_trade_files(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for token in inputs:
        if any(ch in token for ch in "*?[]"):
            for p in glob.glob(token, recursive=True):
                pp = Path(p)
                if pp.is_file() and pp.name == "trades.json":
                    files.append(pp)
            continue
        p = Path(token)
        if p.is_dir():
            files.extend(sorted(p.rglob("trades.json")))
        elif p.is_file() and p.name == "trades.json":
            files.append(p)

    seen: set[str] = set()
    out: List[Path] = []
    for p in files:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _load_trade_rows(files: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            r["_source_trades_path"] = str(p)
            rows.append(r)
    return rows


def _sort_key_for_file(p: Path) -> Tuple[int, str]:
    s = str(p)
    m = TS_RE.search(s)
    if m:
        return (0, m.group(1))
    try:
        return (1, str(int(p.stat().st_mtime)))
    except Exception:
        return (2, s)


def _split_files_auto(files: List[Path], val_fraction: float, min_val_files: int) -> Tuple[List[Path], List[Path]]:
    ordered = sorted(files, key=_sort_key_for_file)
    n = len(ordered)
    if n < 2:
        return ordered, []
    vf = max(0.05, min(0.8, float(val_fraction)))
    val_n = max(int(min_val_files), int(round(n * vf)))
    val_n = min(max(val_n, 1), n - 1)
    train = ordered[:-val_n]
    val = ordered[-val_n:]
    return train, val


def _realized_move_bps_from_trade_row(row: Dict[str, Any]) -> Optional[float]:
    pred_move_bps = _safe_float(row.get("predicted_move_bps"), 0.0)
    exposure = _safe_float(row.get("exposure"), 0.0)
    gross_ret = _safe_float(row.get("gross_ret"), 0.0)
    if pred_move_bps <= 0.0 or exposure <= 0.0:
        return None
    realized_signed_move = gross_ret / max(exposure, 1e-12)
    realized_move_bps = abs(realized_signed_move) * 10000.0
    if not math.isfinite(realized_move_bps) or realized_move_bps < 0.0:
        return None
    return float(realized_move_bps)


def _eval_rows(rows: Iterable[Dict[str, Any]], calibrator: Optional[TradeHeadCalibrator]) -> Dict[str, Any]:
    n_rows = 0
    n_used = 0
    sum_w = 0.0
    sum_abs_err = 0.0
    sum_sq_err = 0.0
    sum_abs_pct_err = 0.0
    sum_abs_ratio_err = 0.0
    sum_raw_abs_err = 0.0
    sum_raw_sq_err = 0.0
    sum_raw_abs_pct_err = 0.0
    sum_raw_abs_ratio_err = 0.0
    ratio_values_raw: List[float] = []
    ratio_values_cal: List[float] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        n_rows += 1
        pred_move_bps = _safe_float(row.get("predicted_move_bps"), 0.0)
        conf = max(0.0, min(1.0, _safe_float(row.get("confidence"), 0.0)))
        exposure = _safe_float(row.get("exposure"), 0.0)
        realized_move_bps = _realized_move_bps_from_trade_row(row)
        if realized_move_bps is None or pred_move_bps <= 0.0:
            continue
        weight = max(exposure, 1e-6)
        cal_pred = float(
            calibrator.calibrate_move_bps(pred_move_bps, conf) if calibrator is not None else pred_move_bps
        )
        cal_pred = max(cal_pred, 1e-9)
        raw_pred = max(pred_move_bps, 1e-9)

        raw_abs_err = abs(raw_pred - realized_move_bps)
        cal_abs_err = abs(cal_pred - realized_move_bps)
        raw_sq_err = (raw_pred - realized_move_bps) ** 2
        cal_sq_err = (cal_pred - realized_move_bps) ** 2
        denom_real = max(realized_move_bps, 1e-6)
        raw_abs_pct_err = raw_abs_err / denom_real
        cal_abs_pct_err = cal_abs_err / denom_real
        raw_ratio_err = abs((realized_move_bps / raw_pred) - 1.0)
        cal_ratio_err = abs((realized_move_bps / cal_pred) - 1.0)

        n_used += 1
        sum_w += weight
        sum_raw_abs_err += weight * raw_abs_err
        sum_abs_err += weight * cal_abs_err
        sum_raw_sq_err += weight * raw_sq_err
        sum_sq_err += weight * cal_sq_err
        sum_raw_abs_pct_err += weight * raw_abs_pct_err
        sum_abs_pct_err += weight * cal_abs_pct_err
        sum_raw_abs_ratio_err += weight * raw_ratio_err
        sum_abs_ratio_err += weight * cal_ratio_err
        ratio_values_raw.append(float(realized_move_bps / raw_pred))
        ratio_values_cal.append(float(realized_move_bps / cal_pred))

    def _wavg(x: float) -> float:
        return float(x / max(sum_w, 1e-12))

    def _median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        i = len(s) // 2
        if len(s) % 2 == 1:
            return float(s[i])
        return float((s[i - 1] + s[i]) / 2.0)

    raw_wmae = _wavg(sum_raw_abs_err)
    cal_wmae = _wavg(sum_abs_err)
    raw_wrmse = math.sqrt(max(_wavg(sum_raw_sq_err), 0.0))
    cal_wrmse = math.sqrt(max(_wavg(sum_sq_err), 0.0))
    raw_wmape = _wavg(sum_raw_abs_pct_err)
    cal_wmape = _wavg(sum_abs_pct_err)
    raw_wratioe = _wavg(sum_raw_abs_ratio_err)
    cal_wratioe = _wavg(sum_abs_ratio_err)

    return {
        "rows_seen": int(n_rows),
        "samples_used": int(n_used),
        "weight_sum": float(sum_w),
        "raw": {
            "weighted_mae_bps": float(raw_wmae),
            "weighted_rmse_bps": float(raw_wrmse),
            "weighted_mape": float(raw_wmape),
            "weighted_abs_ratio_error": float(raw_wratioe),
            "ratio_median": float(_median(ratio_values_raw)),
        },
        "calibrated": {
            "weighted_mae_bps": float(cal_wmae),
            "weighted_rmse_bps": float(cal_wrmse),
            "weighted_mape": float(cal_wmape),
            "weighted_abs_ratio_error": float(cal_wratioe),
            "ratio_median": float(_median(ratio_values_cal)),
        },
        "improvement": {
            "weighted_mae_bps": float(raw_wmae - cal_wmae),
            "weighted_rmse_bps": float(raw_wrmse - cal_wrmse),
            "weighted_mape": float(raw_wmape - cal_wmape),
            "weighted_abs_ratio_error": float(raw_wratioe - cal_wratioe),
        },
    }


@dataclass(frozen=True)
class CandidateSpec:
    min_scale: float
    min_bin_count: int
    confidence_bin_edges: Tuple[float, ...]

    def key(self) -> str:
        edges_tag = "-".join(f"{x:.2f}".rstrip("0").rstrip(".") for x in self.confidence_bin_edges)
        return f"ms{self.min_scale:.3f}_mb{self.min_bin_count}_edges[{edges_tag}]"


def _candidate_specs(args: argparse.Namespace) -> List[CandidateSpec]:
    min_scales = _parse_float_list(args.min_scale_grid) or [float(args.min_scale)]
    min_bin_counts = _parse_int_list(args.min_bin_count_grid) or [int(args.min_bin_count)]
    edge_grids = _parse_edge_grid(args.confidence_bin_edges_grid)
    specs: List[CandidateSpec] = []
    seen = set()
    for ms in min_scales:
        for mbc in min_bin_counts:
            for edges in edge_grids:
                spec = CandidateSpec(
                    min_scale=float(ms),
                    min_bin_count=int(mbc),
                    confidence_bin_edges=tuple(float(x) for x in edges),
                )
                key = spec.key()
                if key in seen:
                    continue
                seen.add(key)
                specs.append(spec)
    return specs


def _load_baseline_calibrator(path: Optional[str]) -> Tuple[Optional[TradeHeadCalibrator], Optional[Path], Optional[str]]:
    p = discover_trade_head_calibration_path(path)
    if not p:
        return None, None, None
    try:
        return TradeHeadCalibrator.load(p), p, None
    except Exception as e:
        return None, p, str(e)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
    return obj


def main():
    p = argparse.ArgumentParser(description="OOS governor for HRM trade-head calibration")
    p.add_argument(
        "--inputs",
        type=str,
        default="walkforward_results,walkforward_sweeps,simmer_runs,simmer_runs_multislice_smoke",
        help="Comma-separated files/dirs/globs to scan for trades.json (used if train/val inputs not provided)",
    )
    p.add_argument("--train-inputs", type=str, default="", help="Explicit training trade inputs (overrides auto split)")
    p.add_argument("--val-inputs", type=str, default="", help="Explicit validation trade inputs (overrides auto split)")
    p.add_argument("--val-fraction", type=float, default=0.30, help="Auto split: fraction of newest files for validation")
    p.add_argument("--min-val-files", type=int, default=3, help="Auto split: minimum validation trades files")
    p.add_argument(
        "--out",
        type=str,
        default="models/trained/hrm_trade_head_calibration.json",
        help="Promotion target path for calibration artifact",
    )
    p.add_argument("--report-dir", type=str, default="", help="Report output dir (default reports/calibration_governor/<ts>)")
    p.add_argument("--current-calibration", type=str, default="", help="Explicit current calibration path (otherwise auto-discover)")
    p.add_argument("--min-scale", type=float, default=0.05)
    p.add_argument("--max-scale", type=float, default=2.0)
    p.add_argument("--min-bin-count", type=int, default=30)
    p.add_argument(
        "--confidence-bin-edges",
        type=str,
        default="0.0,0.35,0.50,0.65,0.80,1.01",
        help="Fallback single candidate edges (used if grid not provided)",
    )
    p.add_argument("--min-scale-grid", type=str, default="0.03,0.05,0.08")
    p.add_argument("--min-bin-count-grid", type=str, default="20,30,50")
    p.add_argument(
        "--confidence-bin-edges-grid",
        type=str,
        default="0.0,0.35,0.50,0.65,0.80,1.01;0.0,0.40,0.60,0.80,1.01;0.0,0.50,0.70,1.01",
        help="Semicolon-separated edge sets; each set is comma-separated floats",
    )
    p.add_argument("--promote", action="store_true", help="Write best candidate to --out if it passes gate")
    p.add_argument("--min-improvement-bps", type=float, default=0.25, help="Required OOS weighted MAE improvement vs baseline")
    p.add_argument("--min-val-samples", type=int, default=50, help="Minimum held-out samples to permit promotion")
    p.add_argument("--max-candidates", type=int, default=0, help="Optional cap on candidate evaluations (0 = all)")
    args = p.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else Path("reports") / "calibration_governor" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.train_inputs or args.val_inputs:
        train_files = _discover_trade_files(_parse_csv(args.train_inputs))
        val_files = _discover_trade_files(_parse_csv(args.val_inputs))
        split_mode = "explicit"
    else:
        all_files = _discover_trade_files(_parse_csv(args.inputs))
        train_files, val_files = _split_files_auto(all_files, float(args.val_fraction), int(args.min_val_files))
        split_mode = "auto_newest_holdout"

    if not train_files:
        raise SystemExit("No training trades files found")
    if not val_files:
        raise SystemExit("No validation trades files found (need at least 2 files or explicit val inputs)")

    train_rows = _load_trade_rows(train_files)
    val_rows = _load_trade_rows(val_files)
    if not train_rows:
        raise SystemExit("No training trade rows loaded")
    if not val_rows:
        raise SystemExit("No validation trade rows loaded")

    # Baseline evaluation (current calibration artifact if available), plus raw/no-cal baseline.
    baseline_cal, baseline_path, baseline_load_error = _load_baseline_calibrator(args.current_calibration or None)
    raw_eval = _eval_rows(val_rows, calibrator=None)
    baseline_eval = _eval_rows(val_rows, calibrator=baseline_cal) if baseline_cal is not None else None

    specs = _candidate_specs(args)
    if int(args.max_candidates) > 0:
        specs = specs[: int(args.max_candidates)]
    if not specs:
        raise SystemExit("No calibration candidates configured")

    candidate_results: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        try:
            payload = fit_trade_head_calibration_from_trade_rows(
                train_rows,
                confidence_bin_edges=list(spec.confidence_bin_edges),
                min_bin_count=int(spec.min_bin_count),
                min_scale=float(spec.min_scale),
                max_scale=float(args.max_scale),
            )
            calibrator = TradeHeadCalibrator(payload)
            eval_metrics = _eval_rows(val_rows, calibrator=calibrator)
            fit_stats = dict(payload.get("fit_stats", {}))
            move_meta = dict(payload.get("move_calibration", {}))
            candidate_results.append(
                {
                    "candidate_index": i,
                    "spec": {
                        "min_scale": float(spec.min_scale),
                        "min_bin_count": int(spec.min_bin_count),
                        "confidence_bin_edges": [float(x) for x in spec.confidence_bin_edges],
                        "max_scale": float(args.max_scale),
                    },
                    "fit_stats": fit_stats,
                    "eval": eval_metrics,
                    "payload": payload,
                    "score": {
                        # Primary: OOS calibrated weighted MAE (lower is better).
                        "weighted_mae_bps": float(eval_metrics["calibrated"]["weighted_mae_bps"]),
                        # Secondary: OOS calibrated weighted RMSE.
                        "weighted_rmse_bps": float(eval_metrics["calibrated"]["weighted_rmse_bps"]),
                        # Tertiary: improvement vs raw.
                        "mae_improvement_bps_vs_raw": float(eval_metrics["improvement"]["weighted_mae_bps"]),
                    },
                }
            )
        except Exception as e:
            candidate_results.append(
                {
                    "candidate_index": i,
                    "spec": {
                        "min_scale": float(spec.min_scale),
                        "min_bin_count": int(spec.min_bin_count),
                        "confidence_bin_edges": [float(x) for x in spec.confidence_bin_edges],
                        "max_scale": float(args.max_scale),
                    },
                    "error": str(e),
                }
            )

    good_candidates = [c for c in candidate_results if "eval" in c]
    ranked_candidates = sorted(
        good_candidates,
        key=lambda c: (
            -float(c["eval"]["improvement"]["weighted_mae_bps"]),
            -float(c["eval"]["improvement"]["weighted_rmse_bps"]),
            float(c["eval"]["calibrated"]["weighted_mae_bps"]),
            float(c["eval"]["calibrated"]["weighted_rmse_bps"]),
        ),
        reverse=False,  # key arranged negative for improvements, but we want best first by improvements then lower errors
    )
    # Simpler explicit sort for readability/correctness.
    ranked_candidates = sorted(
        good_candidates,
        key=lambda c: (
            float(c["eval"]["calibrated"]["weighted_mae_bps"]),
            float(c["eval"]["calibrated"]["weighted_rmse_bps"]),
            -float(c["eval"]["improvement"]["weighted_mae_bps"]),
            -float(c["eval"]["improvement"]["weighted_rmse_bps"]),
        )
    )
    best_candidate = ranked_candidates[0] if ranked_candidates else None

    baseline_wmae = None
    if baseline_eval is not None:
        baseline_wmae = float(baseline_eval["calibrated"]["weighted_mae_bps"])
    else:
        baseline_wmae = float(raw_eval["raw"]["weighted_mae_bps"])

    decision: Dict[str, Any] = {
        "promote_requested": bool(args.promote),
        "promoted": False,
        "reason": None,
        "output_path": str(Path(args.out).resolve()),
        "backup_path": None,
    }

    if best_candidate is None:
        decision["reason"] = "no_valid_candidates"
    else:
        best_eval = best_candidate["eval"]
        best_wmae = float(best_eval["calibrated"]["weighted_mae_bps"])
        best_impr_vs_raw = float(best_eval["improvement"]["weighted_mae_bps"])
        val_samples = int(best_eval.get("samples_used", 0) or 0)
        improvement_vs_baseline = float(baseline_wmae - best_wmae)
        decision["best_candidate_validation_improvement_wmae_bps_vs_baseline"] = improvement_vs_baseline
        decision["best_candidate_validation_improvement_wmae_bps_vs_raw"] = best_impr_vs_raw
        decision["best_candidate_val_samples"] = val_samples

        if val_samples < int(args.min_val_samples):
            decision["reason"] = f"insufficient_val_samples:{val_samples}<{int(args.min_val_samples)}"
        elif improvement_vs_baseline < float(args.min_improvement_bps):
            decision["reason"] = (
                f"improvement_below_threshold:{improvement_vs_baseline:.6f}<{float(args.min_improvement_bps):.6f}"
            )
        elif not bool(args.promote):
            decision["reason"] = "promotion_not_requested"
        else:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                backup = out_path.with_name(out_path.name + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copy2(out_path, backup)
                decision["backup_path"] = str(backup.resolve())
            with open(out_path, "w") as f:
                json.dump(best_candidate["payload"], f, indent=2)
            decision["promoted"] = True
            decision["reason"] = "promoted"

    report = {
        "created_at": datetime.now().isoformat(),
        "purpose": "OOS trade-head calibration governor",
        "split": {
            "mode": split_mode,
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "train_file_count": int(len(train_files)),
            "val_file_count": int(len(val_files)),
            "train_row_count": int(len(train_rows)),
            "val_row_count": int(len(val_rows)),
        },
        "baseline": {
            "path": str(baseline_path.resolve()) if baseline_path else None,
            "load_error": baseline_load_error,
            "raw_eval_on_val": raw_eval,
            "current_calibration_eval_on_val": baseline_eval,
        },
        "search": {
            "min_scale_grid": _parse_float_list(args.min_scale_grid) or [float(args.min_scale)],
            "min_bin_count_grid": _parse_int_list(args.min_bin_count_grid) or [int(args.min_bin_count)],
            "confidence_bin_edges_grid": _parse_edge_grid(args.confidence_bin_edges_grid),
            "max_scale": float(args.max_scale),
            "candidates_attempted": int(len(candidate_results)),
            "candidates_valid": int(len(good_candidates)),
        },
        "candidates_ranked": [
            {
                k: v
                for k, v in c.items()
                if k in {"candidate_index", "spec", "fit_stats", "eval", "score"}
            }
            for c in ranked_candidates
        ],
        "candidate_errors": [
            {
                k: v for k, v in c.items()
                if k in {"candidate_index", "spec", "error"}
            }
            for c in candidate_results if "error" in c
        ],
        "best_candidate": (
            {
                k: v
                for k, v in best_candidate.items()
                if k in {"candidate_index", "spec", "fit_stats", "eval", "score"}
            }
            if best_candidate else None
        ),
        "decision": decision,
    }

    with open(report_dir / "calibration_governor_report.json", "w") as f:
        json.dump(_sanitize(report), f, indent=2)

    print("\nCalibration governor complete")
    print(f"Report dir: {report_dir.resolve()}")
    print(
        f"Split: train_files={len(train_files)} train_rows={len(train_rows)} | "
        f"val_files={len(val_files)} val_rows={len(val_rows)}"
    )
    print(
        "Baseline OOS wMAE(bps): "
        f"raw={float(raw_eval['raw']['weighted_mae_bps']):.4f} "
        + (
            f"| current_cal={float(baseline_eval['calibrated']['weighted_mae_bps']):.4f}"
            if baseline_eval is not None
            else "| current_cal=(none)"
        )
    )
    if best_candidate is not None:
        ev = best_candidate["eval"]
        print(
            "Best candidate OOS: "
            f"wMAE={float(ev['calibrated']['weighted_mae_bps']):.4f} "
            f"wRMSE={float(ev['calibrated']['weighted_rmse_bps']):.4f} "
            f"ΔwMAE_vs_raw={float(ev['improvement']['weighted_mae_bps']):+.4f}"
        )
        print(f"Best candidate spec: {best_candidate['spec']}")
    print(f"Decision: {decision.get('reason')} | promoted={bool(decision.get('promoted', False))}")


if __name__ == "__main__":
    main()
