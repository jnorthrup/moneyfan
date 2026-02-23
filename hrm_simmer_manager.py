#!/usr/bin/env python3
"""
Background HRM simmer loop: low-rate training + walk-forward validation + gated promotion.

This keeps "constant learning" on the back burner:
- runs small stochastic training cycles (with low-rate trade-head updates)
- evaluates the resulting checkpoint on a fixed walk-forward slice
- restores the prior checkpoint if the new one fails validation gates
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hrm.regime_validation_manifest import (
    DEFAULT_REGIME_VALIDATION_MANIFEST_PATH,
    load_regime_validation_manifest,
    manifest_to_validation_profiles,
    summarize_validation_manifest_profiles,
)
from train import EpisodeTrainingConfig, EpochEpisodeTrainer


ARTIFACT_BASENAMES = [
    "hrm_latest_weights.npz",
    "hrm_latest_model_config.json",
    "hrm_latest_feature_schema.json",
]
ARTIFACT_DIR = Path("models/trained")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _artifact_paths() -> List[Path]:
    return [ARTIFACT_DIR / name for name in ARTIFACT_BASENAMES]


def _backup_artifacts(dst_dir: Path) -> Dict[str, Optional[str]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, Optional[str]] = {}
    for src in _artifact_paths():
        if src.exists():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied[src.name] = str(dst.resolve())
        else:
            copied[src.name] = None
    return copied


def _restore_artifacts(src_dir: Path):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_BASENAMES:
        src = src_dir / name
        dst = ARTIFACT_DIR / name
        if src.exists():
            shutil.copy2(src, dst)


def _base_validation_profile(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "name": "primary",
        "regime": "default",
        "tags": ["default"],
        "weight": 1.0,
        "mandatory": True,
        "symbols": str(args.validation_symbols),
        "seq_len": int(args.validation_seq_len),
        "min_history": int(args.validation_seq_len),
        "top_k": int(args.validation_top_k),
        "hold_bars": int(args.validation_hold_bars),
        "cooldown_bars": int(args.validation_cooldown_bars),
        "signal_threshold": float(args.validation_signal_threshold),
        "commission_bps": float(args.validation_commission_bps),
        "slippage_bps": float(args.validation_slippage_bps),
        "trade_head_calibration": str(args.validation_trade_head_calibration or ""),
        "no_trade_head_calibration": bool(args.validation_no_trade_head_calibration),
        "no_risk_head_repair": bool(args.validation_no_risk_head_repair),
        "repair_min_stop_loss_pct": float(args.validation_repair_min_stop_loss_pct),
        "repair_max_stop_loss_pct": float(args.validation_repair_max_stop_loss_pct),
        "repair_min_take_profit_pct": float(args.validation_repair_min_take_profit_pct),
        "max_bars": int(args.validation_max_bars),
        "max_steps": int(args.validation_max_steps),
        "start": (str(args.validation_start).strip() if getattr(args, "validation_start", "") else ""),
        "end": (str(args.validation_end).strip() if getattr(args, "validation_end", "") else ""),
        "no_mechanical_veto": bool(args.validation_no_veto),
        "veto_confidence_override_threshold": float(args.validation_veto_confidence_override_threshold),
        "veto_confidence_override_size_scale": float(args.validation_veto_confidence_override_size_scale),
        "veto_confidence_override_reasons": str(args.validation_veto_confidence_override_reasons),
        "carry_memory": bool(args.validation_carry_memory),
    }


def _validation_slice_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base = _base_validation_profile(args)
    use_regime_manifest = bool(getattr(args, "validation_use_regime_manifest", True))
    slices_json = str(getattr(args, "validation_slices_json", "") or "").strip()
    if not slices_json:
        if use_regime_manifest:
            explicit_manifest = str(getattr(args, "validation_regime_manifest", "") or "").strip()
            manifest_path: Optional[Path] = None
            if explicit_manifest:
                manifest_path = Path(explicit_manifest)
            elif DEFAULT_REGIME_VALIDATION_MANIFEST_PATH.exists():
                manifest_path = DEFAULT_REGIME_VALIDATION_MANIFEST_PATH
            if manifest_path is not None and manifest_path.exists():
                manifest = load_regime_validation_manifest(manifest_path)
                return manifest_to_validation_profiles(manifest, base, source_path=str(manifest_path))
        return [base]

    p = Path(slices_json)
    with open(p, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"validation_slices_json must be a non-empty list: {p}")

    specs: List[Dict[str, Any]] = []
    for i, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"validation slice #{i} is not an object")
        spec = dict(base)
        for k, v in row.items():
            spec[k] = v
        spec["name"] = str(spec.get("name") or f"slice_{i:02d}")
        spec["regime"] = str(spec.get("regime") or spec["name"])
        tags = spec.get("tags", base.get("tags", []))
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.split(",") if x.strip()]
        elif not isinstance(tags, list):
            tags = list(base.get("tags", []))
        spec["tags"] = [str(x) for x in tags]
        spec["weight"] = float(spec.get("weight", base.get("weight", 1.0)))
        if not math.isfinite(spec["weight"]) or spec["weight"] < 0:
            spec["weight"] = 0.0
        spec["mandatory"] = bool(spec.get("mandatory", base.get("mandatory", False)))
        spec["symbols"] = str(spec.get("symbols") or base["symbols"])
        spec["seq_len"] = int(spec.get("seq_len", base["seq_len"]))
        spec["min_history"] = int(spec.get("min_history", spec["seq_len"]))
        spec["top_k"] = int(spec.get("top_k", base["top_k"]))
        spec["hold_bars"] = int(spec.get("hold_bars", base["hold_bars"]))
        spec["cooldown_bars"] = int(spec.get("cooldown_bars", base["cooldown_bars"]))
        spec["signal_threshold"] = float(spec.get("signal_threshold", base["signal_threshold"]))
        spec["commission_bps"] = float(spec.get("commission_bps", base["commission_bps"]))
        spec["slippage_bps"] = float(spec.get("slippage_bps", base["slippage_bps"]))
        spec["max_bars"] = int(spec.get("max_bars", base["max_bars"]))
        spec["max_steps"] = int(spec.get("max_steps", base["max_steps"]))
        spec["trade_head_calibration"] = str(spec.get("trade_head_calibration") or "")
        spec["no_trade_head_calibration"] = bool(spec.get("no_trade_head_calibration", base["no_trade_head_calibration"]))
        spec["no_risk_head_repair"] = bool(spec.get("no_risk_head_repair", base["no_risk_head_repair"]))
        spec["repair_min_stop_loss_pct"] = float(spec.get("repair_min_stop_loss_pct", base["repair_min_stop_loss_pct"]))
        spec["repair_max_stop_loss_pct"] = float(spec.get("repair_max_stop_loss_pct", base["repair_max_stop_loss_pct"]))
        spec["repair_min_take_profit_pct"] = float(spec.get("repair_min_take_profit_pct", base["repair_min_take_profit_pct"]))
        spec["start"] = str(spec.get("start") or "")
        spec["end"] = str(spec.get("end") or "")
        spec["no_mechanical_veto"] = bool(spec.get("no_mechanical_veto", base["no_mechanical_veto"]))
        spec["veto_confidence_override_threshold"] = float(
            spec.get("veto_confidence_override_threshold", base["veto_confidence_override_threshold"])
        )
        spec["veto_confidence_override_size_scale"] = float(
            spec.get("veto_confidence_override_size_scale", base["veto_confidence_override_size_scale"])
        )
        spec["veto_confidence_override_reasons"] = str(
            spec.get("veto_confidence_override_reasons", base["veto_confidence_override_reasons"])
        )
        spec["carry_memory"] = bool(spec.get("carry_memory", base["carry_memory"]))
        specs.append(spec)
    return specs


def _run_walkforward(args: argparse.Namespace, out_dir: Path, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = dict(_base_validation_profile(args) if profile is None else profile)
    cmd = [
        sys.executable or "python3",
        "hrm_walkforward_backtest.py",
        "--symbols", str(profile["symbols"]),
        "--seq-len", str(int(profile["seq_len"])),
        "--min-history", str(int(profile.get("min_history", profile["seq_len"]))),
        "--top-k", str(int(profile["top_k"])),
        "--hold-bars", str(int(profile["hold_bars"])),
        "--cooldown-bars", str(int(profile["cooldown_bars"])),
        "--signal-threshold", str(float(profile["signal_threshold"])),
        "--commission-bps", str(float(profile["commission_bps"])),
        "--slippage-bps", str(float(profile["slippage_bps"])),
        "--repair-min-stop-loss-pct", str(float(profile["repair_min_stop_loss_pct"])),
        "--repair-max-stop-loss-pct", str(float(profile["repair_max_stop_loss_pct"])),
        "--repair-min-take-profit-pct", str(float(profile["repair_min_take_profit_pct"])),
        "--max-bars", str(int(profile["max_bars"])),
        "--max-steps", str(int(profile["max_steps"])),
        "--out-dir", str(out_dir),
    ]
    if profile.get("start"):
        cmd += ["--start", str(profile["start"])]
    if profile.get("end"):
        cmd += ["--end", str(profile["end"])]
    if str(profile.get("trade_head_calibration") or ""):
        cmd += ["--trade-head-calibration", str(profile["trade_head_calibration"])]
    if bool(profile.get("no_trade_head_calibration", False)):
        cmd.append("--no-trade-head-calibration")
    if bool(profile.get("no_risk_head_repair", False)):
        cmd.append("--no-risk-head-repair")
    if float(profile.get("veto_confidence_override_threshold", -1.0)) > 0.0:
        cmd += [
            "--veto-confidence-override-threshold", str(float(profile["veto_confidence_override_threshold"])),
            "--veto-confidence-override-size-scale", str(float(profile["veto_confidence_override_size_scale"])),
            "--veto-confidence-override-reasons", str(profile["veto_confidence_override_reasons"]),
        ]
    if bool(profile.get("no_mechanical_veto", False)):
        cmd.append("--no-mechanical-veto")
    if bool(profile.get("carry_memory", False)):
        cmd.append("--carry-memory")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = {
        "profile": profile,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_dir": str(out_dir.resolve()),
    }
    metrics_path = out_dir / "metrics.json"
    if proc.returncode == 0 and metrics_path.exists():
        with open(metrics_path, "r") as f:
            result["metrics"] = json.load(f)
    return result


def _promotion_gate_components(
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    c = candidate_metrics or {}
    b = baseline_metrics or {}
    c_eq = _safe_float(c.get("final_equity"), 0.0)
    b_eq = _safe_float(b.get("final_equity"), 0.0)
    c_dd = _safe_float(c.get("max_drawdown"), 0.0)
    c_trades = int(c.get("total_trades", 0) or 0)
    c_pf = c.get("profit_factor", 0.0)
    if isinstance(c_pf, str):
        c_pf_val = float("inf") if c_pf.lower() == "infinity" else 0.0
    else:
        c_pf_val = _safe_float(c_pf, 0.0)

    min_final_equity = b_eq + float(args.min_equity_improvement)
    pass_eq = c_eq >= min_final_equity
    pass_dd = c_dd >= float(args.max_drawdown_floor)
    pass_trades = c_trades >= int(args.min_trades)
    pass_pf = c_pf_val >= float(args.min_profit_factor)
    return {
        "baseline_final_equity": b_eq,
        "candidate_final_equity": c_eq,
        "equity_delta": float(c_eq - b_eq),
        "min_final_equity_required": min_final_equity,
        "candidate_max_drawdown": c_dd,
        "max_drawdown_floor": float(args.max_drawdown_floor),
        "candidate_total_trades": c_trades,
        "min_trades": int(args.min_trades),
        "candidate_profit_factor": c_pf_val,
        "min_profit_factor": float(args.min_profit_factor),
        "pass_eq": pass_eq,
        "pass_dd": pass_dd,
        "pass_trades": pass_trades,
        "pass_pf": pass_pf,
    }


def _passes_promotion_gate(candidate: Dict[str, Any], baseline: Dict[str, Any], args: argparse.Namespace) -> tuple[bool, Dict[str, Any]]:
    details = _promotion_gate_components(candidate.get("metrics") or {}, baseline.get("metrics") or {}, args)
    return bool(details["pass_eq"] and details["pass_dd"] and details["pass_trades"] and details["pass_pf"]), details


def _passes_promotion_gate_multi(
    candidate_evals: List[Dict[str, Any]],
    baseline_evals: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, Dict[str, Any]]:
    if not candidate_evals or not baseline_evals or len(candidate_evals) != len(baseline_evals):
        return False, {"error": "invalid eval list lengths"}

    slice_details: List[Dict[str, Any]] = []
    total_eq_delta = 0.0
    total_eq_delta_weighted = 0.0
    total_weight = 0.0
    win_weight = 0.0
    mandatory_total = 0
    mandatory_pass_all = 0
    mandatory_pass_eq = 0
    pass_dd_all = True
    pass_trades_all = True
    pass_pf_all = True
    slice_wins = 0
    for idx, (cand, base) in enumerate(zip(candidate_evals, baseline_evals), start=1):
        prof = dict(cand.get("profile") or base.get("profile") or {})
        comp = _promotion_gate_components(cand.get("metrics") or {}, base.get("metrics") or {}, args)
        name = str(prof.get("name") or f"slice_{idx:02d}")
        regime = str(prof.get("regime") or name)
        tags = [str(x) for x in (prof.get("tags") or [])] if isinstance(prof.get("tags"), list) else []
        weight = max(0.0, _safe_float(prof.get("weight"), 1.0))
        mandatory = bool(prof.get("mandatory", False))
        pass_all_local = bool(comp.get("pass_eq", False) and comp.get("pass_dd", False) and comp.get("pass_trades", False) and comp.get("pass_pf", False))
        row = {
            "slice_index": idx,
            "slice_name": name,
            "regime": regime,
            "tags": tags,
            "weight": float(weight),
            "mandatory": bool(mandatory),
            "profile": prof,
            "pass_all_local": pass_all_local,
            **comp,
        }
        slice_details.append(row)
        total_eq_delta += float(comp.get("equity_delta", 0.0))
        total_eq_delta_weighted += float(comp.get("equity_delta", 0.0)) * float(weight)
        total_weight += float(weight)
        pass_dd_all = bool(pass_dd_all and comp.get("pass_dd", False))
        pass_trades_all = bool(pass_trades_all and comp.get("pass_trades", False))
        pass_pf_all = bool(pass_pf_all and comp.get("pass_pf", False))
        if bool(comp.get("pass_eq", False)):
            slice_wins += 1
            win_weight += float(weight)
        if mandatory:
            mandatory_total += 1
            if bool(comp.get("pass_eq", False)):
                mandatory_pass_eq += 1
            if pass_all_local:
                mandatory_pass_all += 1

    n = len(slice_details)
    mean_eq_delta = float(total_eq_delta / max(n, 1))
    weighted_mean_eq_delta = float(total_eq_delta_weighted / max(total_weight, 1e-12)) if total_weight > 0 else 0.0
    weighted_win_fraction = float(win_weight / max(total_weight, 1e-12)) if total_weight > 0 else 0.0
    required_slice_wins = int(args.min_slice_wins) if int(args.min_slice_wins) > 0 else int(n)
    pass_slice_wins = int(slice_wins) >= int(required_slice_wins)
    min_weighted_win_fraction = float(args.min_slice_win_weight_fraction)
    if min_weighted_win_fraction <= 0.0:
        pass_weighted_slice_wins = True
    else:
        pass_weighted_slice_wins = weighted_win_fraction >= min_weighted_win_fraction
    require_mandatory_all = bool(args.require_mandatory_slices_pass_all)
    if mandatory_total <= 0:
        pass_mandatory = True
    elif require_mandatory_all:
        pass_mandatory = int(mandatory_pass_all) >= int(mandatory_total)
    else:
        pass_mandatory = int(mandatory_pass_eq) >= int(mandatory_total)
    pass_mean_eq = mean_eq_delta >= float(args.min_equity_improvement)
    use_weighted_mean = bool(args.weighted_slice_gate)
    eq_gate_value = weighted_mean_eq_delta if use_weighted_mean else mean_eq_delta
    pass_eq_mean_gate = eq_gate_value >= float(args.min_equity_improvement)
    promoted = bool(
        pass_dd_all
        and pass_trades_all
        and pass_pf_all
        and pass_slice_wins
        and pass_weighted_slice_wins
        and pass_mandatory
        and pass_eq_mean_gate
    )
    details = {
        "num_slices": int(n),
        "slice_wins": int(slice_wins),
        "required_slice_wins": int(required_slice_wins),
        "sum_equity_delta": float(total_eq_delta),
        "mean_equity_delta": float(mean_eq_delta),
        "sum_weight": float(total_weight),
        "weighted_mean_equity_delta": float(weighted_mean_eq_delta),
        "weighted_slice_win_fraction": float(weighted_win_fraction),
        "min_slice_win_weight_fraction": float(min_weighted_win_fraction),
        "pass_weighted_slice_wins": bool(pass_weighted_slice_wins),
        "weighted_slice_gate_enabled": bool(use_weighted_mean),
        "eq_mean_gate_value": float(eq_gate_value),
        "min_equity_improvement_per_slice": float(args.min_equity_improvement),
        "pass_mean_eq": bool(pass_mean_eq),
        "pass_eq_mean_gate": bool(pass_eq_mean_gate),
        "pass_slice_wins": bool(pass_slice_wins),
        "mandatory_slices_total": int(mandatory_total),
        "mandatory_slices_pass_eq": int(mandatory_pass_eq),
        "mandatory_slices_pass_all": int(mandatory_pass_all),
        "require_mandatory_slices_pass_all": bool(require_mandatory_all),
        "pass_mandatory_slices": bool(pass_mandatory),
        "pass_dd_all": bool(pass_dd_all),
        "pass_trades_all": bool(pass_trades_all),
        "pass_pf_all": bool(pass_pf_all),
        "per_slice": slice_details,
    }
    return promoted, details


def _run_training_cycle(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = EpisodeTrainingConfig(
        n_epoch_episodes=int(args.train_episodes),
        notional=float(args.train_notional),
        pair_width=int(args.train_pair_width),
        bar_sequences_per_episode=int(args.train_bar_sequences),
        min_bar_window=int(args.train_min_window),
        max_bar_window=int(args.train_max_window),
        epochs=int(args.train_epochs),
        candles_per_extent=int(args.train_candles_per_extent),
        optimizer_name=str(args.optimizer),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        use_mechanical_veto=bool(args.train_use_veto),
        trade_update_prob=float(args.trade_update_prob),
        trade_update_min_abs_return=float(args.trade_update_min_abs_return),
    )
    trainer = EpochEpisodeTrainer(cfg)
    trainer.run_episode_training()
    last = trainer.results[-1] if trainer.results else {}
    return {
        "config": asdict(cfg),
        "last_result": last,
        "num_results": len(trainer.results),
    }


def _run_calibration_governor(args: argparse.Namespace, report_dir: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable or "python3",
        "hrm_trade_head_calibration_governor.py",
        "--report-dir", str(report_dir),
        "--min-improvement-bps", str(float(args.calibration_governor_min_improvement_bps)),
        "--min-val-samples", str(int(args.calibration_governor_min_val_samples)),
        "--val-fraction", str(float(args.calibration_governor_val_fraction)),
        "--min-val-files", str(int(args.calibration_governor_min_val_files)),
        "--max-candidates", str(int(args.calibration_governor_max_candidates)),
    ]
    if str(args.calibration_governor_out or "").strip():
        cmd += ["--out", str(args.calibration_governor_out)]
    if str(args.calibration_governor_inputs or "").strip():
        cmd += ["--inputs", str(args.calibration_governor_inputs)]
    if str(args.calibration_governor_train_inputs or "").strip():
        cmd += ["--train-inputs", str(args.calibration_governor_train_inputs)]
    if str(args.calibration_governor_val_inputs or "").strip():
        cmd += ["--val-inputs", str(args.calibration_governor_val_inputs)]
    if str(args.calibration_governor_current_calibration or "").strip():
        cmd += ["--current-calibration", str(args.calibration_governor_current_calibration)]
    if str(args.calibration_governor_min_scale_grid or "").strip():
        cmd += ["--min-scale-grid", str(args.calibration_governor_min_scale_grid)]
    if str(args.calibration_governor_min_bin_count_grid or "").strip():
        cmd += ["--min-bin-count-grid", str(args.calibration_governor_min_bin_count_grid)]
    if str(args.calibration_governor_confidence_bin_edges_grid or "").strip():
        cmd += ["--confidence-bin-edges-grid", str(args.calibration_governor_confidence_bin_edges_grid)]
    if float(args.calibration_governor_max_scale) > 0:
        cmd += ["--max-scale", str(float(args.calibration_governor_max_scale))]
    if bool(args.calibration_governor_promote):
        cmd.append("--promote")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result: Dict[str, Any] = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report_dir": str(report_dir.resolve()),
    }
    report_path = report_dir / "calibration_governor_report.json"
    if proc.returncode == 0 and report_path.exists():
        try:
            with open(report_path, "r") as f:
                result["report"] = json.load(f)
        except Exception as e:
            result["report_load_error"] = str(e)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Background HRM simmer manager")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--forever", action="store_true",
                   help="Run simmer cycles indefinitely until interrupted.")
    p.add_argument("--out-dir", type=str, default="simmer_runs")
    p.add_argument("--sleep-seconds", type=float, default=0.0)

    # Training simmer settings
    p.add_argument("--train-episodes", type=int, default=1)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--train-notional", type=float, default=100.0)
    p.add_argument("--train-pair-width", type=int, default=8)
    p.add_argument("--train-bar-sequences", type=int, default=12)
    p.add_argument("--train-min-window", type=int, default=64)
    p.add_argument("--train-max-window", type=int, default=96)
    p.add_argument("--train-candles-per-extent", type=int, default=1200)
    p.add_argument("--train-use-veto", action="store_true")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "lion", "muon"])
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--trade-update-prob", type=float, default=0.10)
    p.add_argument("--trade-update-min-abs-return", type=float, default=0.0)

    # Calibration governor (agent substep: OOS fit/sweep/promote trade-head calibration)
    p.add_argument(
        "--calibration-governor",
        dest="calibration_governor_enabled",
        action="store_true",
        help="Run OOS calibration governor before baseline validation (default: enabled).",
    )
    p.add_argument(
        "--no-calibration-governor",
        dest="calibration_governor_enabled",
        action="store_false",
        help="Disable calibration governor substep.",
    )
    p.set_defaults(calibration_governor_enabled=True)
    p.add_argument(
        "--calibration-governor-promote",
        dest="calibration_governor_promote",
        action="store_true",
        help="Allow governor to promote a better calibration artifact (default: enabled).",
    )
    p.add_argument(
        "--calibration-governor-no-promote",
        dest="calibration_governor_promote",
        action="store_false",
        help="Run governor in report-only mode.",
    )
    p.set_defaults(calibration_governor_promote=True)
    p.add_argument(
        "--calibration-governor-soft-fail",
        dest="calibration_governor_soft_fail",
        action="store_true",
        help="Continue simmer cycle if governor fails (default: enabled).",
    )
    p.add_argument(
        "--calibration-governor-hard-fail",
        dest="calibration_governor_soft_fail",
        action="store_false",
        help="Abort simmer cycle if governor fails.",
    )
    p.set_defaults(calibration_governor_soft_fail=True)
    p.add_argument("--calibration-governor-inputs", type=str, default="")
    p.add_argument("--calibration-governor-train-inputs", type=str, default="")
    p.add_argument("--calibration-governor-val-inputs", type=str, default="")
    p.add_argument("--calibration-governor-current-calibration", type=str, default="")
    p.add_argument("--calibration-governor-out", type=str, default="models/trained/hrm_trade_head_calibration.json")
    p.add_argument("--calibration-governor-val-fraction", type=float, default=0.30)
    p.add_argument("--calibration-governor-min-val-files", type=int, default=3)
    p.add_argument("--calibration-governor-min-improvement-bps", type=float, default=0.25)
    p.add_argument("--calibration-governor-min-val-samples", type=int, default=50)
    p.add_argument("--calibration-governor-max-candidates", type=int, default=12)
    p.add_argument("--calibration-governor-max-scale", type=float, default=2.0)
    p.add_argument("--calibration-governor-min-scale-grid", type=str, default="0.03,0.05,0.08")
    p.add_argument("--calibration-governor-min-bin-count-grid", type=str, default="20,30,50")
    p.add_argument(
        "--calibration-governor-confidence-bin-edges-grid",
        type=str,
        default="0.0,0.35,0.50,0.65,0.80,1.01;0.0,0.40,0.60,0.80,1.01;0.0,0.50,0.70,1.01",
    )

    # Validation gate settings (profit profile discovered so far)
    p.add_argument("--validation-symbols", type=str, default="BTCUSDT,ETHUSDT")
    p.add_argument("--validation-seq-len", type=int, default=64)
    p.add_argument("--validation-top-k", type=int, default=1)
    p.add_argument("--validation-hold-bars", type=int, default=12)
    p.add_argument("--validation-cooldown-bars", type=int, default=-1)
    p.add_argument("--validation-signal-threshold", type=float, default=0.55)
    p.add_argument("--validation-commission-bps", type=float, default=5.0)
    p.add_argument("--validation-slippage-bps", type=float, default=3.0)
    p.add_argument("--validation-start", type=str, default="", help="Optional base validation start timestamp/date")
    p.add_argument("--validation-end", type=str, default="", help="Optional base validation end timestamp/date")
    p.add_argument("--validation-slices-json", type=str, default="",
                   help="Optional JSON file: list of validation slice objects (name/start/end and optional overrides)")
    p.add_argument("--validation-regime-manifest", type=str, default="",
                   help="Optional canonical regime manifest JSON (used when validation-slices-json is not set)")
    p.add_argument(
        "--validation-use-regime-manifest",
        dest="validation_use_regime_manifest",
        action="store_true",
        help="Use canonical regime manifest for validation slices when available (default: enabled).",
    )
    p.add_argument(
        "--validation-no-regime-manifest",
        dest="validation_use_regime_manifest",
        action="store_false",
        help="Disable canonical regime manifest auto-loading and fall back to a single base validation slice.",
    )
    p.set_defaults(validation_use_regime_manifest=True)
    p.add_argument("--validation-trade-head-calibration", type=str, default="")
    p.add_argument("--validation-no-trade-head-calibration", action="store_true")
    p.add_argument("--validation-no-risk-head-repair", action="store_true")
    p.add_argument("--validation-repair-min-stop-loss-pct", type=float, default=0.002)
    p.add_argument("--validation-repair-max-stop-loss-pct", type=float, default=0.15)
    p.add_argument("--validation-repair-min-take-profit-pct", type=float, default=0.003)
    p.add_argument("--validation-max-bars", type=int, default=420)
    p.add_argument("--validation-max-steps", type=int, default=120)
    p.add_argument("--validation-veto-confidence-override-threshold", type=float, default=-1.0)
    p.add_argument("--validation-veto-confidence-override-size-scale", type=float, default=0.5)
    p.add_argument("--validation-veto-confidence-override-reasons", type=str,
                   default="low_confidence,poor_risk_reward")
    p.add_argument(
        "--validation-no-veto",
        dest="validation_no_veto",
        action="store_true",
        help="Disable mechanical veto during validation (default: disabled for current best profile).",
    )
    p.add_argument(
        "--validation-use-veto",
        dest="validation_no_veto",
        action="store_false",
        help="Force mechanical veto during validation.",
    )
    p.set_defaults(validation_no_veto=True)
    p.add_argument("--validation-carry-memory", action="store_true")

    # Promotion gate
    p.add_argument("--min-equity-improvement", type=float, default=0.01,
                   help="Minimum absolute final-equity improvement vs baseline to promote")
    p.add_argument("--max-drawdown-floor", type=float, default=-0.02,
                   help="Candidate must have max_drawdown >= this (negative)")
    p.add_argument("--min-trades", type=int, default=1)
    p.add_argument("--min-profit-factor", type=float, default=0.8)
    p.add_argument("--min-slice-wins", type=int, default=0,
                   help="For multi-slice validation, minimum slices that must beat baseline (0 = all slices)")
    p.add_argument(
        "--weighted-slice-gate",
        dest="weighted_slice_gate",
        action="store_true",
        help="Use weighted mean equity delta across slices for promotion (default: enabled).",
    )
    p.add_argument(
        "--unweighted-slice-gate",
        dest="weighted_slice_gate",
        action="store_false",
        help="Use simple mean equity delta across slices.",
    )
    p.set_defaults(weighted_slice_gate=True)
    p.add_argument(
        "--min-slice-win-weight-fraction",
        type=float,
        default=0.0,
        help="Optional minimum fraction of total slice weight that must beat baseline (0 disables).",
    )
    p.add_argument(
        "--require-mandatory-slices-pass-all",
        action="store_true",
        help="Require mandatory slices to pass all local gates (eq/dd/trades/pf), not just equity.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        validation_slices = _validation_slice_specs(args)
    except Exception as e:
        raise SystemExit(f"Failed to load validation slices: {e}")

    cycle_idx = 0
    while True:
        cycle_idx += 1
        cycle_dir = root / f"cycle_{cycle_idx:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Simmer Cycle {cycle_idx} ===")
        print(f"Cycle dir: {cycle_dir}")
        base_profile = _base_validation_profile(args)
        print(
            "Validation profile: "
            f"symbols={base_profile['symbols']} "
            f"thr={float(base_profile['signal_threshold']):.2f} "
            f"top_k={int(base_profile['top_k'])} "
            f"hold={int(base_profile['hold_bars'])} "
            f"cooldown={int(base_profile['cooldown_bars'])} "
            f"veto={'off' if bool(base_profile['no_mechanical_veto']) else 'on'} "
            f"veto_conf_override={float(base_profile['veto_confidence_override_threshold']):.2f} "
            f"veto_conf_size={float(base_profile['veto_confidence_override_size_scale']):.2f} "
            f"veto_conf_reasons={str(base_profile['veto_confidence_override_reasons'])} "
            f"calib={'off' if bool(base_profile['no_trade_head_calibration']) else ('custom' if bool(base_profile['trade_head_calibration']) else 'auto')} "
            f"risk_repair={'off' if bool(base_profile['no_risk_head_repair']) else 'on'} "
            f"memory={'on' if bool(base_profile['carry_memory']) else 'off'} "
            f"slices={len(validation_slices)}"
        )
        if len(validation_slices) > 1:
            print(
                "Validation slices: "
                + ", ".join(
                    f"{str(s.get('name'))}/{str(s.get('regime') or s.get('name'))}"
                    f"(w={float(_safe_float(s.get('weight'), 1.0)):.2f}"
                    f"{',M' if bool(s.get('mandatory', False)) else ''})"
                    f"[{str(s.get('start') or '')}..{str(s.get('end') or '')}]"
                    for s in validation_slices
                )
            )
        validation_manifest_summary = summarize_validation_manifest_profiles(validation_slices)
        if validation_manifest_summary:
            print(
                "Validation manifest: "
                f"{validation_manifest_summary.get('manifest_name') or '(unnamed)'} "
                f"v{validation_manifest_summary.get('manifest_version')} | "
                f"path={validation_manifest_summary.get('manifest_source_path')} | "
                f"slices={int(validation_manifest_summary.get('slice_count', 0))} | "
                f"mandatory={list(validation_manifest_summary.get('mandatory_slices', []))}"
            )

        calibration_governor_result: Optional[Dict[str, Any]] = None
        if bool(args.calibration_governor_enabled):
            gov_dir = cycle_dir / "calibration_governor"
            print(
                "Calibration governor: "
                f"{'promote' if bool(args.calibration_governor_promote) else 'report-only'} "
                f"(min_impr={float(args.calibration_governor_min_improvement_bps):.3f}bps, "
                f"min_val_samples={int(args.calibration_governor_min_val_samples)}, "
                f"max_candidates={int(args.calibration_governor_max_candidates)})"
            )
            calibration_governor_result = _run_calibration_governor(args, gov_dir)
            print(calibration_governor_result.get("stdout", ""))
            if calibration_governor_result.get("returncode") != 0:
                print(calibration_governor_result.get("stderr", ""))
                if not bool(args.calibration_governor_soft_fail):
                    raise SystemExit("Calibration governor failed and hard-fail mode is enabled")
            else:
                report = calibration_governor_result.get("report") or {}
                decision = dict(report.get("decision") or {})
                best = dict(report.get("best_candidate") or {})
                best_eval = dict(best.get("eval") or {})
                baseline = dict(report.get("baseline") or {})
                current_cal_eval = dict(baseline.get("current_calibration_eval_on_val") or {})
                current_cal_wmae = None
                if current_cal_eval:
                    current_cal_wmae = _safe_float(
                        ((current_cal_eval.get("calibrated") or {}).get("weighted_mae_bps")),
                        0.0,
                    )
                else:
                    current_cal_wmae = _safe_float(
                        ((baseline.get("raw_eval_on_val") or {}).get("raw") or {}).get("weighted_mae_bps"),
                        0.0,
                    )
                best_wmae = _safe_float(((best_eval.get("calibrated") or {}).get("weighted_mae_bps")), 0.0)
                print(
                    "Calibration governor decision: "
                    f"{str(decision.get('reason'))} | promoted={bool(decision.get('promoted', False))} | "
                    f"val_wMAE {current_cal_wmae:.4f} -> {best_wmae:.4f} bps"
                )

        backup_dir = cycle_dir / "backup_before"
        backup_manifest = _backup_artifacts(backup_dir)

        baseline_evals: List[Dict[str, Any]] = []
        for slice_i, slice_profile in enumerate(validation_slices, start=1):
            slice_name = str(slice_profile.get("name") or f"slice_{slice_i:02d}")
            out_dir = cycle_dir / "baseline_eval" / f"{slice_i:02d}_{slice_name}"
            baseline_eval = _run_walkforward(args, out_dir, profile=slice_profile)
            baseline_evals.append(baseline_eval)
            if baseline_eval.get("returncode") != 0:
                raise SystemExit(
                    f"Baseline validation failed on {slice_name}:\n{baseline_eval.get('stderr','')}"
                )
            print(baseline_eval.get("stdout", ""))

        train_summary = _run_training_cycle(args)

        candidate_evals: List[Dict[str, Any]] = []
        candidate_failure: Optional[Dict[str, Any]] = None
        for slice_i, slice_profile in enumerate(validation_slices, start=1):
            slice_name = str(slice_profile.get("name") or f"slice_{slice_i:02d}")
            out_dir = cycle_dir / "candidate_eval" / f"{slice_i:02d}_{slice_name}"
            candidate_eval = _run_walkforward(args, out_dir, profile=slice_profile)
            candidate_evals.append(candidate_eval)
            if candidate_eval.get("returncode") != 0 and candidate_failure is None:
                candidate_failure = {
                    "slice_index": slice_i,
                    "slice_name": slice_name,
                    "returncode": candidate_eval.get("returncode"),
                    "stderr": candidate_eval.get("stderr", ""),
                }
            print(candidate_eval.get("stdout", ""))

        if candidate_failure is not None:
            failed_eval = candidate_evals[int(candidate_failure["slice_index"]) - 1]
            print(failed_eval.get("stderr", ""))
            _restore_artifacts(backup_dir)
            promoted = False
            gate_details = {"error": "candidate validation failed", **candidate_failure}
        else:
            if len(validation_slices) == 1:
                promoted, gate_details = _passes_promotion_gate(candidate_evals[0], baseline_evals[0], args)
            else:
                promoted, gate_details = _passes_promotion_gate_multi(candidate_evals, baseline_evals, args)
            if not promoted:
                _restore_artifacts(backup_dir)

        if isinstance(gate_details, dict):
            if "per_slice" in gate_details:
                print(
                    "Promotion gate (multi-slice): "
                    f"wins={int(gate_details.get('slice_wins', 0))}/{int(gate_details.get('required_slice_wins', 0))} "
                    f"mean_eq_delta={float(gate_details.get('mean_equity_delta', 0.0)):+.4f} "
                    f"weighted_mean_eq_delta={float(gate_details.get('weighted_mean_equity_delta', 0.0)):+.4f} "
                    f"win_weight_frac={float(gate_details.get('weighted_slice_win_fraction', 0.0)):.2f} "
                    f"mandatory={int(gate_details.get('mandatory_slices_pass_eq', 0))}/"
                    f"{int(gate_details.get('mandatory_slices_total', 0))} "
                    f"dd_all={'yes' if bool(gate_details.get('pass_dd_all', False)) else 'no'} "
                    f"pf_all={'yes' if bool(gate_details.get('pass_pf_all', False)) else 'no'} "
                    f"trades_all={'yes' if bool(gate_details.get('pass_trades_all', False)) else 'no'}"
                )
                for row in gate_details.get("per_slice") or []:
                    print(
                        f"  - {str(row.get('slice_name', 'slice'))}: "
                        f"regime={str(row.get('regime', ''))} "
                        f"w={float(row.get('weight', 1.0)):.2f}{' M' if bool(row.get('mandatory', False)) else ''} "
                        f"eq_delta={float(row.get('equity_delta', 0.0)):+.4f} "
                        f"dd={float(row.get('candidate_max_drawdown', 0.0)):+.4f} "
                        f"trades={int(row.get('candidate_total_trades', 0) or 0)} "
                        f"pf={float(row.get('candidate_profit_factor', 0.0)):.3f} "
                        f"pass_eq={'yes' if bool(row.get('pass_eq', False)) else 'no'} "
                        f"pass_all={'yes' if bool(row.get('pass_all_local', False)) else 'no'}"
                    )
            elif "error" not in gate_details:
                print(
                    "Promotion gate: "
                    f"eq_delta={float(gate_details.get('equity_delta', 0.0)):+.4f} "
                    f"dd={float(gate_details.get('candidate_max_drawdown', 0.0)):+.4f} "
                    f"trades={int(gate_details.get('candidate_total_trades', 0) or 0)} "
                    f"pf={float(gate_details.get('candidate_profit_factor', 0.0)):.3f} "
                    f"pass_eq={'yes' if bool(gate_details.get('pass_eq', False)) else 'no'} "
                    f"pass_dd={'yes' if bool(gate_details.get('pass_dd', False)) else 'no'} "
                    f"pass_trades={'yes' if bool(gate_details.get('pass_trades', False)) else 'no'} "
                    f"pass_pf={'yes' if bool(gate_details.get('pass_pf', False)) else 'no'}"
                )

        summary = {
            "created_at": _now_iso(),
            "cycle_index": cycle_idx,
            "backup_manifest": backup_manifest,
            "calibration_governor": (
                {
                    "command": calibration_governor_result.get("command"),
                    "returncode": calibration_governor_result.get("returncode"),
                    "report_dir": calibration_governor_result.get("report_dir"),
                    "report_load_error": calibration_governor_result.get("report_load_error"),
                    "report": calibration_governor_result.get("report"),
                } if isinstance(calibration_governor_result, dict) else None
            ),
            "effective_validation_profile": base_profile,
            "validation_manifest": validation_manifest_summary,
            "validation_slices": validation_slices,
            "train_summary": train_summary,
            "baseline_eval": ({
                k: v for k, v in baseline_evals[0].items() if k in {"profile", "command", "returncode", "out_dir", "metrics"}
            } if len(baseline_evals) == 1 else None),
            "candidate_eval": ({
                k: v for k, v in candidate_evals[0].items() if k in {"profile", "command", "returncode", "out_dir", "metrics"}
            } if len(candidate_evals) == 1 else None),
            "baseline_evals": [
                {k: v for k, v in ev.items() if k in {"profile", "command", "returncode", "out_dir", "metrics"}}
                for ev in baseline_evals
            ],
            "candidate_evals": [
                {k: v for k, v in ev.items() if k in {"profile", "command", "returncode", "out_dir", "metrics"}}
                for ev in candidate_evals
            ],
            "promotion": {
                "promoted": promoted,
                "gate_details": gate_details,
            },
        }
        with open(cycle_dir / "simmer_cycle_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if promoted:
            print("✅ Candidate promoted")
        else:
            print("↩️  Candidate rejected, restored prior checkpoint")

        if not bool(args.forever) and cycle_idx >= int(args.cycles):
            break

        if float(args.sleep_seconds) > 0:
            import time
            time.sleep(float(args.sleep_seconds))


if __name__ == "__main__":
    main()
