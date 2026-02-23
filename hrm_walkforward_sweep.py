#!/usr/bin/env python3
"""Grid sweep wrapper for hrm_walkforward_backtest.py."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_list(raw: str, cast):
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    return [cast(x) for x in items]


def _parse_bool_list(raw: str) -> List[bool]:
    vals = []
    for x in [s.strip().lower() for s in (raw or "").split(",") if s.strip()]:
        if x in {"1", "true", "yes", "on", "y"}:
            vals.append(True)
        elif x in {"0", "false", "no", "off", "n"}:
            vals.append(False)
        else:
            raise ValueError(f"invalid bool token: {x}")
    return vals


def _variant_id(i: int, params: Dict[str, Any]) -> str:
    veto = "veto" if params["use_mechanical_veto"] else "noveto"
    mem = "mem" if params["carry_memory"] else "nomem"
    veto_conf = float(params.get("veto_conf_override_threshold", -1.0))
    veto_size = float(params.get("veto_conf_override_size_scale", 0.5))
    veto_conf_tag = f"_vc{veto_conf:.2f}_vs{veto_size:.2f}" if veto_conf > 0 else ""
    return (
        f"v{i:02d}_h{params['hold_bars']}_k{params['top_k']}_thr{params['signal_threshold']:.2f}_"
        f"pm{params['min_pred_move_bps']:.0f}_ec{params['edge_cost_multiplier']:.1f}_{veto}_{mem}"
        f"{veto_conf_tag}"
    )


def _sort_key(row: Dict[str, Any], max_dd_constraint: float) -> tuple:
    m = row.get("metrics") or {}
    max_dd = float(m.get("max_drawdown", 0.0) or 0.0)
    constrained_ok = (max_dd_constraint >= 0) or (max_dd >= max_dd_constraint)
    final_eq = float(m.get("final_equity", 0.0) or 0.0)
    sharpe = float(m.get("sharpe_ratio", 0.0) or 0.0)
    return (1 if constrained_ok else 0, final_eq, sharpe)


def main():
    p = argparse.ArgumentParser(description="Sweep HRM walk-forward backtest params")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--min-history", type=int, default=64)
    p.add_argument("--hold-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=-1)
    p.add_argument("--hold-bars-grid", type=str, default="", help="Optional comma-separated hold bars grid")
    p.add_argument("--max-bars", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--commission-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--no-risk-head-repair", action="store_true")
    p.add_argument("--repair-min-stop-loss-pct", type=float, default=0.002)
    p.add_argument("--repair-max-stop-loss-pct", type=float, default=0.15)
    p.add_argument("--repair-min-take-profit-pct", type=float, default=0.003)
    p.add_argument("--thresholds", type=str, default="0.25,0.40,0.55")
    p.add_argument("--topks", type=str, default="1,2")
    p.add_argument("--veto", type=str, default="true,false", help="Comma-separated booleans")
    p.add_argument("--carry-memory", type=str, default="false,true", help="Comma-separated booleans")
    p.add_argument("--min-pred-move-bps-grid", type=str, default="0,10,20", help="Comma-separated minimum predicted move bps")
    p.add_argument("--edge-cost-mult-grid", type=str, default="1.0,1.5", help="Comma-separated edge/cost multipliers")
    p.add_argument("--veto-conf-override-threshold-grid", type=str, default="-1",
                   help="Comma-separated veto confidence override thresholds (-1 disables)")
    p.add_argument("--veto-conf-override-size-scale-grid", type=str, default="0.5",
                   help="Comma-separated size scales when veto is overridden")
    p.add_argument("--veto-conf-override-reasons", type=str, default="low_confidence,poor_risk_reward",
                   help="Comma-separated veto reasons eligible for confidence override ('any' to allow all)")
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--trade-head-calibration", type=str, default="")
    p.add_argument("--no-trade-head-calibration", action="store_true")
    p.add_argument("--online-pretrain-steps", type=int, default=0)
    p.add_argument("--start", type=str, default="")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--max-drawdown-stop", type=float, default=-0.25)
    p.add_argument("--max-dd-constraint", type=float, default=-0.15, help="Ranking constraint (negative). Set >=0 to disable")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--python", type=str, default=sys.executable or "python3")
    args = p.parse_args()

    thresholds = _parse_list(args.thresholds, float)
    topks = _parse_list(args.topks, int)
    veto_vals = _parse_bool_list(args.veto)
    mem_vals = _parse_bool_list(args.carry_memory)
    pred_move_bps_grid = _parse_list(args.min_pred_move_bps_grid, float)
    edge_cost_mult_grid = _parse_list(args.edge_cost_mult_grid, float)
    veto_conf_override_thr_grid = _parse_list(args.veto_conf_override_threshold_grid, float)
    veto_conf_override_size_grid = _parse_list(args.veto_conf_override_size_scale_grid, float)
    hold_bars_grid = _parse_list(args.hold_bars_grid, int) if args.hold_bars_grid else [int(args.hold_bars)]

    if (
        not thresholds or not topks or not veto_vals or not mem_vals
        or not pred_move_bps_grid or not edge_cost_mult_grid or not hold_bars_grid
        or not veto_conf_override_thr_grid or not veto_conf_override_size_grid
    ):
        raise SystemExit("Empty grid dimension")

    root = Path(args.out_dir) if args.out_dir else Path("walkforward_sweeps") / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)

    variants = []
    for i, (hold_bars, thr, k, veto, mem, pred_move_bps, edge_cost_mult, veto_conf_thr, veto_conf_size) in enumerate(
        itertools.product(
            hold_bars_grid,
            thresholds,
            topks,
            veto_vals,
            mem_vals,
            pred_move_bps_grid,
            edge_cost_mult_grid,
            veto_conf_override_thr_grid,
            veto_conf_override_size_grid,
        ),
        start=1,
    ):
        # Veto confidence override only matters when mechanical veto is enabled.
        if (not bool(veto)) and float(veto_conf_thr) > 0.0:
            continue
        params = {
            "hold_bars": int(hold_bars),
            "signal_threshold": float(thr),
            "top_k": int(k),
            "use_mechanical_veto": bool(veto),
            "carry_memory": bool(mem),
            "min_pred_move_bps": float(pred_move_bps),
            "edge_cost_multiplier": float(edge_cost_mult),
            "veto_conf_override_threshold": float(veto_conf_thr),
            "veto_conf_override_size_scale": float(veto_conf_size),
        }
        variant_name = _variant_id(i, params)
        run_dir = root / variant_name
        cmd = [
            args.python,
            "hrm_walkforward_backtest.py",
            "--symbols", args.symbols,
            "--initial-capital", str(args.initial_capital),
            "--seq-len", str(args.seq_len),
            "--min-history", str(args.min_history),
            "--hold-bars", str(hold_bars),
            "--cooldown-bars", str(args.cooldown_bars),
            "--max-bars", str(args.max_bars),
            "--max-steps", str(args.max_steps),
            "--commission-bps", str(args.commission_bps),
            "--slippage-bps", str(args.slippage_bps),
            "--repair-min-stop-loss-pct", str(args.repair_min_stop_loss_pct),
            "--repair-max-stop-loss-pct", str(args.repair_max_stop_loss_pct),
            "--repair-min-take-profit-pct", str(args.repair_min_take_profit_pct),
            "--signal-threshold", str(thr),
            "--top-k", str(k),
            "--min-pred-move-bps", str(pred_move_bps),
            "--edge-cost-multiplier", str(edge_cost_mult),
            "--online-pretrain-steps", str(args.online_pretrain_steps),
            "--max-drawdown-stop", str(args.max_drawdown_stop),
            "--out-dir", str(run_dir),
        ]
        if args.start:
            cmd += ["--start", args.start]
        if args.end:
            cmd += ["--end", args.end]
        if args.weights:
            cmd += ["--weights", args.weights]
        if args.trade_head_calibration:
            cmd += ["--trade-head-calibration", args.trade_head_calibration]
        if args.no_trade_head_calibration:
            cmd += ["--no-trade-head-calibration"]
        if mem:
            cmd += ["--carry-memory"]
        if args.no_risk_head_repair:
            cmd += ["--no-risk-head-repair"]
        if not veto:
            cmd += ["--no-mechanical-veto"]
        elif float(veto_conf_thr) > 0.0:
            cmd += [
                "--veto-confidence-override-threshold", str(veto_conf_thr),
                "--veto-confidence-override-size-scale", str(veto_conf_size),
                "--veto-confidence-override-reasons", str(args.veto_conf_override_reasons),
            ]

        print(f"\n[{i}] Running {variant_name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            variants.append({
                "variant": variant_name,
                "params": params,
                "status": "error",
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:],
            })
            continue

        metrics_path = run_dir / "metrics.json"
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            variants.append({
                "variant": variant_name,
                "params": params,
                "status": "error",
                "returncode": 0,
                "stderr": f"metrics read failed: {e}",
            })
            continue

        variants.append({
            "variant": variant_name,
            "params": params,
            "status": "ok",
            "run_dir": str(run_dir.resolve()),
            "metrics": metrics,
        })

    ranked = sorted(variants, key=lambda r: _sort_key(r, float(args.max_dd_constraint)), reverse=True)
    summary = {
        "created_at": datetime.now().isoformat(),
        "root_dir": str(root.resolve()),
        "grid": {
            "thresholds": thresholds,
            "topks": topks,
            "hold_bars": args.hold_bars,
            "hold_bars_grid": hold_bars_grid,
            "cooldown_bars": args.cooldown_bars,
            "veto": veto_vals,
            "carry_memory": mem_vals,
            "min_pred_move_bps_grid": pred_move_bps_grid,
            "edge_cost_mult_grid": edge_cost_mult_grid,
            "veto_conf_override_threshold_grid": veto_conf_override_thr_grid,
            "veto_conf_override_size_scale_grid": veto_conf_override_size_grid,
            "veto_conf_override_reasons": str(args.veto_conf_override_reasons),
            "trade_head_calibration": str(args.trade_head_calibration),
            "no_trade_head_calibration": bool(args.no_trade_head_calibration),
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "no_risk_head_repair": bool(args.no_risk_head_repair),
            "repair_min_stop_loss_pct": float(args.repair_min_stop_loss_pct),
            "repair_max_stop_loss_pct": float(args.repair_max_stop_loss_pct),
            "repair_min_take_profit_pct": float(args.repair_min_take_profit_pct),
            "max_steps": args.max_steps,
            "max_bars": args.max_bars,
        },
        "variants": variants,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }

    with open(root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSweep complete")
    print(f"Results: {root}")
    if ranked:
        best = ranked[0]
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
