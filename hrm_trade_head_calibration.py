#!/usr/bin/env python3
"""Fit a post-hoc calibration artifact for HRM trade magnitude heads from trades.json files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hrm.trade_head_calibration import fit_trade_head_calibration_from_trade_rows


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _discover_trade_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for token in inputs:
        if any(ch in token for ch in "*?[]"):
            for p in glob.glob(token, recursive=True):
                path = Path(p)
                if path.is_file() and path.name == "trades.json":
                    files.append(path)
            continue
        p = Path(token)
        if p.is_dir():
            files.extend(sorted(p.rglob("trades.json")))
        elif p.is_file():
            if p.name == "trades.json":
                files.append(p)
    # de-dup, preserve order
    out: List[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _load_trade_rows(files: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    row = dict(row)
                    row["_source_trades_path"] = str(p)
                    rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description="Fit HRM trade-head calibration from backtest trades")
    p.add_argument(
        "--inputs",
        type=str,
        default="walkforward_results,walkforward_sweeps",
        help="Comma-separated files/dirs/globs to scan for trades.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="models/trained/hrm_trade_head_calibration.json",
        help="Output calibration artifact path",
    )
    p.add_argument("--min-bin-count", type=int, default=30)
    p.add_argument("--min-scale", type=float, default=0.05)
    p.add_argument("--max-scale", type=float, default=2.0)
    p.add_argument(
        "--confidence-bin-edges",
        type=str,
        default="0.0,0.35,0.50,0.65,0.80,1.01",
        help="Comma-separated confidence bin edges",
    )
    args = p.parse_args()

    input_tokens = _parse_list(args.inputs)
    if not input_tokens:
        raise SystemExit("No inputs provided")
    trade_files = _discover_trade_files(input_tokens)
    if not trade_files:
        raise SystemExit("No trades.json files discovered")

    rows = _load_trade_rows(trade_files)
    if not rows:
        raise SystemExit("No trade rows loaded")

    edges = [float(x) for x in _parse_list(args.confidence_bin_edges)]
    payload = fit_trade_head_calibration_from_trade_rows(
        rows,
        confidence_bin_edges=edges,
        min_bin_count=int(args.min_bin_count),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
    )
    payload["fit_stats"]["trade_files_seen"] = int(len(trade_files))
    payload["fit_stats"]["input_tokens"] = list(input_tokens)
    payload["fit_stats"]["sample_trade_files"] = [str(p) for p in trade_files[:20]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    move = payload["move_calibration"]
    print(f"✅ Wrote calibration: {out_path}")
    print(
        f"Trade files {payload['fit_stats']['trade_files_seen']} | "
        f"rows {payload['fit_stats']['trade_rows_seen']} | "
        f"samples {payload['fit_stats']['samples_used']}"
    )
    print(
        f"Global move scale {move['global_scale']:.4f} "
        f"(raw median ratio {payload['fit_stats']['ratio_median_raw']:.4f})"
    )
    for b in move.get("bins", []):
        print(
            f"  conf[{b['conf_min']:.2f},{b['conf_max']:.2f}) "
            f"n={b['count']} scale={b['scale']:.4f} med={b['median_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()

