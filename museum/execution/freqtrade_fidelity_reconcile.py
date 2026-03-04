#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception as e:
                rows.append(
                    {
                        "_parse_error": str(e),
                        "_line_no": i,
                        "_raw_line": raw[:2000],
                    }
                )
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                rows.append({"value": parsed, "_line_no": i})
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: _json_safe(v) for k, v in row.items()})


def _get_nested(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _first_present(obj: Dict[str, Any], paths: Sequence[str]) -> Any:
    for p in paths:
        v = _get_nested(obj, p)
        if v is not None:
            return v
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _sign(x: Optional[float]) -> int:
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def extract_fill_view(fill_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a fill/trade event row into a reconciliation-friendly view.

    Supports:
    - flat moneyfan-style rows with `signal_id`, `entry_price`, `exit_price`, `pnl_pct`
    - nested Freqtrade-like rows under `trade.*`
    """
    signal_id = str(_first_present(fill_row, ["signal_id", "trade.signal_id", "metadata.signal_id"]) or "").strip() or None
    pair = _first_present(fill_row, ["pair", "trade.pair"])
    side = _first_present(fill_row, ["side", "trade.side"])
    if side is None:
        is_short = _as_bool(_first_present(fill_row, ["is_short", "trade.is_short"]))
        if is_short is True:
            side = "short"
        elif is_short is False:
            side = "long"
    side = str(side).lower() if side is not None else None

    entry_price = _as_float(_first_present(fill_row, ["entry_price", "trade.entry_price", "open_rate", "trade.open_rate"]))
    exit_price = _as_float(_first_present(fill_row, ["exit_price", "trade.exit_price", "close_rate", "trade.close_rate"]))

    pnl_abs = _as_float(_first_present(fill_row, ["pnl_abs", "trade.pnl_abs", "profit_abs", "trade.profit_abs"]))
    pnl_pct = _as_float(_first_present(fill_row, ["pnl_pct", "trade.pnl_pct", "profit_ratio", "trade.profit_ratio"]))
    fees_abs = _as_float(_first_present(fill_row, ["fees_abs", "trade.fees_abs", "fee_open_cost", "trade.fee_open_cost"]))

    status = _first_present(fill_row, ["status", "trade.status"])
    if status is None:
        is_open = _as_bool(_first_present(fill_row, ["is_open", "trade.is_open"]))
        if is_open is True:
            status = "open"
        elif is_open is False:
            status = "closed"
    status = str(status).lower() if status is not None else None

    fill_ts = _first_present(
        fill_row,
        [
            "ts_utc",
            "timestamp",
            "trade.close_date",
            "trade.open_date",
            "close_date",
            "open_date",
        ],
    )
    exchange_trade_id = _first_present(fill_row, ["exchange_trade_id", "trade.id", "id"])
    schema = _first_present(fill_row, ["schema"]) or None

    if pnl_pct is None and entry_price is not None and exit_price is not None:
        raw = (float(exit_price) - float(entry_price)) / max(float(entry_price), 1e-12)
        if side == "short":
            raw = -raw
        pnl_pct = float(raw)

    return {
        "signal_id": signal_id,
        "pair": pair,
        "side": side,
        "status": status,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "fees_abs": fees_abs,
        "fill_ts_utc": fill_ts,
        "exchange_trade_id": exchange_trade_id,
        "fill_schema": schema,
        "raw": fill_row,
    }


def _latest_by_signal_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        signal_id = str(row.get("signal_id", "") or "").strip()
        if not signal_id:
            continue
        out[signal_id] = row
    return out


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mx = fmean(xs)
    my = fmean(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = math.sqrt(dx2 * dy2)
    if den <= 0.0:
        return None
    return float(num / den)


def _rmse(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(math.sqrt(fmean([v * v for v in values])))


def reconcile_fidelity_logs(
    dispatch_rows: List[Dict[str, Any]],
    ack_rows: List[Dict[str, Any]],
    fill_rows: List[Dict[str, Any]],
    require_success_ack_for_fill_match: bool = False,
) -> Dict[str, Any]:
    dispatch_valid: List[Dict[str, Any]] = []
    dispatch_parse_errors = 0
    for row in dispatch_rows:
        if row.get("_parse_error"):
            dispatch_parse_errors += 1
            continue
        if str(row.get("schema", "") or "") != "moneyfan.hrm.fidelity.dispatch.v1":
            continue
        if not str(row.get("signal_id", "") or "").strip():
            continue
        dispatch_valid.append(row)

    ack_valid: List[Dict[str, Any]] = []
    ack_parse_errors = 0
    for row in ack_rows:
        if row.get("_parse_error"):
            ack_parse_errors += 1
            continue
        if str(row.get("schema", "") or "") != "moneyfan.freqtrade.dispatch_ack.v1":
            continue
        if not str(row.get("signal_id", "") or "").strip():
            continue
        ack_valid.append(row)

    fill_parse_errors = 0
    fill_views: List[Dict[str, Any]] = []
    for row in fill_rows:
        if row.get("_parse_error"):
            fill_parse_errors += 1
            continue
        view = extract_fill_view(row)
        if not view.get("signal_id"):
            continue
        fill_views.append(view)

    ack_latest = _latest_by_signal_id(ack_valid)
    fill_latest = _latest_by_signal_id(fill_views)

    dispatch_ids = {str(r["signal_id"]) for r in dispatch_valid}
    ack_ids = set(ack_latest.keys())
    fill_ids = set(fill_latest.keys())

    detail_rows: List[Dict[str, Any]] = []
    matched_pred_bps: List[float] = []
    matched_real_bps: List[float] = []
    pred_error_bps: List[float] = []
    abs_pred_error_bps: List[float] = []
    directional_hits: List[float] = []
    adverse_entry_slippage_bps_values: List[float] = []

    for d in dispatch_valid:
        signal_id = str(d["signal_id"])
        ack = ack_latest.get(signal_id)
        fill = fill_latest.get(signal_id)

        instr = d.get("instrument") if isinstance(d.get("instrument"), dict) else {}
        pred = d.get("prediction") if isinstance(d.get("prediction"), dict) else {}
        risk = d.get("risk") if isinstance(d.get("risk"), dict) else {}

        ack_status = ack.get("status") if isinstance(ack, dict) else None
        ack_success = ack_status in {"dry_run_forwarded", "webhook_forwarded"}
        has_ack = ack is not None
        has_fill = fill is not None
        fill_match_allowed = bool(has_fill and (not require_success_ack_for_fill_match or ack_success))

        dispatch_side = str(instr.get("side", "") or "").lower() or None
        dispatch_price = _as_float(instr.get("price"))
        entry_price = _as_float(fill.get("entry_price") if fill else None)
        exit_price = _as_float(fill.get("exit_price") if fill else None)

        pred_fwd_return = _as_float(pred.get("pred_fwd_return"))
        pred_bps = float(pred_fwd_return * 10000.0) if pred_fwd_return is not None else None

        realized_pct = _as_float(fill.get("pnl_pct") if fill else None)
        realized_bps = float(realized_pct * 10000.0) if realized_pct is not None else None

        adverse_entry_slippage_bps: Optional[float] = None
        if dispatch_price is not None and entry_price is not None and dispatch_price > 0:
            raw_entry_move_bps = (float(entry_price) - float(dispatch_price)) / float(dispatch_price) * 10000.0
            if dispatch_side == "short":
                adverse_entry_slippage_bps = float(-raw_entry_move_bps)
            else:
                adverse_entry_slippage_bps = float(raw_entry_move_bps)

        error_bps = None
        abs_error = None
        directional_match = None
        if fill_match_allowed and pred_bps is not None and realized_bps is not None:
            error_bps = float(realized_bps - pred_bps)
            abs_error = abs(error_bps)
            pred_sign = _sign(pred_bps)
            real_sign = _sign(realized_bps)
            if pred_sign != 0 and real_sign != 0:
                directional_match = bool(pred_sign == real_sign)
                directional_hits.append(1.0 if directional_match else 0.0)
            matched_pred_bps.append(float(pred_bps))
            matched_real_bps.append(float(realized_bps))
            pred_error_bps.append(float(error_bps))
            abs_pred_error_bps.append(float(abs_error))

        if adverse_entry_slippage_bps is not None and fill_match_allowed:
            adverse_entry_slippage_bps_values.append(float(adverse_entry_slippage_bps))

        if has_ack and has_fill and fill_match_allowed:
            reconcile_status = "ack_fill_matched"
        elif has_fill and not fill_match_allowed:
            reconcile_status = "fill_present_but_ack_required_missing"
        elif has_ack and not has_fill:
            reconcile_status = "ack_no_fill"
        elif (not has_ack) and has_fill:
            reconcile_status = "fill_no_ack"
        else:
            reconcile_status = "dispatch_only"

        detail_rows.append(
            {
                "signal_id": signal_id,
                "reconcile_status": reconcile_status,
                "dispatch_ts_utc": d.get("ts_utc"),
                "dispatch_pair": instr.get("pair"),
                "dispatch_symbol": instr.get("symbol"),
                "dispatch_side": dispatch_side,
                "dispatch_price": dispatch_price,
                "dispatch_iteration": d.get("iteration"),
                "ack_status": ack_status,
                "ack_mode": ack.get("mode") if ack else None,
                "ack_http_status": ack.get("http_status") if ack else None,
                "fill_status": fill.get("status") if fill else None,
                "fill_ts_utc": fill.get("fill_ts_utc") if fill else None,
                "fill_pair": fill.get("pair") if fill else None,
                "fill_side": fill.get("side") if fill else None,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_abs": _as_float(fill.get("pnl_abs") if fill else None),
                "pnl_pct": realized_pct,
                "pred_fwd_return": pred_fwd_return,
                "predicted_return_bps": pred_bps,
                "realized_return_bps": realized_bps,
                "pred_error_bps": error_bps,
                "abs_pred_error_bps": abs_error,
                "directional_match": directional_match,
                "confidence": _as_float(pred.get("confidence")),
                "score": _as_float(pred.get("score")),
                "score_mode": pred.get("score_mode"),
                "net_effective_predicted_edge_bps": _as_float(pred.get("net_effective_predicted_edge_bps")),
                "trade_head_calibration_loaded": bool(pred.get("trade_head_calibration_loaded", False)),
                "risk_tier": risk.get("risk_tier"),
                "veto_overridden": bool(risk.get("veto_overridden", False)),
                "adverse_entry_slippage_bps": adverse_entry_slippage_bps,
            }
        )

    # Orphans not present in dispatch fidelity log.
    for signal_id in sorted(ack_ids - dispatch_ids):
        ack = ack_latest[signal_id]
        detail_rows.append(
            {
                "signal_id": signal_id,
                "reconcile_status": "orphan_ack",
                "ack_status": ack.get("status"),
                "ack_mode": ack.get("mode"),
                "ack_http_status": ack.get("http_status"),
            }
        )
    for signal_id in sorted(fill_ids - dispatch_ids):
        fill = fill_latest[signal_id]
        detail_rows.append(
            {
                "signal_id": signal_id,
                "reconcile_status": "orphan_fill",
                "fill_status": fill.get("status"),
                "fill_pair": fill.get("pair"),
                "fill_side": fill.get("side"),
                "entry_price": fill.get("entry_price"),
                "exit_price": fill.get("exit_price"),
                "pnl_abs": fill.get("pnl_abs"),
                "pnl_pct": fill.get("pnl_pct"),
            }
        )

    dispatch_with_ack = sum(1 for r in detail_rows if r.get("reconcile_status") in {"ack_fill_matched", "ack_no_fill"})
    dispatch_with_fill = sum(
        1
        for r in detail_rows
        if r.get("reconcile_status")
        in {"ack_fill_matched", "fill_no_ack", "fill_present_but_ack_required_missing"}
    )
    fully_reconciled = sum(1 for r in detail_rows if r.get("reconcile_status") == "ack_fill_matched")
    orphan_acks = sum(1 for r in detail_rows if r.get("reconcile_status") == "orphan_ack")
    orphan_fills = sum(1 for r in detail_rows if r.get("reconcile_status") == "orphan_fill")
    closed_or_open_matches = [
        r
        for r in detail_rows
        if r.get("reconcile_status") == "ack_fill_matched" and r.get("realized_return_bps") is not None
    ]

    summary = {
        "dispatch_total": int(len(dispatch_valid)),
        "dispatch_with_ack": int(dispatch_with_ack),
        "dispatch_with_fill": int(dispatch_with_fill),
        "dispatch_fully_reconciled": int(fully_reconciled),
        "orphan_ack_count": int(orphan_acks),
        "orphan_fill_count": int(orphan_fills),
        "records_with_realized_return": int(len(closed_or_open_matches)),
        "require_success_ack_for_fill_match": bool(require_success_ack_for_fill_match),
        "parse_errors": {
            "dispatch_log": int(dispatch_parse_errors),
            "ack_log": int(ack_parse_errors),
            "fill_log": int(fill_parse_errors),
        },
        "fidelity_metrics": {
            "mean_abs_pred_error_bps": float(fmean(abs_pred_error_bps)) if abs_pred_error_bps else None,
            "rmse_pred_error_bps": _rmse(pred_error_bps),
            "directional_accuracy": float(fmean(directional_hits)) if directional_hits else None,
            "pearson_pred_vs_realized_bps": _pearson(matched_pred_bps, matched_real_bps),
            "mean_adverse_entry_slippage_bps": float(fmean(adverse_entry_slippage_bps_values))
            if adverse_entry_slippage_bps_values
            else None,
        },
    }

    detail_rows.sort(
        key=lambda r: (
            str(r.get("reconcile_status", "")),
            str(r.get("signal_id", "")),
        )
    )

    return {
        "schema": "moneyfan.hrm.freqtrade.fidelity_reconciliation.v1",
        "generated_at_utc": utc_now_iso(),
        "summary": summary,
        "records": detail_rows,
    }


def reconcile_from_paths(
    dispatch_log_path: Path,
    ack_log_path: Path,
    fill_log_path: Path,
    require_success_ack_for_fill_match: bool = False,
    exchange_target: str = "",
    data_source: str = "",
) -> Dict[str, Any]:
    dispatch_rows = read_jsonl(dispatch_log_path)
    ack_rows = read_jsonl(ack_log_path)
    fill_rows = read_jsonl(fill_log_path)
    report = reconcile_fidelity_logs(
        dispatch_rows=dispatch_rows,
        ack_rows=ack_rows,
        fill_rows=fill_rows,
        require_success_ack_for_fill_match=require_success_ack_for_fill_match,
    )
    report["inputs"] = {
        "dispatch_log_path": str(dispatch_log_path),
        "ack_log_path": str(ack_log_path),
        "fill_log_path": str(fill_log_path),
        "exchange_target": str(exchange_target or ""),
        "data_source": str(data_source or ""),
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconcile HRM fidelity dispatch predictions against Freqtrade fill events")
    p.add_argument("--dispatch-log-path", type=str, default="runtime/hrm_fidelity_dispatch.jsonl",
                   help="HRM fidelity dispatch JSONL from run.py offload mode")
    p.add_argument("--ack-log-path", type=str, default="runtime/freqtrade_dispatch_ack.jsonl",
                   help="Bridge ack JSONL from freqtrade_handoff_bridge.py")
    p.add_argument("--fill-log-path", type=str, required=True,
                   help="External fill/trade JSONL keyed by signal_id")
    p.add_argument("--out-json", type=str, default="runtime/hrm_freqtrade_fidelity_reconciliation.json",
                   help="Output JSON reconciliation report")
    p.add_argument("--out-csv", type=str, default="runtime/hrm_freqtrade_fidelity_reconciliation.csv",
                   help="Output CSV detail rows")
    p.add_argument("--require-success-ack-for-fill-match", action="store_true",
                   help="Only treat fills as matched when a success ack exists for the same signal_id")
    p.add_argument("--exchange-target", type=str, default="",
                   help="Target execution venue / exchange abstraction label (e.g. coinbase_advanced)")
    p.add_argument("--data-source", type=str, default="",
                   help="Primary data source label for this evaluation context (e.g. binance)")
    p.add_argument("--print-summary", action="store_true",
                   help="Print compact summary after writing outputs")
    return p


def run_cli(args: argparse.Namespace) -> int:
    report = reconcile_from_paths(
        dispatch_log_path=Path(args.dispatch_log_path),
        ack_log_path=Path(args.ack_log_path),
        fill_log_path=Path(args.fill_log_path),
        require_success_ack_for_fill_match=bool(args.require_success_ack_for_fill_match),
        exchange_target=str(args.exchange_target or ""),
        data_source=str(args.data_source or ""),
    )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    write_json(out_json, report)
    write_csv(out_csv, list(report.get("records", [])))

    if bool(args.print_summary):
        s = report.get("summary", {})
        fm = s.get("fidelity_metrics", {}) if isinstance(s.get("fidelity_metrics"), dict) else {}
        print(
            "📊 HRM/Freqtrade fidelity reconciliation "
            f"dispatch={s.get('dispatch_total', 0)} "
            f"matched={s.get('dispatch_fully_reconciled', 0)} "
            f"with_realized={s.get('records_with_realized_return', 0)} "
            f"MAE={fm.get('mean_abs_pred_error_bps')}bps "
            f"DirAcc={fm.get('directional_accuracy')}"
        )
        print(f"📝 JSON: {out_json}")
        print(f"🧾 CSV: {out_csv}")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
