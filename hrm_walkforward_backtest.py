#!/usr/bin/env python3
"""
Walk-forward HRM backtest (checkpoint-driven, fee/slippage-aware).

Uses the trained HRM artifacts emitted by train.py:
  - *_weights.npz
  - *_model_config.json
  - *_feature_schema.json

Simulates one-bar-ahead trades on a common timestamp grid across symbols using the
HRM trade heads (return/conviction/SL/TP/size), optional mechanical veto, and
ranked top-k selection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import run as runner
from hrm.order_intent import NormalizedTradeIntent, RiskTier

mx = runner.mx


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _iso_now() -> str:
    return datetime.now().isoformat()


def _annualization_factor_from_timestamps(timestamps: List[pd.Timestamp]) -> float:
    # Best-effort inference; fall back to 5m crypto cadence.
    if len(timestamps) < 3:
        return float(365 * 24 * 12)
    diffs = []
    for a, b in zip(timestamps[:-1], timestamps[1:]):
        if pd.isna(a) or pd.isna(b):
            continue
        dt = (b - a).total_seconds()
        if dt > 0:
            diffs.append(dt)
    if not diffs:
        return float(365 * 24 * 12)
    median_sec = float(np.median(diffs))
    if median_sec <= 0:
        return float(365 * 24 * 12)
    return float((365.0 * 24.0 * 3600.0) / median_sec)


def _compute_max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (x - peak) / max(peak, 1e-12)
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _risk_tier_size_cap(risk_tier: RiskTier) -> float:
    return {
        RiskTier.NORMAL: 1.00,
        RiskTier.CAUTION: 0.60,
        RiskTier.PROTECTIVE: 0.35,
    }.get(risk_tier, 1.00)


class HRMWalkForwardBacktester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.symbols = [runner.TradingConfig.normalize_symbol_for_data(s) for s in args.symbols]
        self.config = runner.TradingConfig(
            mode="paper",
            capital=float(args.initial_capital),
            symbols=self.symbols,
            seq_len=int(args.seq_len),
            feature_lookback_bars=max(int(args.max_bars) if args.max_bars > 0 else 2048, int(args.seq_len) + 64),
            signal_threshold=float(args.signal_threshold),
            top_k=int(args.top_k),
            online_pretrain_steps=int(args.online_pretrain_steps),
            sleep_seconds=0.0,
            max_iterations=1,
            use_mechanical_veto=bool(args.use_mechanical_veto),
            weights_path=(args.weights if args.weights else None),
            veto_confidence_override_threshold=float(args.veto_confidence_override_threshold),
            veto_confidence_override_size_scale=float(args.veto_confidence_override_size_scale),
            veto_confidence_override_reasons=str(args.veto_confidence_override_reasons or ""),
            repair_risk_heads=not bool(args.no_risk_head_repair),
            repair_min_stop_loss_pct=float(args.repair_min_stop_loss_pct),
            repair_max_stop_loss_pct=float(args.repair_max_stop_loss_pct),
            repair_min_take_profit_pct=float(args.repair_min_take_profit_pct),
            trade_head_calibration_path=(args.trade_head_calibration if args.trade_head_calibration else None),
            use_trade_head_calibration=not bool(args.no_trade_head_calibration),
        )
        self.engine = runner.TradingEngine(self.config)
        self.results_dir = self._build_results_dir(args.out_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.equity = float(args.initial_capital)
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.bar_returns: List[float] = []
        self._cum_commission = 0.0
        self._cum_slippage = 0.0
        self.candidate_stats: Dict[str, Any] = {
            "candidate_rows": 0,
            "raw_vetoed_candidates": 0,
            "raw_vetoed_at_signal_threshold": 0,
            "veto_overridden_candidates": 0,
            "raw_veto_counterfactual_topk_candidates": 0,
            "raw_veto_counterfactual_topk_reason_counts": {},
            "raw_veto_displaced_topk_slots": 0,
            "raw_veto_displaced_steps": 0,
            "raw_veto_displaced_shadow_trades": 0,
            "raw_veto_displaced_reason_counts": {},
            "raw_veto_displaced_shadow_pnl": 0.0,
            "raw_veto_displaced_shadow_pnl_by_reason": {},
            "raw_veto_displaced_shadow_gross_profit": 0.0,
            "raw_veto_displaced_shadow_gross_profit_by_reason": {},
            "raw_veto_displaced_shadow_gross_loss": 0.0,
            "raw_veto_displaced_shadow_gross_loss_by_reason": {},
            "raw_veto_counterfactual_local_pnl_delta": 0.0,
            "risk_heads_repaired_candidates": 0,
            "risk_head_repair_tag_counts": {},
            "raw_veto_reason_counts": {},
        }

        self.data_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.common_timestamps: List[pd.Timestamp] = []

    def _build_results_dir(self, out_dir: str) -> Path:
        if out_dir:
            return Path(out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sym_part = "-".join(self.symbols[:3])
        return Path("walkforward_results") / f"hrm_{sym_part}_{stamp}"

    def _load_symbol_data(self, symbol: str) -> Dict[str, Any]:
        df = self.engine.pipeline.load_candles([symbol], None, None)
        if df.empty:
            raise RuntimeError(f"No candles for {symbol}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        if self.args.start:
            df = df[df["timestamp"] >= self.args.start]
        if self.args.end:
            df = df[df["timestamp"] <= self.args.end]
        df = df.reset_index(drop=True)
        if self.args.max_bars > 0 and len(df) > self.args.max_bars:
            df = df.iloc[-int(self.args.max_bars):].reset_index(drop=True)

        if len(df) < max(80, self.args.seq_len + 2):
            raise RuntimeError(f"Insufficient bars for {symbol}: {len(df)}")

        features = self.engine.pipeline.compute_signals(df, self.engine.model_config.n_signals)
        self.engine._ensure_model_input_dim(int(features.shape[1]))

        row_timestamps = [pd.Timestamp(ts) for ts in self.engine.pipeline.last_feature_timestamps]
        row_symbols = list(self.engine.pipeline.last_feature_symbols)
        if len(row_timestamps) != len(features):
            # Fallback to df timestamps if pipeline metadata not present.
            row_timestamps = [pd.Timestamp(ts) for ts in df["timestamp"].astype(str).tolist()]

        close_return_channel = int(self.engine.model_config.n_signals)
        log_returns = features[:, close_return_channel].astype(np.float64)
        closes = df["close"].astype(float).to_numpy()

        ts_to_idx: Dict[pd.Timestamp, int] = {}
        for i, ts in enumerate(row_timestamps):
            ts_to_idx[ts] = i

        return {
            "df": df,
            "features": features.astype(np.float32),
            "timestamps": row_timestamps,
            "row_symbols": row_symbols,
            "ts_to_idx": ts_to_idx,
            "closes": closes,
            "log_returns": log_returns,
            "memory": None,
        }

    def prepare(self):
        for symbol in self.symbols:
            self.data_by_symbol[symbol] = self._load_symbol_data(symbol)

        timestamp_sets = [set(d["ts_to_idx"].keys()) for d in self.data_by_symbol.values()]
        common = set.intersection(*timestamp_sets) if timestamp_sets else set()
        common_sorted = sorted(common)

        # Require next-bar availability for every symbol used in the timestamp.
        filtered: List[pd.Timestamp] = []
        min_hist = max(int(self.args.min_history), int(self.args.seq_len))
        for ts in common_sorted:
            ok = True
            for symbol, d in self.data_by_symbol.items():
                idx = d["ts_to_idx"][ts]
                if idx < (min_hist - 1):
                    ok = False
                    break
                if idx + 1 >= len(d["features"]):
                    ok = False
                    break
            if ok:
                filtered.append(ts)

        if self.args.max_steps > 0 and len(filtered) > self.args.max_steps:
            filtered = filtered[-int(self.args.max_steps):]

        if not filtered:
            raise RuntimeError("No common timestamps with sufficient history across symbols")

        self.common_timestamps = filtered

    def _score_signal(self, intent: NormalizedTradeIntent, ignore_veto: bool = False) -> float:
        if (not ignore_veto) and intent.vetoed:
            return 0.0
        if float(intent.direction) == 0.0:
            return 0.0
        if not self._passes_edge_gate(intent):
            return 0.0
        return float(max(0.0, self.engine._net_effective_predicted_edge_bps(intent)))

    def _predicted_move_bps(self, intent: NormalizedTradeIntent) -> float:
        return float(self.engine._predicted_move_bps(intent))

    def _calibrated_predicted_move_bps(self, intent: NormalizedTradeIntent) -> float:
        return float(self.engine._calibrated_predicted_move_bps(intent))

    def _predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        return float(self.engine._predicted_edge_bps(intent))

    def _calibrated_predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        return float(self.engine._calibrated_predicted_edge_bps(intent))

    def _roundtrip_cost_bps(self) -> float:
        return float((self.args.commission_bps + self.args.slippage_bps) * 2.0)

    def _passes_edge_gate(self, intent: NormalizedTradeIntent) -> bool:
        pred_move_bps = self.engine._effective_predicted_move_bps(intent)
        pred_edge_bps = self.engine._effective_predicted_edge_bps(intent)
        if pred_move_bps < float(self.args.min_pred_move_bps):
            return False
        if pred_edge_bps < (float(self.args.edge_cost_multiplier) * self._roundtrip_cost_bps()):
            return False
        return True

    def _bar_candidates(self, ts: pd.Timestamp) -> List[Dict[str, Any]]:
        drawdown_pct = 0.0
        if self.equity_curve:
            running_equity = [float(p["equity"]) for p in self.equity_curve]
            peak = max(running_equity) if running_equity else self.equity
            drawdown_pct = (self.equity - peak) / max(peak, 1e-12)

        out: List[Dict[str, Any]] = []
        for symbol in self.symbols:
            d = self.data_by_symbol[symbol]
            idx = d["ts_to_idx"][ts]
            start_idx = max(0, idx - self.args.seq_len + 1)
            seq = d["features"][start_idx: idx + 1]
            if len(seq) < self.args.min_history:
                continue

            batch_np = seq.reshape(1, len(seq), -1).astype(np.float32)
            batch_mx = mx.array(batch_np)
            memory = d["memory"] if self.args.carry_memory else None

            try:
                for _ in range(int(self.args.online_pretrain_steps)):
                    _, memory = self.engine.hrm_trainer.pretrain_step(batch_mx, memory=memory)
                output_mx, memory = self.engine.hrm_trainer.model.forward(batch_mx, memory=memory, mode="trade")
                mx.eval(output_mx)
                if self.args.carry_memory:
                    d["memory"] = memory
                output_np = np.array(output_mx[0, :], dtype=np.float32)
            except Exception as e:
                continue

            raw_base_intent = self.engine.hrm_runtime._build_trade_intent(symbol, output_np)
            base_intent, repair_meta = self.engine._repair_trade_intent_risk_heads(raw_base_intent)
            veto_applied_intent = self.engine._apply_veto(base_intent, drawdown_pct)
            intent, veto_meta = self.engine._maybe_override_veto_with_confidence(veto_applied_intent)

            raw_move = float(d["log_returns"][idx + 1])
            current_price = _safe_float(d["closes"][idx], 0.0)
            next_price = _safe_float(d["closes"][idx + 1], current_price)
            score = self._score_signal(intent)
            score_no_raw_veto = self._score_signal(intent, ignore_veto=True)
            sl = max(abs(_safe_float(intent.stop_loss_pct)), 1e-6)
            tp = max(_safe_float(intent.take_profit_pct), 0.0)
            rr = tp / sl
            pred_move_bps = self._predicted_move_bps(intent)
            pred_edge_bps = self._predicted_edge_bps(intent)
            calibrated_pred_move_bps = self._calibrated_predicted_move_bps(intent)
            calibrated_pred_edge_bps = self._calibrated_predicted_edge_bps(intent)
            move_calibration_scale = float(self.engine._move_calibration_scale(intent))
            legacy_score = (
                abs(_safe_float(intent.pred_fwd_return))
                * max(_safe_float(intent.confidence), 0.0)
                * max(rr, 0.25)
                * max(_safe_float(intent.position_fraction), 0.05)
                * max(move_calibration_scale, 0.0)
            )
            net_effective_pred_edge_bps = float(self.engine._net_effective_predicted_edge_bps(intent))

            out.append(
                {
                    "timestamp": str(ts),
                    "symbol": symbol,
                    "idx": int(idx),
                    "intent": intent,
                    "score": score,
                    "score_no_raw_veto": score_no_raw_veto,
                    "legacy_score": float(legacy_score),
                    "score_mode": "net_effective_predicted_edge_bps",
                    "raw_move": raw_move,
                    "current_price": current_price,
                    "next_price": next_price,
                    "pred_move_bps": pred_move_bps,
                    "pred_edge_bps": pred_edge_bps,
                    "calibrated_pred_move_bps": calibrated_pred_move_bps,
                    "calibrated_pred_edge_bps": calibrated_pred_edge_bps,
                    "effective_pred_move_bps": self.engine._effective_predicted_move_bps(intent),
                    "effective_pred_edge_bps": self.engine._effective_predicted_edge_bps(intent),
                    "net_effective_pred_edge_bps": net_effective_pred_edge_bps,
                    "move_calibration_scale": move_calibration_scale,
                    "roundtrip_cost_bps": self._roundtrip_cost_bps(),
                    "passes_edge_gate": self._passes_edge_gate(intent),
                    "risk_heads_repaired": bool(repair_meta.get("risk_heads_repaired", False)),
                    "risk_head_repair_tags": list(repair_meta.get("risk_head_repair_tags", [])),
                    "raw_stop_loss_pct": float(repair_meta.get("raw_stop_loss_pct", raw_base_intent.stop_loss_pct)),
                    "raw_take_profit_pct": float(repair_meta.get("raw_take_profit_pct", raw_base_intent.take_profit_pct)),
                    "repaired_stop_loss_pct": float(repair_meta.get("repaired_stop_loss_pct", base_intent.stop_loss_pct)),
                    "repaired_take_profit_pct": float(repair_meta.get("repaired_take_profit_pct", base_intent.take_profit_pct)),
                    "raw_vetoed": bool(veto_meta.get("raw_vetoed", False)),
                    "raw_veto_reason": veto_meta.get("raw_veto_reason"),
                    "veto_overridden": bool(veto_meta.get("veto_overridden", False)),
                    "veto_override_trigger": veto_meta.get("veto_override_trigger"),
                    "veto_override_confidence_threshold": float(veto_meta.get("veto_override_confidence_threshold", -1.0)),
                    "veto_override_size_scale": float(veto_meta.get("veto_override_size_scale", 0.0)),
                    "veto_override_reason_allowed": bool(veto_meta.get("veto_override_reason_allowed", False)),
                    "veto_override_reason_filter": list(veto_meta.get("veto_override_reason_filter", [])),
                }
            )
            self.candidate_stats["candidate_rows"] = int(self.candidate_stats.get("candidate_rows", 0)) + 1
            if bool(repair_meta.get("risk_heads_repaired", False)):
                self.candidate_stats["risk_heads_repaired_candidates"] = int(
                    self.candidate_stats.get("risk_heads_repaired_candidates", 0)
                ) + 1
                repair_tag_counts = self.candidate_stats.setdefault("risk_head_repair_tag_counts", {})
                for tag in list(repair_meta.get("risk_head_repair_tags", []) or []):
                    tag_key = str(tag)
                    repair_tag_counts[tag_key] = int(repair_tag_counts.get(tag_key, 0)) + 1
            if bool(veto_meta.get("raw_vetoed", False)):
                self.candidate_stats["raw_vetoed_candidates"] = int(self.candidate_stats.get("raw_vetoed_candidates", 0)) + 1
                if _safe_float(intent.confidence) >= float(self.args.signal_threshold):
                    self.candidate_stats["raw_vetoed_at_signal_threshold"] = int(
                        self.candidate_stats.get("raw_vetoed_at_signal_threshold", 0)
                    ) + 1
                reason = str(veto_meta.get("raw_veto_reason") or "unknown")
                reason_counts = self.candidate_stats.setdefault("raw_veto_reason_counts", {})
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if bool(veto_meta.get("veto_overridden", False)):
                self.candidate_stats["veto_overridden_candidates"] = int(
                    self.candidate_stats.get("veto_overridden_candidates", 0)
                ) + 1
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def _trade_outcome(
        self,
        candidate: Dict[str, Any],
        weight: float,
        equity_before: float,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        intent: NormalizedTradeIntent = candidate["intent"]
        symbol = str(candidate["symbol"])
        d = self.data_by_symbol[symbol]
        sl = max(abs(_safe_float(intent.stop_loss_pct)), 1e-4)
        tp = max(_safe_float(intent.take_profit_pct), 1e-4)
        rr = tp / sl
        tier_cap = _risk_tier_size_cap(intent.risk_tier)
        pos_frac = min(max(_safe_float(intent.position_fraction), 0.0), tier_cap)
        exposure = pos_frac * max(_safe_float(intent.confidence), 0.0)

        entry_idx = int(candidate["idx"])
        hold_bars = max(1, int(self.args.hold_bars))
        last_idx = len(d["log_returns"]) - 1
        exit_idx = min(last_idx, entry_idx + hold_bars)

        cum_log_move = 0.0
        realized_signed_move: Optional[float] = None
        for j in range(entry_idx + 1, exit_idx + 1):
            cum_log_move += _safe_float(d["log_returns"][j], 0.0)
            signed_path_move = _safe_float(intent.direction) * cum_log_move
            if signed_path_move <= -sl:
                realized_signed_move = -sl
                exit_idx = j
                break
            if signed_path_move >= tp:
                realized_signed_move = tp
                exit_idx = j
                break

        if realized_signed_move is None:
            signed_path_move = _safe_float(intent.direction) * cum_log_move
            realized_signed_move = min(max(signed_path_move, -sl), tp)

        clamped_signed_move = float(realized_signed_move)
        gross_ret = exposure * clamped_signed_move

        commission_rate = (float(self.args.commission_bps) / 10000.0) * 2.0
        slippage_rate = (float(self.args.slippage_bps) / 10000.0) * 2.0
        commission_ret = exposure * commission_rate
        slippage_ret = exposure * slippage_rate
        net_ret = gross_ret - commission_ret - slippage_ret

        pnl = equity_before * weight * net_ret
        commission_cash = equity_before * weight * commission_ret
        slippage_cash = equity_before * weight * slippage_ret

        trade_row = {
            "timestamp": candidate["timestamp"],
            "symbol": symbol,
            "direction": float(intent.direction),
            "confidence": _safe_float(intent.confidence),
            "pred_fwd_return": _safe_float(intent.pred_fwd_return),
            "position_fraction": _safe_float(intent.position_fraction),
            "risk_tier": intent.risk_tier.value,
            "vetoed": bool(intent.vetoed),
            "veto_reason": intent.veto_reason,
            "raw_vetoed": bool(candidate.get("raw_vetoed", False)),
            "raw_veto_reason": candidate.get("raw_veto_reason"),
            "veto_overridden": bool(candidate.get("veto_overridden", False)),
            "veto_override_trigger": candidate.get("veto_override_trigger"),
            "veto_override_confidence_threshold": _safe_float(candidate.get("veto_override_confidence_threshold"), -1.0),
            "veto_override_size_scale": _safe_float(candidate.get("veto_override_size_scale"), 0.0),
            "veto_override_reason_allowed": bool(candidate.get("veto_override_reason_allowed", False)),
            "veto_override_reason_filter": list(candidate.get("veto_override_reason_filter", [])),
            "risk_heads_repaired": bool(candidate.get("risk_heads_repaired", False)),
            "risk_head_repair_tags": list(candidate.get("risk_head_repair_tags", [])),
            "raw_stop_loss_pct": _safe_float(candidate.get("raw_stop_loss_pct"), 0.0),
            "raw_take_profit_pct": _safe_float(candidate.get("raw_take_profit_pct"), 0.0),
            "repaired_stop_loss_pct": _safe_float(candidate.get("repaired_stop_loss_pct"), 0.0),
            "repaired_take_profit_pct": _safe_float(candidate.get("repaired_take_profit_pct"), 0.0),
            "score": _safe_float(candidate["score"]),
            "legacy_score": _safe_float(candidate.get("legacy_score"), 0.0),
            "score_mode": candidate.get("score_mode", "unknown"),
            "entry_price": _safe_float(candidate["current_price"]),
            "exit_price": _safe_float(d["closes"][exit_idx], _safe_float(candidate["next_price"])),
            "raw_move_log": float(cum_log_move),
            "stop_loss_pct": sl,
            "take_profit_pct": tp,
            "rr": rr,
            "predicted_move_bps": _safe_float(candidate.get("pred_move_bps"), 0.0),
            "predicted_edge_bps": _safe_float(candidate.get("pred_edge_bps"), 0.0),
            "calibrated_predicted_move_bps": _safe_float(candidate.get("calibrated_pred_move_bps"), 0.0),
            "calibrated_predicted_edge_bps": _safe_float(candidate.get("calibrated_pred_edge_bps"), 0.0),
            "effective_predicted_move_bps": _safe_float(candidate.get("effective_pred_move_bps"), 0.0),
            "effective_predicted_edge_bps": _safe_float(candidate.get("effective_pred_edge_bps"), 0.0),
            "net_effective_predicted_edge_bps": _safe_float(candidate.get("net_effective_pred_edge_bps"), 0.0),
            "move_calibration_scale": _safe_float(candidate.get("move_calibration_scale"), 1.0),
            "roundtrip_cost_bps": _safe_float(candidate.get("roundtrip_cost_bps"), 0.0),
            "hold_bars_config": hold_bars,
            "hold_bars_realized": int(max(1, exit_idx - entry_idx)),
            "exit_idx": int(exit_idx),
            "exposure": exposure,
            "gross_ret": gross_ret,
            "commission_ret": commission_ret,
            "slippage_ret": slippage_ret,
            "net_ret": net_ret,
            "capital_weight": weight,
            "pnl": pnl,
            "equity_after": float(equity_before + pnl),
        }
        return net_ret * weight, commission_cash, slippage_cash, trade_row

    def _apply_trade(self, candidate: Dict[str, Any], weight: float) -> Tuple[float, Dict[str, Any]]:
        ret_contrib, commission_cash, slippage_cash, trade_row = self._trade_outcome(
            candidate,
            weight=weight,
            equity_before=float(self.equity),
        )
        self.equity += float(trade_row["pnl"])
        self._cum_commission += float(commission_cash)
        self._cum_slippage += float(slippage_cash)
        trade_row["equity_after"] = float(self.equity)
        return ret_contrib, trade_row

    def run(self):
        self.prepare()

        max_dd_stop = float(self.args.max_drawdown_stop)
        symbol_cooldown_until_step: Dict[str, int] = {}
        cooldown_bars = int(self.args.cooldown_bars) if int(self.args.cooldown_bars) >= 0 else int(self.args.hold_bars)
        for step_i, ts in enumerate(self.common_timestamps, start=1):
            candidates = self._bar_candidates(ts)
            def _passes_common_filters(c: Dict[str, Any]) -> bool:
                return (
                    abs(_safe_float(c["intent"].direction)) > 0.0
                    and _safe_float(c["intent"].confidence) >= float(self.args.signal_threshold)
                    and bool(c.get("passes_edge_gate", True))
                    and step_i >= int(symbol_cooldown_until_step.get(str(c["symbol"]), 0))
                )

            executable = [
                c for c in candidates
                if (not c["intent"].vetoed) and _passes_common_filters(c)
            ]
            selected = executable[: int(self.args.top_k)]

            counterfactual_no_raw_veto = [c for c in candidates if _passes_common_filters(c)]
            counterfactual_no_raw_veto.sort(
                key=lambda x: float(x.get("score_no_raw_veto", x.get("score", 0.0))),
                reverse=True,
            )
            cf_selected = counterfactual_no_raw_veto[: int(self.args.top_k)]
            cf_raw_vetoed = [c for c in cf_selected if bool(c.get("raw_vetoed", False))]
            if cf_raw_vetoed:
                self.candidate_stats["raw_veto_counterfactual_topk_candidates"] = int(
                    self.candidate_stats.get("raw_veto_counterfactual_topk_candidates", 0)
                ) + int(len(cf_raw_vetoed))
                cf_reason_counts = self.candidate_stats.setdefault("raw_veto_counterfactual_topk_reason_counts", {})
                for c in cf_raw_vetoed:
                    reason = str(c.get("raw_veto_reason") or "unknown")
                    cf_reason_counts[reason] = int(cf_reason_counts.get(reason, 0)) + 1
                actual_keys = {(str(c.get("symbol")), int(c.get("idx", -1))) for c in selected}
                displaced_slots = sum(
                    1
                    for c in cf_raw_vetoed
                    if (str(c.get("symbol")), int(c.get("idx", -1))) not in actual_keys
                )
                if displaced_slots > 0:
                    self.candidate_stats["raw_veto_displaced_topk_slots"] = int(
                        self.candidate_stats.get("raw_veto_displaced_topk_slots", 0)
                    ) + int(displaced_slots)
                    self.candidate_stats["raw_veto_displaced_steps"] = int(
                        self.candidate_stats.get("raw_veto_displaced_steps", 0)
                    ) + 1

                equity_before_bar = float(self.equity)
                actual_shadow_pnl = 0.0
                cf_shadow_pnl = 0.0

                if selected:
                    actual_weight = 1.0 / float(len(selected))
                    for c in selected:
                        _, _, _, shadow_row = self._trade_outcome(c, weight=actual_weight, equity_before=equity_before_bar)
                        actual_shadow_pnl += float(shadow_row.get("pnl", 0.0))

                cf_shadow_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
                if cf_selected:
                    cf_weight = 1.0 / float(len(cf_selected))
                    for c in cf_selected:
                        _, _, _, shadow_row = self._trade_outcome(c, weight=cf_weight, equity_before=equity_before_bar)
                        cf_shadow_pnl += float(shadow_row.get("pnl", 0.0))
                        cf_key = (str(c.get("symbol")), int(c.get("idx", -1)))
                        cf_shadow_rows[cf_key] = shadow_row

                self.candidate_stats["raw_veto_counterfactual_local_pnl_delta"] = float(
                    self.candidate_stats.get("raw_veto_counterfactual_local_pnl_delta", 0.0)
                ) + float(cf_shadow_pnl - actual_shadow_pnl)

                for c in cf_raw_vetoed:
                    key = (str(c.get("symbol")), int(c.get("idx", -1)))
                    if key in actual_keys:
                        continue
                    shadow_row = cf_shadow_rows.get(key)
                    if not shadow_row:
                        continue
                    reason = str(c.get("raw_veto_reason") or "unknown")
                    pnl = float(shadow_row.get("pnl", 0.0))
                    self.candidate_stats["raw_veto_displaced_shadow_trades"] = int(
                        self.candidate_stats.get("raw_veto_displaced_shadow_trades", 0)
                    ) + 1
                    displaced_reason_counts = self.candidate_stats.setdefault("raw_veto_displaced_reason_counts", {})
                    displaced_reason_counts[reason] = int(displaced_reason_counts.get(reason, 0)) + 1
                    self.candidate_stats["raw_veto_displaced_shadow_pnl"] = float(
                        self.candidate_stats.get("raw_veto_displaced_shadow_pnl", 0.0)
                    ) + pnl
                    shadow_pnl_by_reason = self.candidate_stats.setdefault("raw_veto_displaced_shadow_pnl_by_reason", {})
                    shadow_pnl_by_reason[reason] = float(shadow_pnl_by_reason.get(reason, 0.0)) + pnl
                    if pnl >= 0.0:
                        self.candidate_stats["raw_veto_displaced_shadow_gross_profit"] = float(
                            self.candidate_stats.get("raw_veto_displaced_shadow_gross_profit", 0.0)
                        ) + pnl
                        gross_profit_by_reason = self.candidate_stats.setdefault(
                            "raw_veto_displaced_shadow_gross_profit_by_reason", {}
                        )
                        gross_profit_by_reason[reason] = float(gross_profit_by_reason.get(reason, 0.0)) + pnl
                    else:
                        self.candidate_stats["raw_veto_displaced_shadow_gross_loss"] = float(
                            self.candidate_stats.get("raw_veto_displaced_shadow_gross_loss", 0.0)
                        ) + abs(pnl)
                        gross_loss_by_reason = self.candidate_stats.setdefault(
                            "raw_veto_displaced_shadow_gross_loss_by_reason", {}
                        )
                        gross_loss_by_reason[reason] = float(gross_loss_by_reason.get(reason, 0.0)) + abs(pnl)

            bar_ret = 0.0
            if selected:
                weight = 1.0 / float(len(selected))
                for c in selected:
                    ret_contrib, trade_row = self._apply_trade(c, weight=weight)
                    bar_ret += ret_contrib
                    self.trades.append(trade_row)
                    if cooldown_bars > 0:
                        symbol_cooldown_until_step[str(c["symbol"])] = step_i + cooldown_bars

            self.bar_returns.append(float(bar_ret))
            representative = selected[0] if selected else (candidates[0] if candidates else None)
            rep_price = _safe_float(representative.get("current_price"), 0.0) if representative else 0.0
            self.equity_curve.append(
                {
                    "timestamp": str(ts),
                    "equity": float(self.equity),
                    "price": rep_price,
                    "position": float(len(selected)),
                    "selected_symbols": [c["symbol"] for c in selected],
                }
            )

            if self.args.verbose and (step_i % max(1, int(self.args.verbose_every)) == 0):
                print(
                    f"[{step_i}/{len(self.common_timestamps)}] ts={ts} equity={self.equity:.2f} "
                    f"trades={len(self.trades)} bar_ret={bar_ret:.5f}"
                )

            if max_dd_stop < 0:
                curve = [float(x["equity"]) for x in self.equity_curve]
                dd = _compute_max_drawdown(curve)
                if dd <= max_dd_stop:
                    print(f"🛑 Max drawdown stop triggered at {ts}: dd={dd:.2%}")
                    break

        metrics = self._metrics()
        self._write_outputs(metrics)
        self._print_summary(metrics)
        return metrics

    def _metrics(self) -> Dict[str, Any]:
        pnl_series = [float(t["pnl"]) for t in self.trades]
        wins = [x for x in pnl_series if x > 0]
        losses = [x for x in pnl_series if x < 0]

        equity_vals = [float(x["equity"]) for x in self.equity_curve] or [self.config.capital]
        peak_equity = max(equity_vals) if equity_vals else self.config.capital
        trough_equity = min(equity_vals) if equity_vals else self.config.capital
        max_dd = _compute_max_drawdown(equity_vals)

        bar_ret_arr = np.array(self.bar_returns, dtype=float) if self.bar_returns else np.array([], dtype=float)
        annualization = _annualization_factor_from_timestamps(self.common_timestamps)
        if len(bar_ret_arr) > 1 and float(np.std(bar_ret_arr)) > 0:
            sharpe = float((np.mean(bar_ret_arr) / (np.std(bar_ret_arr) + 1e-12)) * math.sqrt(annualization))
        else:
            sharpe = 0.0

        if losses:
            profit_factor = float(sum(wins) / max(abs(sum(losses)), 1e-12))
        else:
            profit_factor = float("inf") if wins else 0.0

        total_pnl = float(sum(pnl_series))
        final_equity = float(self.equity)
        start_equity = float(self.config.capital)
        return_pct = ((final_equity / max(start_equity, 1e-12)) - 1.0) * 100.0
        avg_bar_ret = float(np.mean(bar_ret_arr)) if len(bar_ret_arr) else 0.0
        annualized_return = ((1.0 + avg_bar_ret) ** annualization - 1.0) if avg_bar_ret > -1 else -1.0
        calmar = (annualized_return / abs(max_dd)) if max_dd < 0 else 0.0

        turnover = float(sum(abs(_safe_float(t.get("exposure"), 0.0)) * _safe_float(t.get("capital_weight"), 0.0) for t in self.trades))
        veto_override_trades = int(sum(1 for t in self.trades if bool(t.get("veto_overridden", False))))
        veto_shadow_profit = float(self.candidate_stats.get("raw_veto_displaced_shadow_gross_profit", 0.0))
        veto_shadow_loss = float(self.candidate_stats.get("raw_veto_displaced_shadow_gross_loss", 0.0))
        if veto_shadow_loss > 0:
            veto_shadow_pf: Any = float(veto_shadow_profit / veto_shadow_loss)
        elif veto_shadow_profit > 0:
            veto_shadow_pf = float("inf")
        else:
            veto_shadow_pf = 0.0
        veto_shadow_pnl_by_reason = {
            str(k): float(v)
            for k, v in dict(self.candidate_stats.get("raw_veto_displaced_shadow_pnl_by_reason", {})).items()
        }
        veto_shadow_profit_by_reason = {
            str(k): float(v)
            for k, v in dict(self.candidate_stats.get("raw_veto_displaced_shadow_gross_profit_by_reason", {})).items()
        }
        veto_shadow_loss_by_reason = {
            str(k): float(v)
            for k, v in dict(self.candidate_stats.get("raw_veto_displaced_shadow_gross_loss_by_reason", {})).items()
        }
        veto_shadow_pf_by_reason: Dict[str, Any] = {}
        for reason in sorted(set(veto_shadow_pnl_by_reason) | set(veto_shadow_profit_by_reason) | set(veto_shadow_loss_by_reason)):
            prof = float(veto_shadow_profit_by_reason.get(reason, 0.0))
            loss = float(veto_shadow_loss_by_reason.get(reason, 0.0))
            if loss > 0:
                veto_shadow_pf_by_reason[reason] = float(prof / loss)
            elif prof > 0:
                veto_shadow_pf_by_reason[reason] = float("inf")
            else:
                veto_shadow_pf_by_reason[reason] = 0.0

        return {
            "total_trades": int(len(self.trades)),
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "total_pnl": total_pnl,
            "total_commission": float(self._cum_commission),
            "total_slippage": float(self._cum_slippage),
            "max_drawdown": float(max_dd),
            "peak_equity": float(peak_equity),
            "trough_equity": float(trough_equity),
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "final_equity": final_equity,
            "return_pct": float(return_pct),
            "win_rate": float(len(wins) / max(len(self.trades), 1)),
            "annualized_return": float(annualized_return),
            "calmar_ratio": float(calmar),
            "turnover": turnover,
            "veto_override_trades": veto_override_trades,
            "candidate_rows": int(self.candidate_stats.get("candidate_rows", 0)),
            "raw_vetoed_candidates": int(self.candidate_stats.get("raw_vetoed_candidates", 0)),
            "raw_vetoed_at_signal_threshold": int(self.candidate_stats.get("raw_vetoed_at_signal_threshold", 0)),
            "veto_overridden_candidates": int(self.candidate_stats.get("veto_overridden_candidates", 0)),
            "raw_veto_counterfactual_topk_candidates": int(
                self.candidate_stats.get("raw_veto_counterfactual_topk_candidates", 0)
            ),
            "raw_veto_counterfactual_topk_reason_counts": dict(
                self.candidate_stats.get("raw_veto_counterfactual_topk_reason_counts", {})
            ),
            "raw_veto_displaced_topk_slots": int(self.candidate_stats.get("raw_veto_displaced_topk_slots", 0)),
            "raw_veto_displaced_steps": int(self.candidate_stats.get("raw_veto_displaced_steps", 0)),
            "raw_veto_displaced_shadow_trades": int(self.candidate_stats.get("raw_veto_displaced_shadow_trades", 0)),
            "raw_veto_displaced_reason_counts": dict(self.candidate_stats.get("raw_veto_displaced_reason_counts", {})),
            "raw_veto_displaced_shadow_pnl": float(self.candidate_stats.get("raw_veto_displaced_shadow_pnl", 0.0)),
            "raw_veto_displaced_shadow_pnl_by_reason": veto_shadow_pnl_by_reason,
            "raw_veto_displaced_shadow_gross_profit": veto_shadow_profit,
            "raw_veto_displaced_shadow_gross_profit_by_reason": veto_shadow_profit_by_reason,
            "raw_veto_displaced_shadow_gross_loss": veto_shadow_loss,
            "raw_veto_displaced_shadow_gross_loss_by_reason": veto_shadow_loss_by_reason,
            "raw_veto_displaced_shadow_profit_factor": veto_shadow_pf,
            "raw_veto_displaced_shadow_profit_factor_by_reason": veto_shadow_pf_by_reason,
            "raw_veto_counterfactual_local_pnl_delta": float(
                self.candidate_stats.get("raw_veto_counterfactual_local_pnl_delta", 0.0)
            ),
            "risk_heads_repaired_candidates": int(self.candidate_stats.get("risk_heads_repaired_candidates", 0)),
            "risk_head_repair_tag_counts": dict(self.candidate_stats.get("risk_head_repair_tag_counts", {})),
            "raw_veto_reason_counts": dict(self.candidate_stats.get("raw_veto_reason_counts", {})),
            "bars_evaluated": int(len(self.equity_curve)),
            "common_timestamps": int(len(self.common_timestamps)),
            "symbols": list(self.symbols),
            "seq_len": int(self.args.seq_len),
            "top_k": int(self.args.top_k),
            "score_mode": "net_effective_predicted_edge_bps",
            "hold_bars": int(self.args.hold_bars),
            "cooldown_bars": int(self.args.cooldown_bars),
            "signal_threshold": float(self.args.signal_threshold),
            "commission_bps": float(self.args.commission_bps),
            "slippage_bps": float(self.args.slippage_bps),
            "use_mechanical_veto": bool(self.args.use_mechanical_veto),
            "veto_confidence_override_threshold": float(self.args.veto_confidence_override_threshold),
            "veto_confidence_override_size_scale": float(self.args.veto_confidence_override_size_scale),
            "veto_confidence_override_reasons": str(self.args.veto_confidence_override_reasons or ""),
            "repair_risk_heads": bool(not self.args.no_risk_head_repair),
            "repair_min_stop_loss_pct": float(self.args.repair_min_stop_loss_pct),
            "repair_max_stop_loss_pct": float(self.args.repair_max_stop_loss_pct),
            "repair_min_take_profit_pct": float(self.args.repair_min_take_profit_pct),
            "trade_head_calibration_loaded": bool(self.engine.trade_head_calibrator is not None),
            "trade_head_calibration_path": (
                str(self.engine._candidate_trade_head_calibration_path)
                if getattr(self.engine, "_candidate_trade_head_calibration_path", None)
                else None
            ),
            "carry_memory": bool(self.args.carry_memory),
            "min_pred_move_bps": float(self.args.min_pred_move_bps),
            "edge_cost_multiplier": float(self.args.edge_cost_multiplier),
        }

    def _write_outputs(self, metrics: Dict[str, Any]):
        run_config = {
            "created_at": _iso_now(),
            "args": vars(self.args),
            "normalized_symbols": self.symbols,
            "runner_config": asdict(self.config),
            "weights_candidate": str(self.engine._candidate_weights_path) if self.engine._candidate_weights_path else None,
            "model_config_candidate": str(self.engine._candidate_model_config_path) if getattr(self.engine, "_candidate_model_config_path", None) else None,
            "feature_schema_candidate": str(self.engine._candidate_feature_schema_path) if getattr(self.engine, "_candidate_feature_schema_path", None) else None,
            "weights_loaded": bool(self.engine._weights_loaded),
            "weights_load_error": self.engine._weights_load_error,
        }

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(x) for x in obj]
            if isinstance(obj, float):
                if math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                if math.isnan(obj):
                    return None
            return obj

        with open(self.results_dir / "metrics.json", "w") as f:
            json.dump(_sanitize(metrics), f, indent=2)
        with open(self.results_dir / "equity_curve.json", "w") as f:
            json.dump(_sanitize(self.equity_curve), f, indent=2)
        with open(self.results_dir / "trades.json", "w") as f:
            json.dump(_sanitize(self.trades), f, indent=2)
        with open(self.results_dir / "run_config.json", "w") as f:
            json.dump(_sanitize(run_config), f, indent=2)

    def _print_summary(self, metrics: Dict[str, Any]):
        print("\nWalk-forward complete")
        print(f"Results dir: {self.results_dir}")
        print(
            f"Final equity ${metrics['final_equity']:.2f} | Return {metrics['return_pct']:.2f}% | "
            f"Sharpe {metrics['sharpe_ratio']:.2f} | MaxDD {metrics['max_drawdown']:.2%}"
        )
        print(
            f"Trades {metrics['total_trades']} | WinRate {metrics['win_rate']:.1%} | "
            f"PF {metrics['profit_factor']} | VetoOverrides {metrics.get('veto_override_trades', 0)} | "
            f"Costs ${metrics['total_commission'] + metrics['total_slippage']:.2f}"
        )
        print(
            f"Candidates {metrics.get('candidate_rows', 0)} | RawVetoed {metrics.get('raw_vetoed_candidates', 0)} | "
            f"RawVetoed@Thr {metrics.get('raw_vetoed_at_signal_threshold', 0)} | "
            f"OverrideCandidates {metrics.get('veto_overridden_candidates', 0)}"
        )
        print(
            f"TradeHeadCalib {'on' if metrics.get('trade_head_calibration_loaded') else 'off'} | "
            f"path={metrics.get('trade_head_calibration_path')}"
        )
        print(
            f"RiskRepairs {metrics.get('risk_heads_repaired_candidates', 0)} | "
            f"RepairTags {metrics.get('risk_head_repair_tag_counts', {})}"
        )
        print(
            f"VetoCFTopK {metrics.get('raw_veto_counterfactual_topk_candidates', 0)} | "
            f"VetoDisplacedSlots {metrics.get('raw_veto_displaced_topk_slots', 0)} | "
            f"VetoDisplacedSteps {metrics.get('raw_veto_displaced_steps', 0)}"
        )
        print(
            f"VetoDisplacedShadowPnL ${metrics.get('raw_veto_displaced_shadow_pnl', 0.0):.2f} | "
            f"VetoCFLocalDelta ${metrics.get('raw_veto_counterfactual_local_pnl_delta', 0.0):.2f}"
        )
        if metrics.get("raw_veto_counterfactual_topk_reason_counts"):
            print(f"VetoCFReasonCounts {metrics.get('raw_veto_counterfactual_topk_reason_counts', {})}")
        if metrics.get("raw_veto_displaced_reason_counts"):
            print(
                f"VetoShadowByReasonPnL {metrics.get('raw_veto_displaced_shadow_pnl_by_reason', {})} | "
                f"Counts {metrics.get('raw_veto_displaced_reason_counts', {})}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest a trained HRM checkpoint")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated symbols")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--min-history", type=int, default=64)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--hold-bars", type=int, default=1, help="Max holding horizon in bars for bracket exit simulation")
    p.add_argument("--cooldown-bars", type=int, default=-1, help="Per-symbol entry cooldown in bars (-1 uses hold-bars)")
    p.add_argument("--signal-threshold", type=float, default=0.65)
    p.add_argument("--commission-bps", type=float, default=5.0, help="Per-side commission in bps")
    p.add_argument("--slippage-bps", type=float, default=3.0, help="Per-side slippage in bps")
    p.add_argument("--min-pred-move-bps", type=float, default=0.0, help="Minimum model-implied move (bps) required to trade")
    p.add_argument("--edge-cost-multiplier", type=float, default=1.0, help="Require predicted move >= multiplier * roundtrip costs")
    p.add_argument("--start", type=str, default="", help="Inclusive start timestamp/date")
    p.add_argument("--end", type=str, default="", help="Inclusive end timestamp/date")
    p.add_argument("--max-bars", type=int, default=1200, help="Per-symbol tail bars to use (0 = all)")
    p.add_argument("--max-steps", type=int, default=500, help="Max common timestamps to simulate (0 = all)")
    p.add_argument("--weights", type=str, default="", help="Explicit HRM weights .npz path")
    p.add_argument("--online-pretrain-steps", type=int, default=0)
    p.add_argument("--carry-memory", action="store_true", help="Carry HRM memory across bars per symbol")
    p.add_argument("--no-mechanical-veto", action="store_true")
    p.add_argument(
        "--veto-confidence-override-threshold",
        type=float,
        default=-1.0,
        help="If >0, allow high-confidence signals to override mechanical veto (with size scaling)",
    )
    p.add_argument(
        "--veto-confidence-override-size-scale",
        type=float,
        default=0.5,
        help="Position fraction scale when a veto is overridden by confidence",
    )
    p.add_argument(
        "--veto-confidence-override-reasons",
        type=str,
        default="low_confidence,poor_risk_reward",
        help="Comma-separated veto reasons eligible for confidence override ('any' to allow all)",
    )
    p.add_argument("--no-risk-head-repair", action="store_true",
                   help="Disable conservative repair of malformed stop/target heads before veto")
    p.add_argument("--repair-min-stop-loss-pct", type=float, default=0.002,
                   help="Floor for |stop_loss_pct| before veto")
    p.add_argument("--repair-max-stop-loss-pct", type=float, default=0.15,
                   help="Cap for |stop_loss_pct| before veto")
    p.add_argument("--repair-min-take-profit-pct", type=float, default=0.003,
                   help="Floor for take_profit_pct before veto")
    p.add_argument("--trade-head-calibration", type=str, default="",
                   help="Path to trade-head calibration JSON (defaults to models/trained/hrm_trade_head_calibration.json if present)")
    p.add_argument("--no-trade-head-calibration", action="store_true",
                   help="Disable trade-head calibration even if an artifact exists")
    p.add_argument("--max-drawdown-stop", type=float, default=-0.25, help="Stop simulation if drawdown <= this (negative). Set >=0 to disable")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-every", type=int, default=50)
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")
    args.symbols = symbols
    args.use_mechanical_veto = not bool(args.no_mechanical_veto)
    args.start = args.start or None
    args.end = args.end or None
    return args


def main():
    if mx is None:
        raise SystemExit("MLX not available")
    args = parse_args()
    bt = HRMWalkForwardBacktester(args)
    bt.run()


if __name__ == "__main__":
    main()
