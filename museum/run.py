#!/usr/bin/env python3
"""
Run Live/Paper Trading
======================

Execute the HRM model in paper or live-preview mode against the local candle
store. This runner uses the HRM trade heads (expected return / conviction /
SL / TP / position size), real candle closes for pricing, and optional
mechanical veto filtering.

Examples:
    python run.py --mode paper --capital 500 --iterations 1
    python run.py --mode paper --symbols BTCUSDT,ETHUSDT,SOLUSDT --top-k 2
"""

import sys
import argparse
import json
import time
import signal
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False

from train import CandlePipeline, CandleCache, EpisodeTrainingConfig, EpochEpisodeTrainer
from hrm.order_intent import NormalizedTradeIntent, RiskTier
from hrm.trade_head_calibration import TradeHeadCalibrator, discover_trade_head_calibration_path
from execution.order_intent_adapter import (
    intent_to_coinbase_order_preview,
    intent_to_freqtrade_handoff,
    intent_to_legacy_signal,
)
from execution.guardrail_actions import (
    GuardrailAction,
    GuardrailActionMapper,
    GuardrailActionResult,
    create_guardrail_action_mapper_from_config,
)


DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]


@dataclass
class TradingConfig:
    mode: str = "paper"
    capital: float = 100.0
    broker: str = "coinbase"
    offload_execution_to_freqtrade: bool = False
    freqtrade_handoff_path: str = "runtime/freqtrade_handoff.jsonl"
    emit_hrm_fidelity_dispatch_log: bool = True
    hrm_fidelity_dispatch_log_path: str = "runtime/hrm_fidelity_dispatch.jsonl"
    symbols: Optional[List[str]] = None
    risk_per_trade: float = 0.01
    max_positions: int = 10
    stop_loss: float = 0.05
    take_profit: float = 0.10
    seq_len: int = 256
    feature_lookback_bars: int = 768
    sleep_seconds: float = 60.0
    max_iterations: Optional[int] = None
    top_k: int = 1
    signal_threshold: float = 0.65
    online_pretrain_steps: int = 0
    use_mechanical_veto: bool = True
    weights_path: Optional[str] = None
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    min_pred_move_bps: float = 0.0
    edge_cost_multiplier: float = 1.0
    max_hold_iterations: int = 12
    entry_cooldown_iterations: int = -1
    veto_confidence_override_threshold: float = -1.0
    veto_confidence_override_size_scale: float = 0.5
    veto_confidence_override_reasons: Optional[List[str]] = None
    repair_risk_heads: bool = True
    repair_min_stop_loss_pct: float = 0.002
    repair_max_stop_loss_pct: float = 0.15
    repair_min_take_profit_pct: float = 0.003
    trade_head_calibration_path: Optional[str] = None
    use_trade_head_calibration: bool = True
    state_path: str = "trading_state.json"
    resume_state: bool = True
    max_drawdown_kill_pct: float = 0.12
    max_daily_loss_pct: float = 0.03
    max_daily_loss_abs: float = 0.0
    respect_saved_halt_state: bool = True
    guardrail_enabled: bool = False
    guardrail_warn_drawdown_pct: float = 0.05
    guardrail_derisk_drawdown_pct: float = 0.08
    guardrail_halt_drawdown_pct: float = 0.12
    guardrail_confirmation_window: int = 1
    guardrail_events_log_path: str = "runtime/guardrail_events.jsonl"

    def __post_init__(self):
        raw_symbols = self.symbols or list(DEFAULT_SYMBOLS)
        self.symbols = [self.normalize_symbol_for_data(s) for s in raw_symbols]
        self.seq_len = max(16, int(self.seq_len))
        self.feature_lookback_bars = max(self.seq_len + 64, int(self.feature_lookback_bars))
        self.top_k = max(1, int(self.top_k))
        self.signal_threshold = float(max(0.0, min(1.0, self.signal_threshold)))
        self.online_pretrain_steps = max(0, int(self.online_pretrain_steps))
        self.commission_bps = max(0.0, float(self.commission_bps))
        self.slippage_bps = max(0.0, float(self.slippage_bps))
        self.min_pred_move_bps = max(0.0, float(self.min_pred_move_bps))
        self.edge_cost_multiplier = max(0.0, float(self.edge_cost_multiplier))
        self.max_hold_iterations = max(1, int(self.max_hold_iterations))
        self.entry_cooldown_iterations = int(self.entry_cooldown_iterations)
        self.veto_confidence_override_threshold = float(self.veto_confidence_override_threshold)
        self.veto_confidence_override_size_scale = float(
            max(0.0, min(1.0, self.veto_confidence_override_size_scale))
        )
        raw_override_reasons = self.veto_confidence_override_reasons
        if raw_override_reasons is None:
            reasons: List[str] = []
        elif isinstance(raw_override_reasons, str):
            reasons = [r.strip().lower() for r in raw_override_reasons.split(",") if r.strip()]
        else:
            reasons = [str(r).strip().lower() for r in raw_override_reasons if str(r).strip()]
        if any(r in {"*", "any", "all"} for r in reasons):
            reasons = ["*"]
        self.veto_confidence_override_reasons = reasons
        self.repair_risk_heads = bool(self.repair_risk_heads)
        self.repair_min_stop_loss_pct = max(1e-6, float(self.repair_min_stop_loss_pct))
        self.repair_max_stop_loss_pct = max(self.repair_min_stop_loss_pct, float(self.repair_max_stop_loss_pct))
        self.repair_min_take_profit_pct = max(1e-6, float(self.repair_min_take_profit_pct))
        self.trade_head_calibration_path = (
            str(self.trade_head_calibration_path).strip() if self.trade_head_calibration_path else None
        )
        self.use_trade_head_calibration = bool(self.use_trade_head_calibration)
        self.state_path = str(self.state_path or "trading_state.json").strip() or "trading_state.json"
        self.resume_state = bool(self.resume_state)
        self.offload_execution_to_freqtrade = bool(self.offload_execution_to_freqtrade)
        self.freqtrade_handoff_path = (
            str(self.freqtrade_handoff_path or "runtime/freqtrade_handoff.jsonl").strip()
            or "runtime/freqtrade_handoff.jsonl"
        )
        self.emit_hrm_fidelity_dispatch_log = bool(self.emit_hrm_fidelity_dispatch_log)
        self.hrm_fidelity_dispatch_log_path = (
            str(self.hrm_fidelity_dispatch_log_path or "runtime/hrm_fidelity_dispatch.jsonl").strip()
            or "runtime/hrm_fidelity_dispatch.jsonl"
        )
        self.max_drawdown_kill_pct = max(0.0, float(self.max_drawdown_kill_pct))
        self.max_daily_loss_pct = max(0.0, float(self.max_daily_loss_pct))
        self.max_daily_loss_abs = max(0.0, float(self.max_daily_loss_abs))
        self.respect_saved_halt_state = bool(self.respect_saved_halt_state)
        self.guardrail_enabled = bool(self.guardrail_enabled)
        self.guardrail_warn_drawdown_pct = max(0.0, float(self.guardrail_warn_drawdown_pct))
        self.guardrail_derisk_drawdown_pct = max(
            self.guardrail_warn_drawdown_pct, float(self.guardrail_derisk_drawdown_pct)
        )
        self.guardrail_halt_drawdown_pct = max(
            self.guardrail_derisk_drawdown_pct, float(self.guardrail_halt_drawdown_pct)
        )
        self.guardrail_confirmation_window = max(1, int(self.guardrail_confirmation_window))
        if self.max_iterations is not None:
            self.max_iterations = max(1, int(self.max_iterations))

    @staticmethod
    def normalize_symbol_for_data(symbol: str) -> str:
        s = (symbol or "").strip().upper().replace("/", "").replace("-", "")
        if not s:
            return s
        if s.endswith("USD") and not s.endswith("USDT"):
            s = f"{s[:-3]}USDT"
        return s


class TradingEngine:
    def __init__(self, config: TradingConfig):
        self.config = config

        self.cache = CandleCache()
        self.pipeline = CandlePipeline(self.cache)

        self.hrm_runtime: Optional[EpochEpisodeTrainer] = None
        self.model = None
        self.model_config = None
        self.hrm_trainer = None
        self.hrm_memory_by_symbol: Dict[str, object] = {}

        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.pnl = 0.0
        self.trades: List[Dict] = []
        self.running = False
        self.latest_prices: Dict[str, float] = {}
        self.latest_price_timestamps: Dict[str, str] = {}
        self.current_iteration: int = 0
        self.symbol_cooldown_until_iteration: Dict[str, int] = {}
        self.trade_head_calibrator: Optional[TradeHeadCalibrator] = None
        self.state_path = Path(self.config.state_path)
        self.peak_equity: float = float(self.config.capital)
        self.risk_day_utc: str = self._utc_day_key()
        self.risk_day_start_equity: float = float(self.config.capital)
        self.risk_day_realized_pnl: float = 0.0
        self.halt_reason: Optional[str] = None
        self.guardrail_state: str = "normal"
        self.guardrail_candidate_state: str = "normal"
        self.guardrail_candidate_iterations: int = 0
        self.guardrail_action_mapper: GuardrailActionMapper = (
            create_guardrail_action_mapper_from_config(config)
        )
        self._current_guardrail_action: Optional[GuardrailActionResult] = None

        self._candidate_weights_path = self._discover_weights_path(self.config.weights_path)
        self._candidate_model_config_path = self._discover_model_config_path(
            self.config.weights_path,
            self._candidate_weights_path,
        )
        self._candidate_feature_schema_path = self._discover_feature_schema_path(
            self.config.weights_path,
            self._candidate_weights_path,
        )
        self._weights_loaded = False
        self._weights_load_error: Optional[str] = None
        self._saved_model_config_applied = False
        self._saved_feature_schema_applied = False
        self._warned_untrained = False
        self._candidate_trade_head_calibration_path = self._discover_trade_head_calibration_path(
            self.config.trade_head_calibration_path
        )
        self._trade_head_calibration_loaded = False
        self._trade_head_calibration_load_error: Optional[str] = None
        self._load_state()

        self._init_hrm_runtime()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    @staticmethod
    def _utc_day_key(dt: Optional[datetime] = None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.date().isoformat()

    def _equity(self) -> float:
        return float(self.config.capital + self.pnl)

    def _ensure_risk_day_bucket(self, now_utc: Optional[datetime] = None):
        day_key = self._utc_day_key(now_utc)
        if str(self.risk_day_utc or "") == day_key:
            return
        self.risk_day_utc = day_key
        self.risk_day_start_equity = float(self._equity())
        self.risk_day_realized_pnl = 0.0
        print(f"🗓️  New UTC risk day {self.risk_day_utc}; reset daily loss counters")

    def _record_realized_pnl(self, pnl: float):
        self._ensure_risk_day_bucket()
        self.risk_day_realized_pnl = float(self.risk_day_realized_pnl) + float(pnl)
        self.peak_equity = max(float(self.peak_equity), float(self._equity()))

    def _risk_snapshot(self) -> Dict[str, float]:
        self._ensure_risk_day_bucket()
        equity = float(self._equity())
        self.peak_equity = max(float(self.peak_equity), equity)
        drawdown_pct = float(self._portfolio_drawdown_pct())
        daily_loss_abs = max(0.0, -float(self.risk_day_realized_pnl))
        daily_loss_pct = daily_loss_abs / max(float(self.risk_day_start_equity), 1e-8)
        return {
            "equity": equity,
            "peak_equity": float(self.peak_equity),
            "drawdown_pct": drawdown_pct,
            "daily_realized_pnl": float(self.risk_day_realized_pnl),
            "daily_loss_abs": float(daily_loss_abs),
            "daily_loss_pct": float(daily_loss_pct),
            "risk_day_start_equity": float(self.risk_day_start_equity),
        }

    def _trigger_halt(self, reason: str, snapshot: Optional[Dict[str, float]] = None):
        self.halt_reason = str(reason)
        self.running = False
        if snapshot is None:
            snapshot = self._risk_snapshot()
        print(
            "🛑 HARD HALT: "
            f"{reason} | equity=${snapshot.get('equity', self._equity()):.2f} "
            f"dd={snapshot.get('drawdown_pct', 0.0):.2%} "
            f"daily_loss=${snapshot.get('daily_loss_abs', 0.0):.2f} "
            f"({snapshot.get('daily_loss_pct', 0.0):.2%})"
        )
        self._save_state()

    def _check_kill_switches(self) -> bool:
        snapshot = self._risk_snapshot()
        dd_limit = float(self.config.max_drawdown_kill_pct)
        if dd_limit > 0.0 and snapshot["drawdown_pct"] <= -dd_limit:
            self._trigger_halt(f"max_drawdown_kill_pct_exceeded:{dd_limit:.4f}", snapshot=snapshot)
            return False

        daily_loss_pct_limit = float(self.config.max_daily_loss_pct)
        if daily_loss_pct_limit > 0.0 and snapshot["daily_loss_pct"] >= daily_loss_pct_limit:
            self._trigger_halt(f"max_daily_loss_pct_exceeded:{daily_loss_pct_limit:.4f}", snapshot=snapshot)
            return False

        daily_loss_abs_limit = float(self.config.max_daily_loss_abs)
        if daily_loss_abs_limit > 0.0 and snapshot["daily_loss_abs"] >= daily_loss_abs_limit:
            self._trigger_halt(f"max_daily_loss_abs_exceeded:{daily_loss_abs_limit:.4f}", snapshot=snapshot)
            return False

        if self.config.guardrail_enabled:
            state = self._check_drawdown_guardrails()
            if state == "halt":
                self._trigger_halt("guardrail_halt_triggered", snapshot=snapshot)
                return False

        return True

    def _check_drawdown_guardrails(self) -> str:
        """Evaluate current drawdown against guardrail thresholds and record state.

        States advance normal -> warn -> derisk -> halt as drawdown deepens.
        Returns the current guardrail state string.  When guardrail_enabled is
        False the method is a no-op and always returns 'normal'.
        """
        if not self.config.guardrail_enabled:
            return "normal"

        abs_dd = abs(min(self._portfolio_drawdown_pct(), 0.0))

        if abs_dd >= self.config.guardrail_halt_drawdown_pct:
            candidate = "halt"
        elif abs_dd >= self.config.guardrail_derisk_drawdown_pct:
            candidate = "derisk"
        elif abs_dd >= self.config.guardrail_warn_drawdown_pct:
            candidate = "warn"
        else:
            candidate = "normal"

        if candidate != self.guardrail_state:
            if candidate == self.guardrail_candidate_state:
                self.guardrail_candidate_iterations += 1
            else:
                self.guardrail_candidate_state = candidate
                self.guardrail_candidate_iterations = 1

            if self.guardrail_candidate_iterations >= self.config.guardrail_confirmation_window:
                old_state = self.guardrail_state
                self.guardrail_state = candidate
                self._emit_guardrail_event(old_state, candidate, abs_dd)
        else:
            # Candidate matches current state, reset candidate tracker
            self.guardrail_candidate_state = candidate
            self.guardrail_candidate_iterations = 0

        return self.guardrail_state

    def _emit_guardrail_event(self, old_state: str, new_state: str, drawdown: float):
        event = {
            "schema": "moneyfan.runtime.guardrail.event.v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_state": old_state,
            "new_state": new_state,
            "drawdown_pct": float(drawdown),
            "threshold_warn": float(self.config.guardrail_warn_drawdown_pct),
            "threshold_derisk": float(self.config.guardrail_derisk_drawdown_pct),
            "threshold_halt": float(self.config.guardrail_halt_drawdown_pct),
            "mode": str(self.config.mode),
            "iteration": int(self.current_iteration),
        }
        print(f"🛡️  GUARDRAIL TRANSITION: {old_state} -> {new_state} (dd={drawdown:.2%})")
        log_path = Path(self.config.guardrail_events_log_path)
        self._append_jsonl(log_path, event)

    def _update_guardrail_action(self):
        """Update the current guardrail action based on guardrail state.
        
        This method applies the guardrail action mapping to determine runtime
        behavior modifications based on the current guardrail state.
        """
        action_result = self.guardrail_action_mapper.apply_action(
            current_params={},
            guardrail_state=self.guardrail_state,
            top_k=self.config.top_k,
            risk_per_trade=self.config.risk_per_trade,
            signal_threshold=self.config.signal_threshold,
            max_positions=self.config.max_positions,
        )
        self.current_guardrail_action = action_result.action
        
        # Log action changes
        if action_result.action != GuardrailAction.NORMAL:
            print(
                f"🛡️  GUARDRAIL ACTION: {action_result.action.value} | "
                f"position_scale={action_result.position_size_scale:.2f} "
                f"top_k={action_result.top_k} "
                f"signal_thresh={action_result.signal_threshold:.2f} "
                f"entries={'allowed' if action_result.allow_new_entries else 'blocked'}"
            )

    def _should_allow_new_entries(self) -> tuple[bool, str]:
        """Check if new position entries should be allowed based on guardrail state.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        return self.guardrail_action_mapper.should_allow_entry(
            guardrail_state=self.guardrail_state,
            current_positions=len(self.positions),
            max_positions=self.config.max_positions,
        )

    def _get_effective_signal_threshold(self) -> float:
        """Get the effective signal threshold considering guardrail actions.
        
        Returns:
            Effective signal threshold to use for this iteration
        """
        action_result = self.guardrail_action_mapper.apply_action(
            current_params={},
            guardrail_state=self.guardrail_state,
            top_k=self.config.top_k,
            risk_per_trade=self.config.risk_per_trade,
            signal_threshold=self.config.signal_threshold,
            max_positions=self.config.max_positions,
        )
        return action_result.signal_threshold

    def _get_effective_top_k(self) -> int:
        """Get the effective top-k limit considering guardrail actions.
        
        Returns:
            Effective top-k limit for this iteration
        """
        action_result = self.guardrail_action_mapper.apply_action(
            current_params={},
            guardrail_state=self.guardrail_state,
            top_k=self.config.top_k,
            risk_per_trade=self.config.risk_per_trade,
            signal_threshold=self.config.signal_threshold,
            max_positions=self.config.max_positions,
        )
        return action_result.top_k

    def _get_effective_position_size_scale(self) -> float:
        """Get the effective position size scale considering guardrail actions.
        
        Returns:
            Position size multiplier to apply
        """
        action_result = self.guardrail_action_mapper.apply_action(
            current_params={},
            guardrail_state=self.guardrail_state,
            top_k=self.config.top_k,
            risk_per_trade=self.config.risk_per_trade,
            signal_threshold=self.config.signal_threshold,
            max_positions=self.config.max_positions,
        )
        return action_result.position_size_scale

    def _load_state(self):
        if not bool(getattr(self.config, "resume_state", True)):
            return

        path = Path(getattr(self, "state_path", self.config.state_path))
        if not path.exists() or not path.is_file():
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load state from {path}: {e}")
            return

        if not isinstance(state, dict):
            print(f"⚠️  Ignoring invalid state payload in {path}")
            return

        saved_mode = str(state.get("mode", "") or "")
        if saved_mode and saved_mode != str(self.config.mode):
            print(f"⚠️  Ignoring saved state from mode={saved_mode} (current mode={self.config.mode})")
            return

        try:
            self.pnl = float(state.get("pnl", 0.0))
        except Exception:
            self.pnl = 0.0

        self.positions = dict(state.get("positions", {}) or {})
        self.trades = list(state.get("trades", []) or [])
        self.orders = list(state.get("orders", []) or [])
        self.latest_prices = {
            str(k): float(v)
            for k, v in dict(state.get("latest_prices", {}) or {}).items()
            if k is not None
        }
        self.latest_price_timestamps = {
            str(k): str(v)
            for k, v in dict(state.get("latest_price_timestamps", {}) or {}).items()
            if k is not None and v is not None
        }
        self.current_iteration = int(state.get("current_iteration", 0) or 0)
        self.symbol_cooldown_until_iteration = {
            str(k): int(v)
            for k, v in dict(state.get("symbol_cooldown_until_iteration", {}) or {}).items()
            if k is not None
        }

        risk_state = state.get("risk_state") if isinstance(state.get("risk_state"), dict) else {}
        self.peak_equity = max(float(self.config.capital), float(risk_state.get("peak_equity", self._equity())))
        self.risk_day_utc = str(risk_state.get("risk_day_utc", self._utc_day_key()))
        self.risk_day_start_equity = float(risk_state.get("risk_day_start_equity", max(self._equity(), 1e-8)))
        self.risk_day_realized_pnl = float(risk_state.get("risk_day_realized_pnl", 0.0))
        saved_halt_reason = state.get("halt_reason") or risk_state.get("halt_reason")
        if saved_halt_reason:
            self.halt_reason = str(saved_halt_reason)
            if bool(self.config.respect_saved_halt_state):
                print(f"⚠️  Loaded prior halt state: {self.halt_reason}")

        # Restore guardrail state so confirmation window and state-machine survive a restart
        saved_guardrail_state = str(state.get("guardrail_state", "normal") or "normal")
        if saved_guardrail_state in ("normal", "warn", "derisk", "halt"):
            self.guardrail_state = saved_guardrail_state
        saved_candidate = str(state.get("guardrail_candidate_state", "normal") or "normal")
        if saved_candidate in ("normal", "warn", "derisk", "halt"):
            self.guardrail_candidate_state = saved_candidate
        try:
            self.guardrail_candidate_iterations = int(
                state.get("guardrail_candidate_iterations", 0) or 0
            )
        except (TypeError, ValueError):
            self.guardrail_candidate_iterations = 0

        self._ensure_risk_day_bucket()
        print(
            f"♻️  Restored state from {path} | pnl=${self.pnl:.2f} positions={len(self.positions)} "
            f"trades={len(self.trades)} iter={self.current_iteration} "
            f"guardrail={self.guardrail_state}"
        )

    def _init_hrm_runtime(self):
        if not HAS_MLX:
            print("❌ MLX not available")
            return

        rt_cfg = EpisodeTrainingConfig(
            n_epoch_episodes=1,
            bar_sequences_per_episode=1,
            epochs=1,
            use_mechanical_veto=self.config.use_mechanical_veto,
            optimizer_name="adamw",
            learning_rate=1e-4,
            weight_decay=1e-2,
        )
        self.hrm_runtime = EpochEpisodeTrainer(rt_cfg)
        self.pipeline = self.hrm_runtime.candle_pipeline
        self.model = self.hrm_runtime.model
        self.model_config = self.hrm_runtime.model_config
        self.hrm_trainer = self.hrm_runtime.trainer
        self._try_apply_saved_feature_schema()
        self._try_apply_saved_model_config()

        if self._candidate_weights_path is not None:
            print(f"📦 HRM checkpoint candidate: {self._candidate_weights_path}")
        else:
            print("⚠️  No HRM checkpoint found; runner will use online adaptation only")
        if self._candidate_model_config_path is not None:
            print(f"🧾 HRM config candidate: {self._candidate_model_config_path}")
        if self._candidate_feature_schema_path is not None:
            print(f"🧩 HRM feature schema candidate: {self._candidate_feature_schema_path}")
        if self._candidate_trade_head_calibration_path is not None:
            print(f"📐 Trade-head calibration candidate: {self._candidate_trade_head_calibration_path}")
        self._try_load_trade_head_calibration()

    def _discover_weights_path(self, explicit_path: Optional[str]) -> Optional[Path]:
        candidates: List[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        candidates.extend([
            Path("models/trained/hrm_latest_weights.npz"),
            Path("models/trained/hrm_weights.npz"),
            Path("hrm/checkpoints/hrm_latest_weights.npz"),
        ])
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def _discover_model_config_path(
        self,
        explicit_weights_path: Optional[str],
        resolved_weights_path: Optional[Path],
    ) -> Optional[Path]:
        candidates: List[Path] = []

        if explicit_weights_path:
            explicit = Path(explicit_weights_path)
            if explicit.name.endswith("_weights.npz"):
                candidates.append(explicit.with_name(explicit.name.replace("_weights.npz", "_model_config.json")))

        if resolved_weights_path and resolved_weights_path.name.endswith("_weights.npz"):
            candidates.append(
                resolved_weights_path.with_name(
                    resolved_weights_path.name.replace("_weights.npz", "_model_config.json")
                )
            )

        candidates.extend([
            Path("models/trained/hrm_latest_model_config.json"),
            Path("models/trained/hrm_model_config.json"),
            Path("hrm/checkpoints/hrm_latest_model_config.json"),
        ])

        seen = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists() and p.is_file():
                return p
        return None

    def _discover_feature_schema_path(
        self,
        explicit_weights_path: Optional[str],
        resolved_weights_path: Optional[Path],
    ) -> Optional[Path]:
        candidates: List[Path] = []

        if explicit_weights_path:
            explicit = Path(explicit_weights_path)
            if explicit.name.endswith("_weights.npz"):
                candidates.append(explicit.with_name(explicit.name.replace("_weights.npz", "_feature_schema.json")))

        if resolved_weights_path and resolved_weights_path.name.endswith("_weights.npz"):
            candidates.append(
                resolved_weights_path.with_name(
                    resolved_weights_path.name.replace("_weights.npz", "_feature_schema.json")
                )
            )

        candidates.extend([
            Path("models/trained/hrm_latest_feature_schema.json"),
            Path("models/trained/hrm_feature_schema.json"),
            Path("hrm/checkpoints/hrm_latest_feature_schema.json"),
        ])

        seen = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists() and p.is_file():
                return p
        return None

    def _discover_trade_head_calibration_path(self, explicit_path: Optional[str]) -> Optional[Path]:
        if not bool(self.config.use_trade_head_calibration):
            return None
        return discover_trade_head_calibration_path(explicit_path)

    def _try_apply_saved_feature_schema(self):
        if (
            self._saved_feature_schema_applied
            or self._candidate_feature_schema_path is None
            or self.pipeline is None
        ):
            return

        try:
            with open(self._candidate_feature_schema_path, "r") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to read HRM feature schema from {self._candidate_feature_schema_path}: {e}")
            return

        if not isinstance(payload, dict):
            print(f"⚠️  Invalid feature schema payload in {self._candidate_feature_schema_path}")
            return

        instrument_keys = payload.get("instrument_keys")
        context_feature_keys = payload.get("context_feature_keys")

        if isinstance(instrument_keys, list):
            self.pipeline.instrument_keys = [str(k) for k in instrument_keys]
        if isinstance(context_feature_keys, list):
            self.pipeline.context_feature_keys = [str(k) for k in context_feature_keys]

        self._saved_feature_schema_applied = True
        print(
            f"✅ Applied feature schema (instrument_keys={len(getattr(self.pipeline, 'instrument_keys', []) or [])})"
        )

    def _try_apply_saved_model_config(self):
        if (
            self._saved_model_config_applied
            or self._candidate_model_config_path is None
            or self.hrm_runtime is None
            or self.model_config is None
        ):
            return

        try:
            with open(self._candidate_model_config_path, "r") as f:
                saved_cfg = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to read HRM config from {self._candidate_model_config_path}: {e}")
            return

        if not isinstance(saved_cfg, dict):
            print(f"⚠️  Invalid HRM config payload in {self._candidate_model_config_path}")
            return

        applied = False
        for k, v in saved_cfg.items():
            if hasattr(self.model_config, k):
                try:
                    setattr(self.model_config, k, v)
                    applied = True
                except Exception:
                    continue

        known_input_dim = saved_cfg.get("input_dim")
        if applied:
            self.hrm_runtime._init_model_if_needed(
                force_reinit=True,
                known_input_dim=(int(known_input_dim) if known_input_dim is not None else None),
            )
            self.model = self.hrm_runtime.model
            self.model_config = self.hrm_runtime.model_config
            self.hrm_trainer = self.hrm_runtime.trainer
            self.hrm_memory_by_symbol.clear()
            self._saved_feature_schema_applied = False
            self._try_apply_saved_feature_schema()
            self._saved_model_config_applied = True
            print(
                f"✅ Applied saved HRM config (input_dim={getattr(self.model_config, 'input_dim', 'n/a')}, "
                f"hidden={getattr(self.model_config, 'hidden_dim', 'n/a')})"
            )

    def _try_load_weights(self):
        if not self._saved_model_config_applied:
            self._try_apply_saved_model_config()
        if self._weights_loaded or self._candidate_weights_path is None or self.model is None:
            return
        try:
            self.model.load_weights(str(self._candidate_weights_path))
            self._weights_loaded = True
            print(f"✅ Loaded HRM weights from {self._candidate_weights_path}")
        except Exception as e:
            self._weights_load_error = str(e)
            print(f"⚠️  Failed to load HRM weights from {self._candidate_weights_path}: {e}")

    def _try_load_trade_head_calibration(self):
        if not bool(self.config.use_trade_head_calibration):
            return
        if self._trade_head_calibration_loaded or self._candidate_trade_head_calibration_path is None:
            return
        try:
            self.trade_head_calibrator = TradeHeadCalibrator.load(self._candidate_trade_head_calibration_path)
            self._trade_head_calibration_loaded = True
            print(f"✅ Loaded trade-head calibration from {self._candidate_trade_head_calibration_path}")
        except Exception as e:
            self._trade_head_calibration_load_error = str(e)
            self.trade_head_calibrator = None
            print(f"⚠️  Failed to load trade-head calibration from {self._candidate_trade_head_calibration_path}: {e}")

    def _signal_handler(self, signum, frame):
        print("\n🛑 Shutting down...")
        self.running = False
        self._save_state()
        sys.exit(0)

    def _portfolio_drawdown_pct(self) -> float:
        equity = self.config.capital + self.pnl
        if equity <= 0:
            return -1.0
        historical_equity = [self.config.capital]
        running = self.config.capital
        for t in self.trades:
            running += float(t.get("pnl", 0.0))
            historical_equity.append(running)
        peak = max(historical_equity) if historical_equity else self.config.capital
        return (equity - peak) / max(peak, 1e-8)

    def _refresh_price_from_candles(self, symbol: str, df: Optional[pd.DataFrame] = None) -> Optional[float]:
        if df is None:
            df = self.pipeline.load_candles([symbol], None, None)
            if df.empty:
                return None
            df = df.sort_values("timestamp").reset_index(drop=True)

        if df.empty or "close" not in df.columns:
            return None

        price = float(df["close"].iloc[-1])
        self.latest_prices[symbol] = price
        if "timestamp" in df.columns and len(df):
            self.latest_price_timestamps[symbol] = str(df["timestamp"].iloc[-1])
        return price

    def _ensure_model_input_dim(self, feature_dim: int):
        if self.hrm_runtime is None:
            return
        current_dim = int(getattr(self.model_config, "input_dim", 0) or 0)
        if self.model is None or current_dim != int(feature_dim):
            prev = current_dim or "None"
            print(f"[Runner] Recalibrating HRM input_dim {prev} -> {feature_dim}")
            self.hrm_runtime._init_model_if_needed(force_reinit=True, known_input_dim=int(feature_dim))
            self.model = self.hrm_runtime.model
            self.model_config = self.hrm_runtime.model_config
            self.hrm_trainer = self.hrm_runtime.trainer
            self.hrm_memory_by_symbol.clear()
            self._saved_feature_schema_applied = False
            self._saved_model_config_applied = False
            self._weights_loaded = False
            self._try_apply_saved_feature_schema()
            self._try_load_weights()
        elif not self._weights_loaded:
            self._try_load_weights()

    def _apply_veto(self, base_intent: NormalizedTradeIntent, drawdown_pct: float) -> NormalizedTradeIntent:
        if self.hrm_runtime is None:
            return base_intent
        veto = self.hrm_runtime._mechanical_veto(base_intent, drawdown_pct)
        return NormalizedTradeIntent(
            symbol=base_intent.symbol,
            direction=base_intent.direction,
            pred_fwd_return=base_intent.pred_fwd_return,
            confidence=base_intent.confidence,
            position_fraction=base_intent.position_fraction,
            stop_loss_pct=base_intent.stop_loss_pct,
            take_profit_pct=base_intent.take_profit_pct,
            vetoed=bool(veto.vetoed),
            veto_reason=veto.reason,
            risk_tier=veto.risk_tier,
        )

    def _repair_trade_intent_risk_heads(
        self,
        intent: NormalizedTradeIntent,
    ) -> tuple[NormalizedTradeIntent, Dict[str, object]]:
        meta: Dict[str, object] = {
            "risk_heads_repaired": False,
            "risk_head_repair_tags": [],
            "raw_stop_loss_pct": float(intent.stop_loss_pct),
            "raw_take_profit_pct": float(intent.take_profit_pct),
            "repaired_stop_loss_pct": float(intent.stop_loss_pct),
            "repaired_take_profit_pct": float(intent.take_profit_pct),
        }
        if not bool(self.config.repair_risk_heads):
            return intent, meta

        tags: List[str] = []
        sl_abs = abs(float(intent.stop_loss_pct))
        tp_abs = max(float(intent.take_profit_pct), 0.0)

        if sl_abs < float(self.config.repair_min_stop_loss_pct):
            sl_abs = float(self.config.repair_min_stop_loss_pct)
            tags.append("stop_floor")
        if sl_abs > float(self.config.repair_max_stop_loss_pct):
            sl_abs = float(self.config.repair_max_stop_loss_pct)
            tags.append("stop_cap")
        if tp_abs < float(self.config.repair_min_take_profit_pct):
            tp_abs = float(self.config.repair_min_take_profit_pct)
            tags.append("target_floor")

        if not tags:
            return intent, meta

        repaired = NormalizedTradeIntent(
            symbol=intent.symbol,
            direction=float(intent.direction),
            pred_fwd_return=float(intent.pred_fwd_return),
            confidence=float(intent.confidence),
            position_fraction=float(intent.position_fraction),
            stop_loss_pct=-float(sl_abs),
            take_profit_pct=float(tp_abs),
            vetoed=bool(intent.vetoed),
            veto_reason=intent.veto_reason,
            risk_tier=intent.risk_tier,
        )
        meta["risk_heads_repaired"] = True
        meta["risk_head_repair_tags"] = tags
        meta["repaired_stop_loss_pct"] = float(repaired.stop_loss_pct)
        meta["repaired_take_profit_pct"] = float(repaired.take_profit_pct)
        return repaired, meta

    def _maybe_override_veto_with_confidence(
        self,
        intent: NormalizedTradeIntent,
    ) -> tuple[NormalizedTradeIntent, Dict[str, object]]:
        meta: Dict[str, object] = {
            "raw_vetoed": bool(intent.vetoed),
            "raw_veto_reason": intent.veto_reason,
            "veto_overridden": False,
            "veto_override_trigger": None,
            "veto_override_confidence_threshold": float(self.config.veto_confidence_override_threshold),
            "veto_override_size_scale": float(self.config.veto_confidence_override_size_scale),
            "veto_override_reason_allowed": False,
            "veto_override_reason_filter": list(self.config.veto_confidence_override_reasons or []),
        }
        if not bool(intent.vetoed):
            return intent, meta

        threshold = float(self.config.veto_confidence_override_threshold)
        if threshold <= 0.0:
            return intent, meta

        raw_reason = str(intent.veto_reason or "unknown").strip().lower()
        allowlist = [str(x).strip().lower() for x in (self.config.veto_confidence_override_reasons or []) if str(x).strip()]
        reason_allowed = (not allowlist) or ("*" in allowlist) or (raw_reason in allowlist)
        meta["veto_override_reason_allowed"] = bool(reason_allowed)
        if not reason_allowed:
            return intent, meta

        confidence = max(float(intent.confidence), 0.0)
        if confidence < threshold:
            return intent, meta

        scaled_position_fraction = (
            min(1.0, max(0.0, float(intent.position_fraction)))
            * float(self.config.veto_confidence_override_size_scale)
        )
        overridden_intent = NormalizedTradeIntent(
            symbol=intent.symbol,
            direction=float(intent.direction),
            pred_fwd_return=float(intent.pred_fwd_return),
            confidence=float(intent.confidence),
            position_fraction=float(max(0.0, min(1.0, scaled_position_fraction))),
            stop_loss_pct=float(intent.stop_loss_pct),
            take_profit_pct=float(intent.take_profit_pct),
            vetoed=False,
            veto_reason=intent.veto_reason,
            risk_tier=intent.risk_tier,
        )
        meta["veto_overridden"] = True
        meta["veto_override_trigger"] = f"{raw_reason}:confidence>={threshold:.2f}"
        return overridden_intent, meta

    def _roundtrip_cost_bps(self) -> float:
        return float((self.config.commission_bps + self.config.slippage_bps) * 2.0)

    def _predicted_move_bps(self, intent: NormalizedTradeIntent) -> float:
        sl = max(abs(float(intent.stop_loss_pct)), 1e-6)
        tp = max(float(intent.take_profit_pct), 1e-6)
        pred_mag = abs(float(intent.pred_fwd_return))
        move = max(min(pred_mag, tp), min(sl, tp))
        return float(move * 10000.0)

    def _move_calibration_scale(self, intent: NormalizedTradeIntent) -> float:
        if self.trade_head_calibrator is None:
            return 1.0
        try:
            return float(self.trade_head_calibrator.scale_for_confidence(float(intent.confidence)))
        except Exception:
            return 1.0

    def _calibrated_predicted_move_bps(self, intent: NormalizedTradeIntent) -> float:
        raw_move_bps = self._predicted_move_bps(intent)
        if self.trade_head_calibrator is None:
            return float(raw_move_bps)
        try:
            return float(
                self.trade_head_calibrator.calibrate_move_bps(
                    raw_move_bps,
                    float(intent.confidence),
                )
            )
        except Exception:
            return float(raw_move_bps)

    def _predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        tier_cap = {
            RiskTier.NORMAL: 1.00,
            RiskTier.CAUTION: 0.60,
            RiskTier.PROTECTIVE: 0.35,
        }.get(intent.risk_tier, 1.00)
        size = min(max(float(intent.position_fraction), 0.0), tier_cap)
        conf = max(float(intent.confidence), 0.0)
        return float(self._predicted_move_bps(intent) * size * conf)

    def _calibrated_predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        tier_cap = {
            RiskTier.NORMAL: 1.00,
            RiskTier.CAUTION: 0.60,
            RiskTier.PROTECTIVE: 0.35,
        }.get(intent.risk_tier, 1.00)
        size = min(max(float(intent.position_fraction), 0.0), tier_cap)
        conf = max(float(intent.confidence), 0.0)
        return float(self._calibrated_predicted_move_bps(intent) * size * conf)

    def _effective_predicted_move_bps(self, intent: NormalizedTradeIntent) -> float:
        if self.trade_head_calibrator is None:
            return self._predicted_move_bps(intent)
        return self._calibrated_predicted_move_bps(intent)

    def _effective_predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        if self.trade_head_calibrator is None:
            return self._predicted_edge_bps(intent)
        return self._calibrated_predicted_edge_bps(intent)

    def _net_effective_predicted_edge_bps(self, intent: NormalizedTradeIntent) -> float:
        cost_bps = float(self.config.edge_cost_multiplier) * self._roundtrip_cost_bps()
        return float(self._effective_predicted_edge_bps(intent) - cost_bps)

    def _ranking_score(self, intent: NormalizedTradeIntent) -> float:
        if bool(intent.vetoed) or float(intent.direction) == 0.0:
            return 0.0
        if not self._passes_edge_gate(intent):
            return 0.0
        return float(max(0.0, self._net_effective_predicted_edge_bps(intent)))

    def _passes_edge_gate(self, intent: NormalizedTradeIntent) -> bool:
        pred_move_bps = self._effective_predicted_move_bps(intent)
        pred_edge_bps = self._effective_predicted_edge_bps(intent)
        if pred_move_bps < float(self.config.min_pred_move_bps):
            return False
        if pred_edge_bps < (float(self.config.edge_cost_multiplier) * self._roundtrip_cost_bps()):
            return False
        return True

    def generate_signals(self, symbol: str) -> Dict:
        data_symbol = self.config.normalize_symbol_for_data(symbol)
        df = self.pipeline.load_candles([data_symbol], None, None)
        if df.empty:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "error": f"No candles for {data_symbol}",
            }

        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) > self.config.feature_lookback_bars:
            df = df.iloc[-self.config.feature_lookback_bars :].copy()

        price = self._refresh_price_from_candles(data_symbol, df)
        if price is None:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "error": "No close price",
            }

        if self.hrm_runtime is None or self.model is None or self.hrm_trainer is None or not HAS_MLX:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "price": price,
                "error": "HRM runtime unavailable (MLX missing)",
            }

        try:
            signals = self.pipeline.compute_signals(df, self.model_config.n_signals)
        except Exception as e:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "price": price,
                "error": f"feature_compute_failed: {e}",
            }

        if signals.size == 0 or len(signals) < 64:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "price": price,
                "error": "Insufficient signal bars",
            }

        feature_dim = int(signals.shape[1])
        self._ensure_model_input_dim(feature_dim)

        seq_len = min(int(self.config.seq_len), int(len(signals)))
        batch_np = signals[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)
        batch_mx = mx.array(batch_np)

        memory = self.hrm_memory_by_symbol.get(data_symbol)
        try:
            for _ in range(self.config.online_pretrain_steps):
                _, memory = self.hrm_trainer.pretrain_step(batch_mx, memory=memory)

            output_mx, memory = self.hrm_trainer.model.forward(batch_mx, memory=memory, mode="trade")
            mx.eval(output_mx)
            self.hrm_memory_by_symbol[data_symbol] = memory
            output_np = np.array(output_mx[0, :], dtype=np.float32)
        except Exception as e:
            return {
                "symbol": data_symbol,
                "signal": 0,
                "confidence": 0.0,
                "price": price,
                "error": f"inference_failed: {e}",
            }

        raw_base_intent = self.hrm_runtime._build_trade_intent(data_symbol, output_np)
        base_intent, repair_meta = self._repair_trade_intent_risk_heads(raw_base_intent)
        veto_applied_intent = self._apply_veto(base_intent, self._portfolio_drawdown_pct())
        trade_intent, veto_meta = self._maybe_override_veto_with_confidence(veto_applied_intent)

        pred = float(trade_intent.pred_fwd_return)
        confidence = float(trade_intent.confidence)
        direction = float(trade_intent.direction)
        stop_loss = float(abs(trade_intent.stop_loss_pct))
        take_profit = float(max(trade_intent.take_profit_pct, 0.0))
        position_fraction = float(max(0.0, min(1.0, trade_intent.position_fraction)))
        rr = take_profit / max(stop_loss, 1e-6)
        predicted_move_bps = self._predicted_move_bps(trade_intent)
        predicted_edge_bps = self._predicted_edge_bps(trade_intent)
        move_calibration_scale = self._move_calibration_scale(trade_intent)
        calibrated_predicted_move_bps = self._calibrated_predicted_move_bps(trade_intent)
        calibrated_predicted_edge_bps = self._calibrated_predicted_edge_bps(trade_intent)
        effective_predicted_move_bps = self._effective_predicted_move_bps(trade_intent)
        effective_predicted_edge_bps = self._effective_predicted_edge_bps(trade_intent)
        net_effective_predicted_edge_bps = self._net_effective_predicted_edge_bps(trade_intent)
        roundtrip_cost_bps = self._roundtrip_cost_bps()
        passes_edge_gate = self._passes_edge_gate(trade_intent)

        # Legacy score retained for diagnostics; ranking now uses calibrated net expected edge after costs.
        legacy_raw_score = (
            abs(pred)
            * confidence
            * max(rr, 0.25)
            * max(position_fraction, 0.05)
            * max(move_calibration_scale, 0.0)
        )
        score = self._ranking_score(trade_intent)

        if (not self._weights_loaded) and (not self._warned_untrained):
            self._warned_untrained = True
            if self._weights_load_error:
                print(f"⚠️  Running without loaded HRM checkpoint (load error: {self._weights_load_error})")
            elif self._candidate_weights_path is None:
                print("⚠️  Running without trained HRM weights (checkpoint not found)")

        return {
            "symbol": data_symbol,
            "signal": direction,
            "confidence": confidence,
            "prediction": pred,
            "position_fraction": position_fraction,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "risk_tier": trade_intent.risk_tier.value,
            "risk_heads_repaired": bool(repair_meta.get("risk_heads_repaired", False)),
            "risk_head_repair_tags": list(repair_meta.get("risk_head_repair_tags", [])),
            "raw_stop_loss_pct": float(repair_meta.get("raw_stop_loss_pct", trade_intent.stop_loss_pct)),
            "raw_take_profit_pct": float(repair_meta.get("raw_take_profit_pct", trade_intent.take_profit_pct)),
            "repaired_stop_loss_pct": float(repair_meta.get("repaired_stop_loss_pct", trade_intent.stop_loss_pct)),
            "repaired_take_profit_pct": float(repair_meta.get("repaired_take_profit_pct", trade_intent.take_profit_pct)),
            "vetoed": trade_intent.vetoed,
            "veto_reason": trade_intent.veto_reason,
            "raw_vetoed": bool(veto_meta.get("raw_vetoed", False)),
            "raw_veto_reason": veto_meta.get("raw_veto_reason"),
            "veto_overridden": bool(veto_meta.get("veto_overridden", False)),
            "veto_override_trigger": veto_meta.get("veto_override_trigger"),
            "veto_override_confidence_threshold": float(veto_meta.get("veto_override_confidence_threshold", -1.0)),
            "veto_override_size_scale": float(veto_meta.get("veto_override_size_scale", 0.0)),
            "veto_override_reason_allowed": bool(veto_meta.get("veto_override_reason_allowed", False)),
            "veto_override_reason_filter": list(veto_meta.get("veto_override_reason_filter", [])),
            "score": score,
            "legacy_score": float(legacy_raw_score),
            "score_mode": "net_effective_predicted_edge_bps",
            "rr": rr,
            "predicted_move_bps": predicted_move_bps,
            "predicted_edge_bps": predicted_edge_bps,
            "move_calibration_scale": float(move_calibration_scale),
            "calibrated_predicted_move_bps": float(calibrated_predicted_move_bps),
            "calibrated_predicted_edge_bps": float(calibrated_predicted_edge_bps),
            "effective_predicted_move_bps": float(effective_predicted_move_bps),
            "effective_predicted_edge_bps": float(effective_predicted_edge_bps),
            "net_effective_predicted_edge_bps": float(net_effective_predicted_edge_bps),
            "trade_head_calibration_loaded": bool(self.trade_head_calibrator is not None),
            "roundtrip_cost_bps": roundtrip_cost_bps,
            "passes_edge_gate": passes_edge_gate,
            "price": price,
            "price_timestamp": self.latest_price_timestamps.get(data_symbol),
            "intent": trade_intent,
        }

    def execute_trade(self, signal: Dict):
        symbol = signal["symbol"]
        direction = float(signal["signal"])
        confidence = float(signal["confidence"])

        if direction == 0 or confidence < self.config.signal_threshold:
            return None
        if bool(signal.get("vetoed", False)):
            return None

        if symbol in self.positions:
            pos = self.positions[symbol]
            if np.sign(pos["direction"]) != np.sign(direction):
                self._close_position(symbol)
            else:
                return None

        if len(self.positions) >= self.config.max_positions:
            return None

        stop_loss = float(signal.get("stop_loss_pct", self.config.stop_loss))
        take_profit = float(signal.get("take_profit_pct", self.config.take_profit))
        rr = take_profit / max(stop_loss, 1e-6)
        position_fraction = float(signal.get("position_fraction", 1.0))
        position_fraction = min(1.0, max(0.0, position_fraction))

        base_position_size = self.config.capital * self.config.risk_per_trade / max(stop_loss, 1e-6)
        position_size = base_position_size * position_fraction
        entry_price = self._get_current_price(symbol)
        if entry_price is None:
            return None

        position = {
            "symbol": symbol,
            "direction": direction,
            "size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "risk_tier": signal.get("risk_tier", RiskTier.NORMAL.value),
            "veto_overridden": bool(signal.get("veto_overridden", False)),
            "raw_vetoed": bool(signal.get("raw_vetoed", signal.get("vetoed", False))),
            "raw_veto_reason": signal.get("raw_veto_reason", signal.get("veto_reason")),
            "entry_iteration": int(self.current_iteration),
            "timestamp": datetime.now().isoformat(),
        }

        self.positions[symbol] = position
        cooldown_iters = (
            int(self.config.entry_cooldown_iterations)
            if int(self.config.entry_cooldown_iterations) >= 0
            else int(self.config.max_hold_iterations)
        )
        if cooldown_iters > 0:
            self.symbol_cooldown_until_iteration[symbol] = int(self.current_iteration + cooldown_iters)

        mode_label = "PAPER" if self.config.mode == "paper" else "LIVE-PREVIEW"
        print(
            f"📝 {mode_label}: {direction:+.0f} {symbol} @ {entry_price:.4f} "
            f"(size ${position_size:.2f}, conf {confidence:.2f}, rr {rr:.2f}"
            f"{', veto-override' if bool(signal.get('veto_overridden', False)) else ''})"
        )
        return position

    def _signal_to_intent(self, signal: Dict) -> Optional[NormalizedTradeIntent]:
        if "error" in signal:
            return None
        if isinstance(signal.get("intent"), NormalizedTradeIntent):
            return signal["intent"]

        direction = float(signal.get("signal", 0.0))
        confidence = float(signal.get("confidence", 0.0))
        pred = float(signal.get("prediction", 0.0))

        return NormalizedTradeIntent(
            symbol=signal["symbol"],
            direction=direction,
            pred_fwd_return=pred,
            confidence=confidence,
            position_fraction=min(1.0, max(0.0, float(signal.get("position_fraction", confidence)))),
            stop_loss_pct=-abs(float(signal.get("stop_loss_pct", self.config.stop_loss))),
            take_profit_pct=abs(float(signal.get("take_profit_pct", self.config.take_profit))),
            vetoed=bool(signal.get("vetoed", False)),
            veto_reason=signal.get("veto_reason"),
            risk_tier=RiskTier(str(signal.get("risk_tier", RiskTier.NORMAL.value))),
        )

    def execute_trade_intent(self, intent: NormalizedTradeIntent, signal_row: Optional[Dict] = None):
        if intent.vetoed:
            return None
        if bool(self.config.offload_execution_to_freqtrade):
            return self._emit_freqtrade_handoff_intent(intent, signal_row=signal_row)

        order_preview = intent_to_coinbase_order_preview(intent)
        if signal_row is not None:
            model = order_preview.setdefault("model", {})
            model["veto_overridden"] = bool(signal_row.get("veto_overridden", False))
            model["raw_vetoed"] = bool(signal_row.get("raw_vetoed", signal_row.get("vetoed", False)))
            model["raw_veto_reason"] = signal_row.get("raw_veto_reason", signal_row.get("veto_reason"))
            model["veto_override_trigger"] = signal_row.get("veto_override_trigger")
            model["risk_heads_repaired"] = bool(signal_row.get("risk_heads_repaired", False))
            model["risk_head_repair_tags"] = list(signal_row.get("risk_head_repair_tags", []))
            model["raw_stop_loss_pct"] = signal_row.get("raw_stop_loss_pct")
            model["raw_take_profit_pct"] = signal_row.get("raw_take_profit_pct")
            model["move_calibration_scale"] = signal_row.get("move_calibration_scale")
            model["calibrated_predicted_move_bps"] = signal_row.get("calibrated_predicted_move_bps")
            model["calibrated_predicted_edge_bps"] = signal_row.get("calibrated_predicted_edge_bps")
        self.orders.append(order_preview)

        legacy_signal = intent_to_legacy_signal(intent)
        if signal_row is not None:
            for k in (
                "veto_overridden",
                "raw_vetoed",
                "raw_veto_reason",
                "veto_override_trigger",
                "veto_override_confidence_threshold",
                "veto_override_size_scale",
                "veto_override_reason_allowed",
                "veto_override_reason_filter",
                "risk_heads_repaired",
                "risk_head_repair_tags",
                "raw_stop_loss_pct",
                "raw_take_profit_pct",
                "repaired_stop_loss_pct",
                "repaired_take_profit_pct",
                "move_calibration_scale",
                "calibrated_predicted_move_bps",
                "calibrated_predicted_edge_bps",
                "effective_predicted_move_bps",
                "effective_predicted_edge_bps",
            ):
                if k in signal_row:
                    legacy_signal[k] = signal_row[k]
        return self.execute_trade(legacy_signal)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self._json_safe(row)) + "\n")

    def _new_hrm_signal_id(self, symbol: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        sym = str(symbol or "UNKNOWN").upper()
        return f"hrm-{ts}-{sym}-{uuid4().hex[:10]}"

    def _build_hrm_fidelity_dispatch_event(
        self,
        signal_id: str,
        intent: NormalizedTradeIntent,
        payload: Dict[str, Any],
        signal_row: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        signal_row = signal_row or {}
        model = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
        risk = payload.get("risk", {}) if isinstance(payload.get("risk"), dict) else {}
        return {
            "schema": "moneyfan.hrm.fidelity.dispatch.v1",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "signal_id": signal_id,
            "iteration": int(getattr(self, "current_iteration", 0) or 0),
            "execution_target": "freqtrade",
            "execution_handoff_path": str(self.config.freqtrade_handoff_path),
            "status": "dispatched_pending_external_fill",
            "instrument": {
                "symbol": str(intent.symbol),
                "pair": str(payload.get("pair", intent.symbol)),
                "side": str(payload.get("side", "long")),
                "price": signal_row.get("price"),
                "price_timestamp": signal_row.get("price_timestamp"),
            },
            "prediction": {
                "pred_fwd_return": model.get("pred_fwd_return", intent.pred_fwd_return),
                "confidence": model.get("confidence", intent.confidence),
                "score": model.get("score"),
                "score_mode": model.get("score_mode"),
                "passes_edge_gate": model.get("passes_edge_gate"),
                "predicted_move_bps": model.get("predicted_move_bps"),
                "predicted_edge_bps": model.get("predicted_edge_bps"),
                "calibrated_predicted_move_bps": model.get("calibrated_predicted_move_bps"),
                "calibrated_predicted_edge_bps": model.get("calibrated_predicted_edge_bps"),
                "effective_predicted_move_bps": model.get("effective_predicted_move_bps"),
                "effective_predicted_edge_bps": model.get("effective_predicted_edge_bps"),
                "net_effective_predicted_edge_bps": model.get("net_effective_predicted_edge_bps"),
                "move_calibration_scale": model.get("move_calibration_scale"),
                "trade_head_calibration_loaded": bool(model.get("trade_head_calibration_loaded", False)),
            },
            "risk": {
                "risk_tier": risk.get("risk_tier", intent.risk_tier.value),
                "stop_loss_pct": risk.get("stop_loss_pct", abs(intent.stop_loss_pct)),
                "take_profit_pct": risk.get("take_profit_pct", max(intent.take_profit_pct, 0.0)),
                "position_fraction": risk.get("position_fraction", intent.position_fraction),
                "vetoed": bool(model.get("vetoed", intent.vetoed)),
                "veto_reason": model.get("veto_reason", intent.veto_reason),
                "raw_vetoed": bool(model.get("raw_vetoed", False)),
                "raw_veto_reason": model.get("raw_veto_reason"),
                "veto_overridden": bool(model.get("veto_overridden", False)),
                "veto_override_trigger": model.get("veto_override_trigger"),
                "risk_heads_repaired": bool(model.get("risk_heads_repaired", False)),
                "risk_head_repair_tags": list(model.get("risk_head_repair_tags", [])),
            },
        }

    def _emit_freqtrade_handoff_intent(
        self,
        intent: NormalizedTradeIntent,
        signal_row: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        signal_id = self._new_hrm_signal_id(intent.symbol)
        payload = intent_to_freqtrade_handoff(intent, signal_row=signal_row)
        payload["signal_id"] = signal_id
        payload["dispatch"] = {
            "target": "freqtrade",
            "mode": "handoff_file",
            "handoff_path": str(self.config.freqtrade_handoff_path),
            "source_mode": str(self.config.mode),
            "source_broker_label": str(self.config.broker),
            "iteration": int(getattr(self, "current_iteration", 0) or 0),
            "signal_id": signal_id,
        }
        self.orders.append(payload)
        self._append_jsonl(Path(self.config.freqtrade_handoff_path), payload)
        if bool(self.config.emit_hrm_fidelity_dispatch_log):
            fidelity_event = self._build_hrm_fidelity_dispatch_event(
                signal_id=signal_id,
                intent=intent,
                payload=payload,
                signal_row=signal_row,
            )
            self._append_jsonl(Path(self.config.hrm_fidelity_dispatch_log_path), fidelity_event)
        print(
            "📤 FREQTRADE-HANDOFF: "
            f"{payload.get('side', '?')} {payload.get('pair', intent.symbol)} "
            f"stake_frac={float(payload.get('stake_fraction', 0.0)):.2f} "
            f"conf={float(payload.get('model', {}).get('confidence', 0.0)):.2f} "
            f"signal_id={signal_id[-10:]}"
        )
        return payload

    def _get_current_price(self, symbol: str) -> Optional[float]:
        price = self.latest_prices.get(symbol)
        if price is not None:
            return float(price)
        return self._refresh_price_from_candles(symbol)

    def _close_position(self, symbol: str):
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        exit_price = self._get_current_price(symbol)
        if exit_price is None:
            return

        pnl = (exit_price - pos["entry_price"]) / max(pos["entry_price"], 1e-9) * pos["size"]
        if pos["direction"] < 0:
            pnl = -pnl

        trade = {
            "symbol": symbol,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": float(pnl),
            "timestamp": datetime.now().isoformat(),
        }

        self.trades.append(trade)
        self.pnl += float(pnl)
        self._record_realized_pnl(float(pnl))
        del self.positions[symbol]

        print(f"💰 Closed {symbol}: PnL ${pnl:.2f} (Total: ${self.pnl:.2f})")

    def update_positions(self):
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            pnl_pct = (current_price - pos["entry_price"]) / max(pos["entry_price"], 1e-9)
            if pos["direction"] < 0:
                pnl_pct = -pnl_pct

            if pnl_pct <= -float(pos["stop_loss"]):
                print(f"🛑 Stop loss hit: {symbol}")
                self._close_position(symbol)
            elif pnl_pct >= float(pos["take_profit"]):
                print(f"🎯 Take profit hit: {symbol}")
                self._close_position(symbol)
            else:
                held_iters = int(self.current_iteration) - int(pos.get("entry_iteration", self.current_iteration))
                if held_iters >= int(self.config.max_hold_iterations):
                    print(f"⏱️  Max hold reached: {symbol} ({held_iters} iterations)")
                    self._close_position(symbol)

    def _rank_trade_candidates(self, signal_rows: List[Dict]) -> List[Dict]:
        candidates = []
        # Use effective signal threshold from guardrail actions
        effective_threshold = self._get_effective_signal_threshold()
        
        for row in signal_rows:
            if "error" in row:
                continue
            if row.get("vetoed"):
                continue
            if not bool(row.get("passes_edge_gate", True)):
                continue
            cooldown_until = int(self.symbol_cooldown_until_iteration.get(str(row.get("symbol")), 0))
            if int(self.current_iteration) < cooldown_until:
                continue
            # Use effective threshold (may be raised by guardrail derisk action)
            if float(row.get("confidence", 0.0)) < effective_threshold:
                continue
            if float(row.get("signal", 0.0)) == 0.0:
                continue
            candidates.append(row)
        candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return candidates

    def run(self):
        if not self.model:
            print("❌ Cannot start: HRM model runtime not available")
            return

        print(f"\n🚀 Starting {self.config.mode.upper()} trading")
        print(f"💰 Capital: ${self.config.capital:.2f}")
        print(f"📊 Symbols: {', '.join(self.config.symbols[:8])}{'...' if len(self.config.symbols) > 8 else ''}")
        print(f"⚡ Max positions: {self.config.max_positions} | Top-k per loop: {self.config.top_k}")
        print(
            f"🧠 Seq len: {self.config.seq_len} | Pretrain steps: {self.config.online_pretrain_steps} | "
            f"Max hold iters: {self.config.max_hold_iterations}"
        )
        print(
            "🛡️  Kill-switches: "
            f"max_dd={self.config.max_drawdown_kill_pct:.1%} "
            f"max_daily_loss_pct={self.config.max_daily_loss_pct:.1%} "
            f"max_daily_loss_abs=${self.config.max_daily_loss_abs:.2f}"
        )
        if self.config.guardrail_enabled:
            print(
                "🛡️  Drawdown Guardrails: "
                f"warn={self.config.guardrail_warn_drawdown_pct:.1%} "
                f"derisk={self.config.guardrail_derisk_drawdown_pct:.1%} "
                f"halt={self.config.guardrail_halt_drawdown_pct:.1%} "
                f"window={self.config.guardrail_confirmation_window} "
                f"path={self.config.guardrail_events_log_path}"
            )
        if bool(self.config.offload_execution_to_freqtrade):
            print(
                "🔌 Execution offload: Freqtrade handoff enabled | "
                f"path={self.config.freqtrade_handoff_path} | internal positions disabled"
            )
            if bool(self.config.emit_hrm_fidelity_dispatch_log):
                print(
                    "🧪 HRM fidelity dispatch log: enabled | "
                    f"path={self.config.hrm_fidelity_dispatch_log_path}"
                )
        if self.halt_reason and bool(self.config.respect_saved_halt_state):
            print(f"🛑 Refusing to start due to saved halt state: {self.halt_reason}")
            return
        if self.trade_head_calibrator is not None:
            desc = self.trade_head_calibrator.describe()
            print(f"📐 Trade-head calibration: loaded (global_scale={desc.get('global_scale', 1.0):.3f})")
        elif self._trade_head_calibration_load_error:
            print(f"⚠️  Trade-head calibration load error: {self._trade_head_calibration_load_error}")
        print("\nPress Ctrl+C to stop\n")

        self.running = True
        iteration = int(self.current_iteration or 0)

        while self.running:
            iteration += 1
            self.current_iteration = int(iteration)
            print(f"\n--- Iteration {iteration} ({datetime.now().strftime('%H:%M:%S')}) ---")

            # Realize exits and enforce kill-switches before taking new risk.
            self.update_positions()
            if not self._check_kill_switches():
                break

            # Update guardrail action based on current guardrail state
            self._update_guardrail_action()
            
            # Check if new entries are allowed based on guardrail state
            entries_allowed, entry_reason = self._should_allow_new_entries()
            if not entries_allowed:
                print(f"🛡️  GUARDRAIL: New entries blocked - {entry_reason}")

            signal_rows: List[Dict] = []
            for symbol in self.config.symbols:
                if not self.running:
                    break
                signal_row = self.generate_signals(symbol)
                signal_rows.append(signal_row)
                if "error" in signal_row:
                    print(f"⚠️  {symbol}: {signal_row['error']}")

            ranked = self._rank_trade_candidates(signal_rows)
            if ranked:
                preview = ", ".join(
                    f"{r['symbol']} {r['signal']:+.0f} conf={r['confidence']:.2f} score={r['score']:.4f}"
                    f"{' veto-override' if bool(r.get('veto_overridden', False)) else ''}"
                    for r in ranked[: min(5, len(ranked))]
                )
                print(f"📈 Ranked signals: {preview}")
            else:
                print("📉 No executable signals this iteration")

            # Apply guardrail action modifications
            effective_top_k = self._get_effective_top_k()
            effective_threshold = self._get_effective_signal_threshold()
            position_size_scale = self._get_effective_position_size_scale()
            
            # Log guardrail-modified parameters if different from config
            if effective_top_k != self.config.top_k or effective_threshold != self.config.signal_threshold:
                print(
                    f"🛡️  GUARDRAIL APPLIED: top_k={effective_top_k} (base={self.config.top_k}) "
                    f"threshold={effective_threshold:.2f} (base={self.config.signal_threshold:.2f}) "
                    f"size_scale={position_size_scale:.2f}"
                )

            opened_this_loop = 0
            for row in ranked:
                if opened_this_loop >= effective_top_k:
                    break
                intent = self._signal_to_intent(row)
                if intent is None:
                    continue
                
                # Apply position size scaling based on guardrail state
                if position_size_scale < 1.0 and intent.position_fraction > 0:
                    intent.position_fraction = max(0.01, intent.position_fraction * position_size_scale)
                
                before_positions = len(self.positions)
                result = self.execute_trade_intent(intent, signal_row=row)
                after_positions = len(self.positions)
                if result is not None or after_positions != before_positions:
                    opened_this_loop += 1

            self.update_positions()
            if not self._check_kill_switches():
                break
            self._save_state()

            print(
                f"📊 Status: positions={len(self.positions)} pnl=${self.pnl:.2f} "
                f"orders={len(self.orders)} trades={len(self.trades)} "
                f"guardrail={self.guardrail_state}"
            )

            if self.config.max_iterations is not None and iteration >= self.config.max_iterations:
                print("🛑 Reached configured iteration limit")
                self.running = False
                break

            sleep_s = max(0.0, float(self.config.sleep_seconds))
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _save_state(self):
        snapshot = self._risk_snapshot()
        state = {
            "mode": self.config.mode,
            "capital": self.config.capital,
            "pnl": self.pnl,
            "positions": self.positions,
            "trades": self.trades[-100:],
            "orders": self.orders[-100:],
            "latest_prices": self.latest_prices,
            "latest_price_timestamps": self.latest_price_timestamps,
            "current_iteration": int(self.current_iteration),
            "symbol_cooldown_until_iteration": self.symbol_cooldown_until_iteration,
            "halt_reason": self.halt_reason,
            "guardrail_state": self.guardrail_state,
            "guardrail_candidate_state": self.guardrail_candidate_state,
            "guardrail_candidate_iterations": int(self.guardrail_candidate_iterations),
            "risk_state": {
                "peak_equity": float(self.peak_equity),
                "risk_day_utc": self.risk_day_utc,
                "risk_day_start_equity": float(self.risk_day_start_equity),
                "risk_day_realized_pnl": float(self.risk_day_realized_pnl),
                "equity": float(snapshot.get("equity", self._equity())),
                "drawdown_pct": float(snapshot.get("drawdown_pct", 0.0)),
                "daily_loss_abs": float(snapshot.get("daily_loss_abs", 0.0)),
                "daily_loss_pct": float(snapshot.get("daily_loss_pct", 0.0)),
                "halt_reason": self.halt_reason,
            },
            "timestamp": datetime.now().isoformat(),
        }

        path = Path(getattr(self, "state_path", self.config.state_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Run HRM Trading")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (paper or live-preview)")
    parser.add_argument("--capital", type=float, default=100.0,
                        help="Starting capital")
    parser.add_argument("--broker", default="coinbase",
                        help="Broker/exchange label (execution is preview-only)")
    parser.add_argument("--offload-execution-to-freqtrade", action="store_true",
                        help="Export HRM intents to a Freqtrade handoff JSONL file instead of internal preview execution")
    parser.add_argument("--freqtrade-handoff-path", type=str, default="runtime/freqtrade_handoff.jsonl",
                        help="JSONL file path used when --offload-execution-to-freqtrade is enabled")
    parser.add_argument("--hrm-fidelity-dispatch-log-path", type=str, default="runtime/hrm_fidelity_dispatch.jsonl",
                        help="JSONL file for HRM prediction/calibration snapshots at dispatch time")
    parser.add_argument("--no-hrm-fidelity-dispatch-log", action="store_true",
                        help="Disable extra HRM fidelity dispatch logging when offloading execution")
    parser.add_argument("--risk", type=float, default=0.01,
                        help="Risk per trade (fraction of capital)")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Maximum concurrent positions")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (auto-normalized to local data format, e.g. BTC-USD -> BTCUSDT)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="HRM sequence length for inference")
    parser.add_argument("--feature-bars", type=int, default=768,
                        help="Recent bars used to compute indicators/features per symbol")
    parser.add_argument("--signal-threshold", type=float, default=0.65,
                        help="Minimum conviction to execute")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Max new executions per iteration (ranked by expected edge)")
    parser.add_argument("--online-pretrain-steps", type=int, default=0,
                        help="Optional online HRM world-model updates before inference")
    parser.add_argument("--sleep-seconds", type=float, default=60.0,
                        help="Loop sleep between iterations")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Stop after N iterations (0 = run forever)")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to MLX HRM .npz weights checkpoint")
    parser.add_argument("--no-mechanical-veto", action="store_true",
                        help="Disable rule-based execution veto filter")
    parser.add_argument("--commission-bps", type=float, default=5.0,
                        help="Per-side commission estimate for edge gating (bps)")
    parser.add_argument("--slippage-bps", type=float, default=3.0,
                        help="Per-side slippage estimate for edge gating (bps)")
    parser.add_argument("--min-pred-move-bps", type=float, default=0.0,
                        help="Minimum model-implied move size to consider (bps)")
    parser.add_argument("--edge-cost-multiplier", type=float, default=1.0,
                        help="Require predicted move >= multiplier * roundtrip estimated costs")
    parser.add_argument("--max-hold-iterations", type=int, default=12,
                        help="Force-close positions after this many loop iterations")
    parser.add_argument("--entry-cooldown-iterations", type=int, default=-1,
                        help="Per-symbol entry cooldown in loop iterations (-1 uses max-hold-iterations)")
    parser.add_argument("--veto-confidence-override-threshold", type=float, default=-1.0,
                        help="If >0, allow high-confidence signals to override mechanical veto")
    parser.add_argument("--veto-confidence-override-size-scale", type=float, default=0.5,
                        help="Position size scale applied when veto is overridden by confidence")
    parser.add_argument("--veto-confidence-override-reasons", type=str,
                        default="low_confidence,poor_risk_reward",
                        help="Comma-separated veto reasons eligible for confidence override ('any' to allow all)")
    parser.add_argument("--no-risk-head-repair", action="store_true",
                        help="Disable conservative repair of malformed stop/target heads before veto")
    parser.add_argument("--repair-min-stop-loss-pct", type=float, default=0.002,
                        help="Floor for |stop_loss_pct| before veto")
    parser.add_argument("--repair-max-stop-loss-pct", type=float, default=0.15,
                        help="Cap for |stop_loss_pct| before veto")
    parser.add_argument("--repair-min-take-profit-pct", type=float, default=0.003,
                        help="Floor for take_profit_pct before veto")
    parser.add_argument("--trade-head-calibration", type=str, default="",
                        help="Path to trade-head calibration JSON (defaults to models/trained/hrm_trade_head_calibration.json if present)")
    parser.add_argument("--no-trade-head-calibration", action="store_true",
                        help="Disable trade-head calibration even if an artifact exists")
    parser.add_argument("--state-path", type=str, default="trading_state.json",
                        help="Path to persisted trading state (for crash-safe resume)")
    parser.add_argument("--no-resume-state", action="store_true",
                        help="Start fresh and ignore saved state")
    parser.add_argument("--max-drawdown-kill-pct", type=float, default=0.12,
                        help="Hard-halt if realized portfolio drawdown reaches this fraction (0 disables)")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03,
                        help="Hard-halt if UTC-day realized loss reaches this fraction of day-start equity (0 disables)")
    parser.add_argument("--max-daily-loss-abs", type=float, default=0.0,
                        help="Hard-halt if UTC-day realized loss reaches this absolute dollar value (0 disables)")
    parser.add_argument("--ignore-saved-halt-state", action="store_true",
                        help="Resume even if saved state contains a previous hard-halt reason")
    parser.add_argument("--guardrail-enabled", action="store_true",
                        help="Enable drawdown guardrail state machine (disabled by default)")
    parser.add_argument("--guardrail-warn-drawdown-pct", type=float, default=0.05,
                        help="Drawdown fraction that triggers warn state (default 5%%)")
    parser.add_argument("--guardrail-derisk-drawdown-pct", type=float, default=0.08,
                        help="Drawdown fraction that triggers de-risk state (default 8%%)")
    parser.add_argument("--guardrail-halt-drawdown-pct", type=float, default=0.12,
                        help="Drawdown fraction that triggers halt state (default 12%%)")
    parser.add_argument("--guardrail-confirmation-window", type=int, default=1,
                        help="Iterations of sustained violation required before state transition")
    parser.add_argument("--guardrail-events-log-path", type=str,
                        default="runtime/guardrail_events.jsonl",
                        help="JSONL file for guardrail transition event artifact emission")

    args = parser.parse_args()

    if args.mode == "live":
        print("⚠️  WARNING: Live mode here is preview-only (no broker API execution is wired).")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

    parsed_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    max_iterations = None if int(args.iterations or 0) <= 0 else int(args.iterations)

    config = TradingConfig(
        mode=args.mode,
        capital=args.capital,
        broker=args.broker,
        offload_execution_to_freqtrade=args.offload_execution_to_freqtrade,
        freqtrade_handoff_path=args.freqtrade_handoff_path,
        emit_hrm_fidelity_dispatch_log=not args.no_hrm_fidelity_dispatch_log,
        hrm_fidelity_dispatch_log_path=args.hrm_fidelity_dispatch_log_path,
        symbols=parsed_symbols,
        risk_per_trade=args.risk,
        max_positions=args.max_positions,
        seq_len=args.seq_len,
        feature_lookback_bars=args.feature_bars,
        signal_threshold=args.signal_threshold,
        top_k=args.top_k,
        online_pretrain_steps=args.online_pretrain_steps,
        sleep_seconds=args.sleep_seconds,
        max_iterations=max_iterations,
        use_mechanical_veto=not args.no_mechanical_veto,
        weights_path=(args.weights or None),
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        min_pred_move_bps=args.min_pred_move_bps,
        edge_cost_multiplier=args.edge_cost_multiplier,
        max_hold_iterations=args.max_hold_iterations,
        entry_cooldown_iterations=args.entry_cooldown_iterations,
        veto_confidence_override_threshold=args.veto_confidence_override_threshold,
        veto_confidence_override_size_scale=args.veto_confidence_override_size_scale,
        veto_confidence_override_reasons=args.veto_confidence_override_reasons,
        repair_risk_heads=not args.no_risk_head_repair,
        repair_min_stop_loss_pct=args.repair_min_stop_loss_pct,
        repair_max_stop_loss_pct=args.repair_max_stop_loss_pct,
        repair_min_take_profit_pct=args.repair_min_take_profit_pct,
        trade_head_calibration_path=(args.trade_head_calibration or None),
        use_trade_head_calibration=not args.no_trade_head_calibration,
        state_path=args.state_path,
        resume_state=not args.no_resume_state,
        max_drawdown_kill_pct=args.max_drawdown_kill_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_daily_loss_abs=args.max_daily_loss_abs,
        respect_saved_halt_state=not args.ignore_saved_halt_state,
        guardrail_enabled=args.guardrail_enabled,
        guardrail_warn_drawdown_pct=args.guardrail_warn_drawdown_pct,
        guardrail_derisk_drawdown_pct=args.guardrail_derisk_drawdown_pct,
        guardrail_halt_drawdown_pct=args.guardrail_halt_drawdown_pct,
        guardrail_confirmation_window=args.guardrail_confirmation_window,
        guardrail_events_log_path=args.guardrail_events_log_path,
    )

    engine = TradingEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
