from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from hrm.order_intent import NormalizedTradeIntent


def intent_to_legacy_signal(intent: NormalizedTradeIntent) -> Dict:
    """
    Backward-compatible adapter for the existing TradingEngine.execute_trade(signal).

    This preserves the current execution path while allowing upstream code to move
    to a normalized intent representation.
    """
    return {
        "symbol": intent.symbol,
        "signal": float(intent.direction),
        "confidence": float(intent.confidence),
        "prediction": float(intent.pred_fwd_return),
        "stop_loss_pct": float(abs(intent.stop_loss_pct)),
        "take_profit_pct": float(max(intent.take_profit_pct, 0.0)),
        "position_fraction": float(intent.position_fraction),
        "risk_tier": intent.risk_tier.value,
        "vetoed": bool(intent.vetoed),
        "veto_reason": intent.veto_reason,
    }


def intent_to_coinbase_order_preview(intent: NormalizedTradeIntent) -> Dict:
    """
    Broker adapter seam for Coinbase Advanced Trade style order payloads.

    This is a normalized preview envelope, not a live API request.
    Lane D intentionally avoids coupling TradingEngine to endpoint-specific fields.
    """
    side = "BUY" if intent.direction > 0 else "SELL"
    return {
        "product_id": intent.symbol,
        "side": side,
        "order_type": "BRACKET_MARKET_PREVIEW",
        "risk": {
            "stop_loss_pct": float(abs(intent.stop_loss_pct)),
            "take_profit_pct": float(max(intent.take_profit_pct, 0.0)),
            "position_fraction": float(intent.position_fraction),
            "risk_tier": intent.risk_tier.value,
        },
        "model": {
            "pred_fwd_return": float(intent.pred_fwd_return),
            "confidence": float(intent.confidence),
            "vetoed": bool(intent.vetoed),
            "veto_reason": intent.veto_reason,
        },
        "normalized_intent": {
            **asdict(intent),
            "risk_tier": intent.risk_tier.value,
        },
    }


_FREQTRADE_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "USD",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
    "GBP",
)


def _symbol_to_freqtrade_pair(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not raw:
        return raw
    for quote in _FREQTRADE_QUOTE_SUFFIXES:
        if raw.endswith(quote) and len(raw) > len(quote):
            base = raw[: -len(quote)]
            return f"{base}/{quote}"
    return raw


def intent_to_freqtrade_handoff(
    intent: NormalizedTradeIntent,
    signal_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a normalized intent into a Freqtrade-oriented handoff envelope.

    This is a repo-local transport payload for file/webhook handoff, not a direct
    Freqtrade API contract. It keeps HRM fidelity metadata attached so execution
    can be delegated without losing model diagnostics.
    """
    side = "long" if float(intent.direction) > 0 else "short"
    pair = _symbol_to_freqtrade_pair(intent.symbol)
    payload: Dict[str, Any] = {
        "schema": "moneyfan.freqtrade.handoff.v1",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "symbol": intent.symbol,
        "side": side,
        "enter_long": 1 if side == "long" else 0,
        "enter_short": 1 if side == "short" else 0,
        "stake_fraction": float(max(0.0, min(1.0, intent.position_fraction))),
        "stoploss": -abs(float(intent.stop_loss_pct)),
        "take_profit_pct": float(max(intent.take_profit_pct, 0.0)),
        "risk": {
            "risk_tier": intent.risk_tier.value,
            "stop_loss_pct": float(abs(intent.stop_loss_pct)),
            "take_profit_pct": float(max(intent.take_profit_pct, 0.0)),
            "position_fraction": float(max(0.0, min(1.0, intent.position_fraction))),
        },
        "model": {
            "pred_fwd_return": float(intent.pred_fwd_return),
            "confidence": float(intent.confidence),
            "vetoed": bool(intent.vetoed),
            "veto_reason": intent.veto_reason,
        },
        "normalized_intent": {
            **asdict(intent),
            "risk_tier": intent.risk_tier.value,
        },
    }
    if signal_row:
        payload["model"].update(
            {
                "score": signal_row.get("score"),
                "score_mode": signal_row.get("score_mode"),
                "passes_edge_gate": bool(signal_row.get("passes_edge_gate", True)),
                "predicted_move_bps": signal_row.get("predicted_move_bps"),
                "predicted_edge_bps": signal_row.get("predicted_edge_bps"),
                "move_calibration_scale": signal_row.get("move_calibration_scale"),
                "calibrated_predicted_move_bps": signal_row.get("calibrated_predicted_move_bps"),
                "calibrated_predicted_edge_bps": signal_row.get("calibrated_predicted_edge_bps"),
                "effective_predicted_move_bps": signal_row.get("effective_predicted_move_bps"),
                "effective_predicted_edge_bps": signal_row.get("effective_predicted_edge_bps"),
                "net_effective_predicted_edge_bps": signal_row.get("net_effective_predicted_edge_bps"),
                "trade_head_calibration_loaded": bool(signal_row.get("trade_head_calibration_loaded", False)),
                "risk_heads_repaired": bool(signal_row.get("risk_heads_repaired", False)),
                "risk_head_repair_tags": list(signal_row.get("risk_head_repair_tags", [])),
                "raw_vetoed": bool(signal_row.get("raw_vetoed", signal_row.get("vetoed", False))),
                "raw_veto_reason": signal_row.get("raw_veto_reason", signal_row.get("veto_reason")),
                "veto_overridden": bool(signal_row.get("veto_overridden", False)),
                "veto_override_trigger": signal_row.get("veto_override_trigger"),
            }
        )
        payload["signal_row"] = {
            "symbol": signal_row.get("symbol", intent.symbol),
            "signal": signal_row.get("signal", intent.direction),
            "confidence": signal_row.get("confidence", intent.confidence),
            "score": signal_row.get("score"),
            "price": signal_row.get("price"),
            "price_timestamp": signal_row.get("price_timestamp"),
            "passes_edge_gate": bool(signal_row.get("passes_edge_gate", True)),
        }
    return payload
