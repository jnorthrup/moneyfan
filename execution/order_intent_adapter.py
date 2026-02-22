from __future__ import annotations

from dataclasses import asdict
from typing import Dict

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
