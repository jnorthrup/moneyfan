from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.order_intent_adapter import intent_to_freqtrade_handoff
from hrm.order_intent import NormalizedTradeIntent, RiskTier


def test_intent_to_freqtrade_handoff_preserves_hrm_fidelity_metadata():
    intent = NormalizedTradeIntent(
        symbol="BTCUSDT",
        direction=1.0,
        pred_fwd_return=0.0125,
        confidence=0.81,
        position_fraction=0.42,
        stop_loss_pct=-0.018,
        take_profit_pct=0.031,
        vetoed=False,
        veto_reason=None,
        risk_tier=RiskTier.CAUTION,
    )
    signal_row = {
        "score": 14.25,
        "score_mode": "net_effective_predicted_edge_bps",
        "passes_edge_gate": True,
        "predicted_move_bps": 128.0,
        "predicted_edge_bps": 64.0,
        "move_calibration_scale": 0.9,
        "calibrated_predicted_move_bps": 115.2,
        "calibrated_predicted_edge_bps": 57.6,
        "effective_predicted_move_bps": 100.0,
        "effective_predicted_edge_bps": 49.0,
        "net_effective_predicted_edge_bps": 33.0,
        "trade_head_calibration_loaded": True,
        "risk_heads_repaired": True,
        "risk_head_repair_tags": ["clip_stop_loss"],
        "raw_vetoed": False,
        "raw_veto_reason": None,
        "veto_overridden": False,
        "price": 50250.0,
        "price_timestamp": "2026-02-25T12:34:56Z",
    }

    payload = intent_to_freqtrade_handoff(intent, signal_row=signal_row)

    assert payload["schema"] == "moneyfan.freqtrade.handoff.v1"
    assert payload["pair"] == "BTC/USDT"
    assert payload["side"] == "long"
    assert payload["enter_long"] == 1
    assert payload["enter_short"] == 0
    assert payload["stake_fraction"] == 0.42
    assert payload["stoploss"] == -0.018
    assert payload["take_profit_pct"] == 0.031
    assert payload["risk"]["risk_tier"] == "caution"
    assert payload["model"]["confidence"] == 0.81
    assert payload["model"]["trade_head_calibration_loaded"] is True
    assert payload["model"]["risk_heads_repaired"] is True
    assert payload["model"]["risk_head_repair_tags"] == ["clip_stop_loss"]
    assert payload["model"]["net_effective_predicted_edge_bps"] == 33.0
    assert payload["normalized_intent"]["risk_tier"] == "caution"
    assert payload["signal_row"]["price"] == 50250.0
    assert "ts_utc" in payload


def test_intent_to_freqtrade_handoff_formats_short_pair_and_stoploss():
    intent = NormalizedTradeIntent(
        symbol="ETHUSD",
        direction=-1.0,
        pred_fwd_return=-0.01,
        confidence=0.7,
        position_fraction=1.2,
        stop_loss_pct=-0.02,
        take_profit_pct=0.05,
        risk_tier=RiskTier.PROTECTIVE,
    )

    payload = intent_to_freqtrade_handoff(intent)

    assert payload["pair"] == "ETH/USD"
    assert payload["side"] == "short"
    assert payload["enter_long"] == 0
    assert payload["enter_short"] == 1
    assert payload["stake_fraction"] == 1.0
    assert payload["stoploss"] == -0.02
    assert payload["risk"]["risk_tier"] == "protective"
