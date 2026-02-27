from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrm.order_intent import NormalizedTradeIntent, RiskTier
from run import TradingConfig, TradingEngine


def _engine_stub_for_offload(tmp_path: Path, **cfg_overrides):
    cfg = TradingConfig(
        mode="paper",
        capital=100.0,
        broker="freqtrade",
        offload_execution_to_freqtrade=True,
        freqtrade_handoff_path=str(tmp_path / "freqtrade_handoff.jsonl"),
        hrm_fidelity_dispatch_log_path=str(tmp_path / "hrm_fidelity_dispatch.jsonl"),
        symbols=["BTCUSDT"],
        state_path=str(tmp_path / "state.json"),
        **cfg_overrides,
    )
    engine = TradingEngine.__new__(TradingEngine)
    engine.config = cfg
    engine.orders = []
    engine.positions = {}
    engine.current_iteration = 3
    return engine


def test_execute_trade_intent_offloads_to_freqtrade_jsonl_without_internal_position(tmp_path):
    engine = _engine_stub_for_offload(tmp_path)
    engine.execute_trade = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("internal execute_trade should not run"))

    intent = NormalizedTradeIntent(
        symbol="BTCUSDT",
        direction=1.0,
        pred_fwd_return=0.008,
        confidence=0.76,
        position_fraction=0.35,
        stop_loss_pct=-0.015,
        take_profit_pct=0.03,
        vetoed=False,
        risk_tier=RiskTier.NORMAL,
    )

    result = engine.execute_trade_intent(
        intent,
        signal_row={
            "symbol": "BTCUSDT",
            "signal": 1.0,
            "confidence": 0.76,
            "score": 12.0,
            "passes_edge_gate": True,
            "trade_head_calibration_loaded": True,
        },
    )

    assert isinstance(result, dict)
    assert result["schema"] == "moneyfan.freqtrade.handoff.v1"
    assert result["dispatch"]["target"] == "freqtrade"
    assert result["dispatch"]["iteration"] == 3
    assert result["dispatch"]["signal_id"] == result["signal_id"]
    assert result["pair"] == "BTC/USDT"
    assert len(engine.orders) == 1
    assert engine.positions == {}

    handoff_path = Path(engine.config.freqtrade_handoff_path)
    assert handoff_path.exists()
    lines = handoff_path.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["schema"] == "moneyfan.freqtrade.handoff.v1"
    assert stored["dispatch"]["target"] == "freqtrade"
    assert stored["dispatch"]["signal_id"] == stored["signal_id"]
    assert stored["normalized_intent"]["symbol"] == "BTCUSDT"

    fidelity_log_path = Path(engine.config.hrm_fidelity_dispatch_log_path)
    assert fidelity_log_path.exists()
    fidelity_lines = fidelity_log_path.read_text().strip().splitlines()
    assert len(fidelity_lines) == 1
    fidelity_row = json.loads(fidelity_lines[0])
    assert fidelity_row["schema"] == "moneyfan.hrm.fidelity.dispatch.v1"
    assert fidelity_row["signal_id"] == stored["signal_id"]
    assert fidelity_row["execution_target"] == "freqtrade"
    assert fidelity_row["prediction"]["trade_head_calibration_loaded"] is True


def test_execute_trade_intent_offload_still_respects_veto(tmp_path):
    engine = _engine_stub_for_offload(tmp_path)
    intent = NormalizedTradeIntent(
        symbol="BTCUSDT",
        direction=1.0,
        pred_fwd_return=0.008,
        confidence=0.76,
        position_fraction=0.35,
        stop_loss_pct=-0.015,
        take_profit_pct=0.03,
        vetoed=True,
        veto_reason="low_confidence",
        risk_tier=RiskTier.NORMAL,
    )

    result = engine.execute_trade_intent(intent)

    assert result is None
    assert engine.orders == []
    assert not Path(engine.config.freqtrade_handoff_path).exists()
    assert not Path(engine.config.hrm_fidelity_dispatch_log_path).exists()
