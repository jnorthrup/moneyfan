from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import TradingConfig, TradingEngine


def _engine_stub(tmp_path: Path, **cfg_overrides):
    cfg = TradingConfig(
        mode="paper",
        capital=100.0,
        symbols=["BTCUSDT"],
        state_path=str(tmp_path / "state.json"),
        **cfg_overrides,
    )
    engine = TradingEngine.__new__(TradingEngine)
    engine.config = cfg
    engine.state_path = Path(cfg.state_path)
    engine.positions = {}
    engine.orders = []
    engine.pnl = 0.0
    engine.trades = []
    engine.running = True
    engine.latest_prices = {}
    engine.latest_price_timestamps = {}
    engine.current_iteration = 0
    engine.symbol_cooldown_until_iteration = {}
    engine.trade_head_calibrator = None
    engine.peak_equity = float(cfg.capital)
    engine.risk_day_utc = TradingEngine._utc_day_key()
    engine.risk_day_start_equity = float(cfg.capital)
    engine.risk_day_realized_pnl = 0.0
    engine.halt_reason = None
    return engine


def test_state_roundtrip_restores_risk_and_resume_fields(tmp_path):
    engine = _engine_stub(tmp_path)
    engine.pnl = 12.5
    engine.positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "direction": 1.0,
            "size": 25.0,
            "entry_price": 50000.0,
            "stop_loss": 0.01,
            "take_profit": 0.02,
            "entry_iteration": 7,
        }
    }
    engine.trades = [{"symbol": "ETHUSDT", "pnl": 3.5, "timestamp": "2026-02-23T00:00:00"}]
    engine.orders = [{"id": "preview-1"}]
    engine.latest_prices = {"BTCUSDT": 51000.0}
    engine.latest_price_timestamps = {"BTCUSDT": "2026-02-23 12:00:00"}
    engine.current_iteration = 9
    engine.symbol_cooldown_until_iteration = {"BTCUSDT": 12}
    engine.peak_equity = 120.0
    engine.risk_day_start_equity = 110.0
    engine.risk_day_realized_pnl = -4.0
    engine.halt_reason = "test_halt"

    engine._save_state()

    restored = _engine_stub(tmp_path)
    restored._load_state()

    assert restored.pnl == 12.5
    assert "BTCUSDT" in restored.positions
    assert restored.orders == [{"id": "preview-1"}]
    assert restored.latest_prices["BTCUSDT"] == 51000.0
    assert restored.latest_price_timestamps["BTCUSDT"] == "2026-02-23 12:00:00"
    assert restored.current_iteration == 9
    assert restored.symbol_cooldown_until_iteration["BTCUSDT"] == 12
    assert restored.peak_equity == 120.0
    assert restored.risk_day_start_equity == 110.0
    assert restored.risk_day_realized_pnl == -4.0
    assert restored.halt_reason == "test_halt"


def test_kill_switch_halts_on_daily_loss_abs(tmp_path):
    engine = _engine_stub(
        tmp_path,
        max_drawdown_kill_pct=0.0,
        max_daily_loss_pct=0.0,
        max_daily_loss_abs=5.0,
    )
    engine.risk_day_realized_pnl = -6.25
    engine._save_state = lambda: None

    ok = engine._check_kill_switches()

    assert ok is False
    assert engine.running is False
    assert engine.halt_reason is not None
    assert "max_daily_loss_abs_exceeded" in engine.halt_reason


def test_kill_switch_halts_on_drawdown(tmp_path):
    engine = _engine_stub(
        tmp_path,
        max_drawdown_kill_pct=0.10,
        max_daily_loss_pct=0.0,
        max_daily_loss_abs=0.0,
    )
    engine.trades = [
        {"symbol": "BTCUSDT", "pnl": 10.0, "timestamp": "2026-02-23T00:00:00"},
        {"symbol": "BTCUSDT", "pnl": -20.0, "timestamp": "2026-02-23T00:01:00"},
    ]
    engine.pnl = -10.0
    engine._save_state = lambda: None

    ok = engine._check_kill_switches()

    assert ok is False
    assert engine.running is False
    assert engine.halt_reason is not None
    assert "max_drawdown_kill_pct_exceeded" in engine.halt_reason
