from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from run import TradingConfig, TradingEngine


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine_stub(tmp_path):
    """Return a factory that builds a minimal TradingEngine stub."""

    def _make(**cfg_overrides):
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
        engine.guardrail_state = "normal"
        engine.guardrail_candidate_state = "normal"
        engine.guardrail_candidate_iterations = 0
        return engine

    return _make


# ---------------------------------------------------------------------------
# Config-default tests
# ---------------------------------------------------------------------------

def test_guardrail_disabled_by_default():
    cfg = TradingConfig(symbols=["BTCUSDT"])
    assert cfg.guardrail_enabled is False


def test_guardrail_default_warn_threshold():
    cfg = TradingConfig(symbols=["BTCUSDT"])
    assert cfg.guardrail_warn_drawdown_pct == pytest.approx(0.05)


def test_guardrail_default_derisk_threshold():
    cfg = TradingConfig(symbols=["BTCUSDT"])
    assert cfg.guardrail_derisk_drawdown_pct == pytest.approx(0.08)


def test_guardrail_default_halt_threshold():
    cfg = TradingConfig(symbols=["BTCUSDT"])
    assert cfg.guardrail_halt_drawdown_pct == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Disabled-by-default no-op test
# ---------------------------------------------------------------------------

def test_check_guardrails_noop_when_disabled(engine_stub):
    engine = engine_stub()  # guardrail_enabled=False by default
    # Simulate a 15% drawdown
    engine.pnl = -15.0
    result = engine._check_drawdown_guardrails()
    assert result == "normal"
    assert engine.guardrail_state == "normal"


# ---------------------------------------------------------------------------
# Monotonic drawdown path
# ---------------------------------------------------------------------------

def test_guardrail_monotonic_drawdown_path(engine_stub):
    """Walk drawdown from 0 % -> 5 % -> 8 % -> 12 % and verify state ladder."""
    engine = engine_stub(guardrail_enabled=True)

    # At peak – no drawdown
    engine.pnl = 0.0
    assert engine._check_drawdown_guardrails() == "normal"
    assert engine.guardrail_state == "normal"

    # 5 % drawdown -> warn
    engine.pnl = -5.0
    assert engine._check_drawdown_guardrails() == "warn"
    assert engine.guardrail_state == "warn"

    # 8 % drawdown -> derisk
    engine.pnl = -8.0
    assert engine._check_drawdown_guardrails() == "derisk"
    assert engine.guardrail_state == "derisk"

    # 12 % drawdown -> halt
    engine.pnl = -12.0
    assert engine._check_drawdown_guardrails() == "halt"
    assert engine.guardrail_state == "halt"


# ---------------------------------------------------------------------------
# Confirmation window
# ---------------------------------------------------------------------------

def test_guardrail_confirmation_window(engine_stub):
    """Verify that a transition only happens after N iterations of violations."""
    engine = engine_stub(guardrail_enabled=True, guardrail_confirmation_window=3)

    # Initial state
    assert engine._check_drawdown_guardrails() == "normal"

    # Start 5% drawdown (violation)
    engine.pnl = -5.0

    # Iteration 1: remains normal, candidate is warn
    assert engine._check_drawdown_guardrails() == "normal"
    assert engine.guardrail_candidate_state == "warn"
    assert engine.guardrail_candidate_iterations == 1

    # Iteration 2: remains normal
    assert engine._check_drawdown_guardrails() == "normal"
    assert engine.guardrail_candidate_iterations == 2

    # Iteration 3: finally transitions to warn
    assert engine._check_drawdown_guardrails() == "warn"
    assert engine.guardrail_state == "warn"
    assert engine.guardrail_candidate_iterations == 3


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------

def test_guardrail_artifact_emission(engine_stub, tmp_path):
    """Verify that a transition emits a JSONL artifact with the expected schema."""
    events_path = tmp_path / "guardrail_events.jsonl"
    engine = engine_stub(
        guardrail_enabled=True,
        guardrail_events_log_path=str(events_path),
        guardrail_confirmation_window=1
    )
    # mock _append_jsonl if needed, but here we can just let it write to tmp_path
    # We need to make sure _append_jsonl works in the stub or use real engine methods
    
    # Real method uses Path(self.config.guardrail_events_log_path)
    # The stub doesn't have _append_jsonl and _emit_guardrail_event by default
    # since it's just a __new__'d object.
    
    # Let's attach the real methods to the stub for this test
    engine._append_jsonl = TradingEngine._append_jsonl.__get__(engine, TradingEngine)
    engine._emit_guardrail_event = TradingEngine._emit_guardrail_event.__get__(engine, TradingEngine)
    engine._json_safe = TradingEngine._json_safe.__get__(engine, TradingEngine)

    # Trigger a transition
    engine.pnl = -5.0
    engine._check_drawdown_guardrails()

    assert events_path.exists()
    import json
    with open(events_path, "r") as f:
        event = json.loads(f.readline())
    
    assert event["schema"] == "moneyfan.runtime.guardrail.event.v1"
    assert event["old_state"] == "normal"
    assert event["new_state"] == "warn"
    assert event["drawdown_pct"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Integration with Kill-Switches
# ---------------------------------------------------------------------------

def test_kill_switch_halts_on_guardrail_halt(engine_stub):
    """Verify that _check_kill_switches triggers a hard halt when guardrail is halt."""
    engine = engine_stub(guardrail_enabled=True, guardrail_halt_drawdown_pct=0.10)
    
    # Attach required methods for _trigger_halt and _save_state
    engine._risk_snapshot = TradingEngine._risk_snapshot.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._portfolio_drawdown_pct = TradingEngine._portfolio_drawdown_pct.__get__(engine, TradingEngine)
    engine._check_kill_switches = TradingEngine._check_kill_switches.__get__(engine, TradingEngine)
    engine._check_drawdown_guardrails = TradingEngine._check_drawdown_guardrails.__get__(engine, TradingEngine)
    engine._trigger_halt = TradingEngine._trigger_halt.__get__(engine, TradingEngine)
    engine._emit_guardrail_event = lambda *args: None  # Mock
    engine._save_state = lambda: None  # Mock
    engine._ensure_risk_day_bucket = lambda *args: None  # Mock

    # 11% drawdown (exceeds 10% halt threshold)
    engine.pnl = -11.0
    
    ok = engine._check_kill_switches()
    
    assert ok is False
    assert engine.running is False
    assert engine.halt_reason == "guardrail_halt_triggered"
    assert engine.guardrail_state == "halt"
