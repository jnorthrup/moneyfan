from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
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
        from execution.guardrail_actions import create_guardrail_action_mapper_from_config
        engine.guardrail_action_mapper = create_guardrail_action_mapper_from_config(cfg)
        engine._current_guardrail_action = None
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

    # At peak - no drawdown
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
    engine._append_jsonl = TradingEngine._append_jsonl.__get__(engine, TradingEngine)
    engine._emit_guardrail_event = TradingEngine._emit_guardrail_event.__get__(engine, TradingEngine)
    engine._json_safe = TradingEngine._json_safe.__get__(engine, TradingEngine)

    # Trigger a transition
    engine.pnl = -5.0
    engine._check_drawdown_guardrails()

    assert events_path.exists()
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

    engine._risk_snapshot = TradingEngine._risk_snapshot.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._portfolio_drawdown_pct = TradingEngine._portfolio_drawdown_pct.__get__(engine, TradingEngine)
    engine._check_kill_switches = TradingEngine._check_kill_switches.__get__(engine, TradingEngine)
    engine._check_drawdown_guardrails = TradingEngine._check_drawdown_guardrails.__get__(engine, TradingEngine)
    engine._trigger_halt = TradingEngine._trigger_halt.__get__(engine, TradingEngine)
    engine._emit_guardrail_event = lambda *args: None
    engine._save_state = lambda: None
    engine._ensure_risk_day_bucket = lambda *args: None

    # 11% drawdown (exceeds 10% halt threshold)
    engine.pnl = -11.0

    ok = engine._check_kill_switches()

    assert ok is False
    assert engine.running is False
    assert engine.halt_reason == "guardrail_halt_triggered"
    assert engine.guardrail_state == "halt"


# ---------------------------------------------------------------------------
# Phase 2: Action wiring tests
# ---------------------------------------------------------------------------

def test_update_guardrail_action_disabled(engine_stub):
    """When guardrail is disabled, _update_guardrail_action sets no active action."""
    engine = engine_stub()  # guardrail_enabled=False by default
    engine._update_guardrail_action()
    assert engine._current_guardrail_action is None


def test_effective_top_k_no_guardrail(engine_stub):
    """When guardrail is disabled, effective top-k equals config top-k."""
    engine = engine_stub(top_k=3)
    engine._update_guardrail_action()
    assert engine._get_effective_top_k() == 3


def test_effective_top_k_derisk(engine_stub):
    """Under derisk state, effective top-k is scaled down."""
    engine = engine_stub(guardrail_enabled=True, top_k=4)
    engine.guardrail_state = "derisk"
    engine._update_guardrail_action()
    # Default derisk_top_k_scale=0.5, so 4 * 0.5 = 2
    assert engine._get_effective_top_k() == 2


def test_effective_top_k_halt(engine_stub):
    """Under halt state, effective top-k is 0 (no new entries)."""
    engine = engine_stub(guardrail_enabled=True, top_k=5)
    engine.guardrail_state = "halt"
    engine._update_guardrail_action()
    assert engine._get_effective_top_k() == 0


def test_effective_signal_threshold_derisk(engine_stub):
    """Under derisk state, effective signal threshold is raised."""
    engine = engine_stub(guardrail_enabled=True, signal_threshold=0.65)
    engine.guardrail_state = "derisk"
    engine._update_guardrail_action()
    # Default derisk_confidence_boost=0.10, so 0.65 + 0.10 = 0.75
    assert engine._get_effective_signal_threshold() == pytest.approx(0.75)


def test_effective_position_size_scale_normal(engine_stub):
    """In normal state with guardrails enabled, position size scale is 1.0."""
    engine = engine_stub(guardrail_enabled=True)
    engine.guardrail_state = "normal"
    engine._update_guardrail_action()
    assert engine._get_effective_position_size_scale() == pytest.approx(1.0)


def test_effective_position_size_scale_derisk(engine_stub):
    """Under derisk state, position size scale is reduced to 50%."""
    engine = engine_stub(guardrail_enabled=True)
    engine.guardrail_state = "derisk"
    engine._update_guardrail_action()
    # Default derisk_position_size_scale=0.5
    assert engine._get_effective_position_size_scale() == pytest.approx(0.5)


def test_should_allow_new_entries_disabled(engine_stub):
    """When guardrail is disabled, new entries are always allowed."""
    engine = engine_stub()
    allowed, reason = engine._should_allow_new_entries()
    assert allowed is True
    # mapper returns "allowed" for normal state regardless of guardrail_enabled flag
    assert reason == "allowed"


def test_should_allow_new_entries_halt(engine_stub):
    """Under halt state, new entries are blocked."""
    engine = engine_stub(guardrail_enabled=True)
    engine.guardrail_state = "halt"
    allowed, reason = engine._should_allow_new_entries()
    assert allowed is False
    assert "halt" in reason


def test_should_allow_new_entries_normal(engine_stub):
    """Under normal state with positions available, new entries are allowed."""
    engine = engine_stub(guardrail_enabled=True, max_positions=10)
    engine.guardrail_state = "normal"
    allowed, reason = engine._should_allow_new_entries()
    assert allowed is True


# ---------------------------------------------------------------------------
# Phase 3: State persistence and resume-flow tests
# ---------------------------------------------------------------------------

def _build_saveable_engine(engine_stub, tmp_path, **overrides):
    """Helper that builds a stub with all fields needed to call _save_state."""
    engine = engine_stub(**overrides)
    engine._risk_snapshot = TradingEngine._risk_snapshot.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._save_state = TradingEngine._save_state.__get__(engine, TradingEngine)
    return engine


def test_save_state_includes_guardrail_fields(engine_stub, tmp_path):
    """_save_state must persist guardrail_state and candidate window fields."""
    engine = _build_saveable_engine(engine_stub, tmp_path)
    engine.guardrail_state = "derisk"
    engine.guardrail_candidate_state = "halt"
    engine.guardrail_candidate_iterations = 2

    engine._save_state()

    with open(engine.state_path) as f:
        saved = json.load(f)

    assert saved["guardrail_state"] == "derisk"
    assert saved["guardrail_candidate_state"] == "halt"
    assert saved["guardrail_candidate_iterations"] == 2


def test_load_state_restores_guardrail_fields(engine_stub, tmp_path):
    """After save/load cycle, guardrail state-machine fields are correctly restored."""
    state_path = tmp_path / "state.json"
    state = {
        "mode": "paper",
        "pnl": -8.0,
        "positions": {},
        "trades": [],
        "orders": [],
        "latest_prices": {},
        "latest_price_timestamps": {},
        "current_iteration": 7,
        "symbol_cooldown_until_iteration": {},
        "halt_reason": None,
        "guardrail_state": "warn",
        "guardrail_candidate_state": "derisk",
        "guardrail_candidate_iterations": 1,
        "risk_state": {
            "peak_equity": 100.0,
            "risk_day_utc": "2026-03-09",
            "risk_day_start_equity": 100.0,
            "risk_day_realized_pnl": -8.0,
        },
        "timestamp": "2026-03-09T20:00:00",
    }
    state_path.write_text(json.dumps(state))

    engine = engine_stub(resume_state=True)
    engine.state_path = state_path
    engine.config.state_path = str(state_path)
    engine._ensure_risk_day_bucket = lambda *args: None
    engine._load_state = TradingEngine._load_state.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._utc_day_key = TradingEngine._utc_day_key
    engine._load_state()

    assert engine.guardrail_state == "warn"
    assert engine.guardrail_candidate_state == "derisk"
    assert engine.guardrail_candidate_iterations == 1
    assert engine.current_iteration == 7
    assert engine.pnl == pytest.approx(-8.0)


def test_load_state_rejects_invalid_guardrail_state(engine_stub, tmp_path):
    """Unrecognised saved guardrail_state values are silently ignored."""
    state_path = tmp_path / "state.json"
    state = {
        "mode": "paper",
        "pnl": 0.0,
        "positions": {},
        "trades": [],
        "orders": [],
        "latest_prices": {},
        "latest_price_timestamps": {},
        "current_iteration": 0,
        "symbol_cooldown_until_iteration": {},
        "halt_reason": None,
        "guardrail_state": "UNKNOWN_GARBAGE",
        "guardrail_candidate_state": "also_invalid",
        "guardrail_candidate_iterations": "not_an_int",
        "risk_state": {
            "peak_equity": 100.0,
            "risk_day_utc": "2026-03-09",
            "risk_day_start_equity": 100.0,
            "risk_day_realized_pnl": 0.0,
        },
        "timestamp": "2026-03-09T20:00:00",
    }
    state_path.write_text(json.dumps(state))

    engine = engine_stub(resume_state=True)
    engine.state_path = state_path
    engine.config.state_path = str(state_path)
    engine._ensure_risk_day_bucket = lambda *args: None
    engine._load_state = TradingEngine._load_state.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._utc_day_key = TradingEngine._utc_day_key
    engine._load_state()

    # Invalid values should leave the defaults intact
    assert engine.guardrail_state == "normal"
    assert engine.guardrail_candidate_state == "normal"
    assert engine.guardrail_candidate_iterations == 0


def test_halt_resume_blocks_trading(engine_stub, tmp_path):
    """Engine with respect_saved_halt_state=True loads halt correctly from saved state."""
    state_path = tmp_path / "state.json"
    state = {
        "mode": "paper",
        "pnl": -12.0,
        "positions": {},
        "trades": [],
        "orders": [],
        "latest_prices": {},
        "latest_price_timestamps": {},
        "current_iteration": 5,
        "symbol_cooldown_until_iteration": {},
        "halt_reason": "guardrail_halt_triggered",
        "guardrail_state": "halt",
        "guardrail_candidate_state": "halt",
        "guardrail_candidate_iterations": 3,
        "risk_state": {
            "peak_equity": 100.0,
            "risk_day_utc": "2026-03-09",
            "risk_day_start_equity": 100.0,
            "risk_day_realized_pnl": -12.0,
            "halt_reason": "guardrail_halt_triggered",
        },
        "timestamp": "2026-03-09T20:00:00",
    }
    state_path.write_text(json.dumps(state))

    engine = engine_stub(
        resume_state=True,
        respect_saved_halt_state=True,
    )
    engine.state_path = state_path
    engine.config.state_path = str(state_path)
    engine._ensure_risk_day_bucket = lambda *args: None
    engine._load_state = TradingEngine._load_state.__get__(engine, TradingEngine)
    engine._equity = TradingEngine._equity.__get__(engine, TradingEngine)
    engine._utc_day_key = TradingEngine._utc_day_key
    engine._load_state()

    assert engine.halt_reason == "guardrail_halt_triggered"
    assert engine.guardrail_state == "halt"
    # run() checks: if halt_reason and respect_saved_halt_state -> refuse to start
    assert bool(engine.halt_reason) is True
    assert bool(engine.config.respect_saved_halt_state) is True


def test_state_schema_completeness(engine_stub, tmp_path):
    """Saved state JSON contains all required guardrail schema keys."""
    engine = _build_saveable_engine(engine_stub, tmp_path)
    engine.guardrail_state = "warn"
    engine.guardrail_candidate_state = "derisk"
    engine.guardrail_candidate_iterations = 1
    engine._save_state()

    with open(engine.state_path) as f:
        saved = json.load(f)

    for required_key in (
        "guardrail_state",
        "guardrail_candidate_state",
        "guardrail_candidate_iterations",
        "halt_reason",
        "risk_state",
    ):
        assert required_key in saved, f"Missing required state schema key: {required_key}"

    risk = saved["risk_state"]
    for required_risk_key in ("peak_equity", "risk_day_utc", "drawdown_pct", "halt_reason"):
        assert required_risk_key in risk, (
            f"Missing required risk_state key: {required_risk_key}"
        )
