import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Add hrm to path for direct import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hrm'))

from scorecard import (
    ScorecardEntry,
    Scorecard,
    DecisionPayload,
    compute_alpha,
    compute_pnl,
    compute_win_rate,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_convergence,
    score_decision,
    AggregationWindow,
)


class TestScorecardEntry:
    def test_scorecard_entry_is_frozen_dataclass(self):
        entry = ScorecardEntry(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            alpha=0.05,
            pnl=1000.0,
            win_rate=0.65,
            sharpe=1.8,
            convergence=0.75,
            regime="bull",
            signal=1,
            confidence=0.85,
            position_size=0.5,
        )
        assert is_dataclass(entry)
        with pytest.raises(Exception):
            entry.alpha = 0.10

    def test_scorecard_entry_required_fields(self):
        with pytest.raises(TypeError):
            ScorecardEntry(timestamp=datetime.now())

    def test_scorecard_entry_to_dict(self):
        ts = datetime.now()
        entry = ScorecardEntry(
            timestamp=ts,
            symbol="ETH-USD",
            alpha=0.03,
            pnl=500.0,
            win_rate=0.55,
            sharpe=1.2,
            convergence=0.60,
            regime="neutral",
            signal=0,
            confidence=0.70,
            position_size=0.0,
        )
        d = asdict(entry)
        assert d["symbol"] == "ETH-USD"
        assert d["alpha"] == 0.03
        assert d["regime"] == "neutral"


class TestScorecard:
    def test_scorecard_initializes_empty(self):
        sc = Scorecard()
        assert len(sc.entries) == 0

    def test_scorecard_add_entry(self):
        sc = Scorecard()
        entry = ScorecardEntry(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            alpha=0.05,
            pnl=1000.0,
            win_rate=0.65,
            sharpe=1.8,
            convergence=0.75,
            regime="bull",
            signal=1,
            confidence=0.85,
            position_size=0.5,
        )
        sc.add(entry)
        assert len(sc.entries) == 1

    def test_scorecard_aggregate_by_window_decision(self):
        sc = Scorecard()
        base_ts = datetime.now()
        for i in range(5):
            sc.add(ScorecardEntry(
                timestamp=base_ts + timedelta(minutes=i),
                symbol="BTC-USD",
                alpha=0.01 * (i + 1),
                pnl=100.0 * (i + 1),
                win_rate=0.50 + 0.05 * i,
                sharpe=1.0 + 0.1 * i,
                convergence=0.50 + 0.05 * i,
                regime="bull",
                signal=1,
                confidence=0.80,
                position_size=0.5,
            ))
        summary = sc.aggregate(window=AggregationWindow.DECISION)
        assert "mean_alpha" in summary
        assert "total_pnl" in summary
        assert "mean_sharpe" in summary

    def test_scorecard_aggregate_by_window_episode(self):
        sc = Scorecard()
        base_ts = datetime.now()
        for i in range(10):
            sc.add(ScorecardEntry(
                timestamp=base_ts + timedelta(hours=i),
                symbol="ETH-USD",
                alpha=0.02,
                pnl=200.0,
                win_rate=0.60,
                sharpe=1.5,
                convergence=0.70,
                regime="bear" if i < 5 else "bull",
                signal=-1 if i < 5 else 1,
                confidence=0.75,
                position_size=0.3,
            ))
        summary = sc.aggregate(window=AggregationWindow.EPISODE)
        assert summary["entry_count"] == 10

    def test_scorecard_aggregate_by_window_run(self):
        sc = Scorecard()
        base_ts = datetime.now()
        for i in range(20):
            sc.add(ScorecardEntry(
                timestamp=base_ts + timedelta(days=i),
                symbol="BTC-USD",
                alpha=0.015,
                pnl=150.0,
                win_rate=0.58,
                sharpe=1.3,
                convergence=0.65,
                regime="neutral",
                signal=0,
                confidence=0.60,
                position_size=0.0,
            ))
        summary = sc.aggregate(window=AggregationWindow.RUN)
        assert "regime_distribution" in summary

    def test_scorecard_filter_by_symbol(self):
        sc = Scorecard()
        base_ts = datetime.now()
        sc.add(ScorecardEntry(
            timestamp=base_ts,
            symbol="BTC-USD",
            alpha=0.05, pnl=100.0, win_rate=0.6, sharpe=1.5,
            convergence=0.7, regime="bull", signal=1, confidence=0.8, position_size=0.5,
        ))
        sc.add(ScorecardEntry(
            timestamp=base_ts,
            symbol="ETH-USD",
            alpha=0.03, pnl=50.0, win_rate=0.55, sharpe=1.2,
            convergence=0.6, regime="bull", signal=1, confidence=0.75, position_size=0.3,
        ))
        filtered = sc.filter(symbol="BTC-USD")
        assert len(filtered.entries) == 1
        assert filtered.entries[0].symbol == "BTC-USD"

    def test_scorecard_filter_by_regime(self):
        sc = Scorecard()
        base_ts = datetime.now()
        for regime in ["bull", "bear", "neutral", "bull"]:
            sc.add(ScorecardEntry(
                timestamp=base_ts,
                symbol="BTC-USD",
                alpha=0.02, pnl=100.0, win_rate=0.55, sharpe=1.2,
                convergence=0.6, regime=regime, signal=1, confidence=0.7, position_size=0.4,
            ))
        filtered = sc.filter(regime="bull")
        assert len(filtered.entries) == 2


class TestComputeAlpha:
    def test_compute_alpha_positive_excess_return(self):
        returns = np.array([0.10, 0.05, 0.08, 0.12])
        benchmark = np.array([0.06, 0.04, 0.05, 0.07])
        alpha = compute_alpha(returns, benchmark)
        assert alpha == pytest.approx(0.0325, rel=1e-4)

    def test_compute_alpha_negative_excess_return(self):
        returns = np.array([0.02, 0.01, 0.03, 0.02])
        benchmark = np.array([0.05, 0.04, 0.06, 0.05])
        alpha = compute_alpha(returns, benchmark)
        assert alpha < 0

    def test_compute_alpha_zero_benchmark(self):
        returns = np.array([0.05, 0.03, 0.07])
        benchmark = np.array([0.0, 0.0, 0.0])
        alpha = compute_alpha(returns, benchmark)
        assert alpha == pytest.approx(np.mean(returns), rel=1e-6)

    def test_compute_alpha_mismatched_lengths_raises(self):
        returns = np.array([0.05, 0.03])
        benchmark = np.array([0.02])
        with pytest.raises(ValueError):
            compute_alpha(returns, benchmark)


class TestComputePnl:
    def test_compute_pnl_long_profit(self):
        positions = np.array([100, 100, 100])
        prices = np.array([100.0, 105.0, 110.0])
        pnl = compute_pnl(positions, prices)
        assert pnl == pytest.approx(1000.0, rel=1e-6)

    def test_compute_pnl_short_profit(self):
        positions = np.array([-100, -100, -100])
        prices = np.array([100.0, 95.0, 90.0])
        pnl = compute_pnl(positions, prices)
        assert pnl == pytest.approx(1000.0, rel=1e-6)

    def test_compute_pnl_mixed_positions(self):
        positions = np.array([100, 0, -50])
        prices = np.array([100.0, 110.0, 105.0])
        pnl = compute_pnl(positions, prices)
        assert isinstance(pnl, float)

    def test_compute_pnl_empty_arrays(self):
        positions = np.array([])
        prices = np.array([])
        pnl = compute_pnl(positions, prices)
        assert pnl == 0.0


class TestComputeWinRate:
    def test_compute_win_rate_all_winners(self):
        trade_returns = np.array([0.05, 0.10, 0.03, 0.08])
        wr = compute_win_rate(trade_returns)
        assert wr == 1.0

    def test_compute_win_rate_all_losers(self):
        trade_returns = np.array([-0.05, -0.10, -0.03, -0.08])
        wr = compute_win_rate(trade_returns)
        assert wr == 0.0

    def test_compute_win_rate_mixed(self):
        trade_returns = np.array([0.05, -0.02, 0.10, -0.08, 0.0, 0.03])
        wr = compute_win_rate(trade_returns)
        assert wr == pytest.approx(0.5, rel=1e-6)

    def test_compute_win_rate_empty_returns_zero(self):
        trade_returns = np.array([])
        wr = compute_win_rate(trade_returns)
        assert wr == 0.0


class TestComputeSharpe:
    def test_compute_sharpe_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.01])
        sharpe = compute_sharpe(returns, risk_free=0.0)
        assert sharpe > 0

    def test_compute_sharpe_with_risk_free_rate(self):
        returns = np.array([0.05, 0.06, 0.04, 0.07])
        sharpe = compute_sharpe(returns, risk_free=0.02)
        assert sharpe > 0

    def test_compute_sharpe_zero_std_returns_inf(self):
        returns = np.array([0.05, 0.05, 0.05, 0.05])
        sharpe = compute_sharpe(returns, risk_free=0.0)
        assert sharpe == float("inf")

    def test_compute_sharpe_negative_returns(self):
        returns = np.array([-0.01, -0.02, -0.015, -0.005])
        sharpe = compute_sharpe(returns, risk_free=0.0)
        assert sharpe < 0


class TestComputeSortino:
    def test_compute_sortino_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.01])
        sortino = compute_sortino(returns, risk_free=0.0)
        assert sortino > 0

    def test_compute_sortino_only_positive_returns_inf(self):
        returns = np.array([0.05, 0.06, 0.04, 0.07])
        sortino = compute_sortino(returns, risk_free=0.0)
        assert sortino == float("inf")

    def test_compute_sortino_mixed_returns(self):
        returns = np.array([0.10, -0.05, 0.08, -0.03, 0.06])
        sortino = compute_sortino(returns, risk_free=0.0)
        sharpe = compute_sharpe(returns, risk_free=0.0)
        assert sortino > sharpe

    def test_compute_sortino_with_risk_free_rate(self):
        returns = np.array([0.05, -0.02, 0.08, -0.01, 0.04])
        sortino = compute_sortino(returns, risk_free=0.01)
        assert isinstance(sortino, float)


class TestComputeMaxDrawdown:
    def test_compute_max_drawdown_no_drawdown(self):
        cumulative = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        dd = compute_max_drawdown(cumulative)
        assert dd == 0.0

    def test_compute_max_drawdown_single_drawdown(self):
        cumulative = np.array([100.0, 90.0, 95.0, 100.0])
        dd = compute_max_drawdown(cumulative)
        assert dd == pytest.approx(0.10, rel=1e-6)

    def test_compute_max_drawdown_multiple_drawdowns(self):
        cumulative = np.array([100.0, 110.0, 95.0, 105.0, 85.0, 90.0])
        dd = compute_max_drawdown(cumulative)
        assert dd == pytest.approx(0.15, rel=1e-6)

    def test_compute_max_drawdown_empty_returns_zero(self):
        cumulative = np.array([])
        dd = compute_max_drawdown(cumulative)
        assert dd == 0.0


class TestComputeConvergence:
    def test_compute_convergence_full_agreement(self):
        signals = np.array([1, 1, 1, 1])
        confidences = np.array([0.9, 0.85, 0.95, 0.88])
        conv = compute_convergence(signals, confidences)
        assert conv == pytest.approx(1.0, rel=1e-6)

    def test_compute_convergence_no_agreement(self):
        signals = np.array([1, -1, 1, -1])
        confidences = np.array([0.9, 0.9, 0.9, 0.9])
        conv = compute_convergence(signals, confidences)
        assert conv == pytest.approx(0.0, rel=1e-6)

    def test_compute_convergence_weighted_by_confidence(self):
        signals = np.array([1, 1, 1, -1])
        confidences = np.array([0.9, 0.9, 0.9, 0.3])
        conv = compute_convergence(signals, confidences)
        assert 0.5 < conv < 1.0

    def test_compute_convergence_empty_returns_zero(self):
        signals = np.array([])
        confidences = np.array([])
        conv = compute_convergence(signals, confidences)
        assert conv == 0.0


class TestDecisionPayload:
    def test_decision_payload_is_dataclass(self):
        payload = DecisionPayload(
            signal=1,
            confidence=0.85,
            regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.8},
            timestamp=datetime.now(),
        )
        assert is_dataclass(payload)
        assert payload.signal == 1
        assert payload.confidence == 0.85

    def test_decision_payload_to_dict(self):
        ts = datetime.now()
        payload = DecisionPayload(
            signal=-1,
            confidence=0.70,
            regime="bear",
            metrics={"pnl": -500.0, "win_rate": 0.40},
            timestamp=ts,
        )
        d = asdict(payload)
        assert d["regime"] == "bear"
        assert d["metrics"]["pnl"] == -500.0


class TestScoreDecision:
    def test_score_decision_high_alpha_high_sharpe(self):
        payload = DecisionPayload(
            signal=1,
            confidence=0.90,
            regime="bull",
            metrics={"alpha": 0.08, "sharpe": 2.5},
            timestamp=datetime.now(),
        )
        outcomes = {"realized_pnl": 5000.0, "realized_alpha": 0.07}
        score = score_decision(payload, outcomes)
        assert 0.0 <= score <= 1.0

    def test_score_decision_low_confidence_penalty(self):
        payload_high = DecisionPayload(
            signal=1, confidence=0.90, regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.5},
            timestamp=datetime.now(),
        )
        payload_low = DecisionPayload(
            signal=1, confidence=0.40, regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.5},
            timestamp=datetime.now(),
        )
        outcomes = {"realized_pnl": 1000.0, "realized_alpha": 0.04}
        score_high = score_decision(payload_high, outcomes)
        score_low = score_decision(payload_low, outcomes)
        assert score_high > score_low

    def test_score_decision_signal_alignment_bonus(self):
        payload_aligned = DecisionPayload(
            signal=1, confidence=0.80, regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.5},
            timestamp=datetime.now(),
        )
        payload_misaligned = DecisionPayload(
            signal=-1, confidence=0.80, regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.5},
            timestamp=datetime.now(),
        )
        outcomes_bull = {"realized_pnl": 1000.0, "realized_alpha": 0.04, "market_direction": 1}
        score_aligned = score_decision(payload_aligned, outcomes_bull)
        score_misaligned = score_decision(payload_misaligned, outcomes_bull)
        assert score_aligned > score_misaligned

    def test_score_decision_missing_outcomes_returns_zero(self):
        payload = DecisionPayload(
            signal=1, confidence=0.80, regime="bull",
            metrics={"alpha": 0.05, "sharpe": 1.5},
            timestamp=datetime.now(),
        )
        score = score_decision(payload, {})
        assert score == 0.0


class TestAggregationWindow:
    def test_aggregation_window_enum_values(self):
        assert AggregationWindow.DECISION.value == "decision"
        assert AggregationWindow.EPISODE.value == "episode"
        assert AggregationWindow.RUN.value == "run"
