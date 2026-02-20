import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hrm"))

from training_framework import (  # noqa: E402
    DEFAULT_COMPETITOR_MODELS_24,
    CandidateMetrics,
    FailFastThresholds,
    HRMTrainingFramework,
    ObjectiveWeights,
    TrainingScrews,
    build_fractal_screws_grid,
    build_data_failfast_metrics,
    compute_liquidity_flow_capture,
    compute_weighted_objective,
    evaluate_failfast,
    judge_fractal_winner,
)


def _build_candles(symbols: int = 4, rows_per_symbol: int = 10) -> pd.DataFrame:
    rows = []
    for s in range(symbols):
        symbol = f"S{s+1}"
        for i in range(rows_per_symbol):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=i),
                    "symbol": symbol,
                    "close": float(100 + i + s),
                    "volume": float(10 + i),
                }
            )
    return pd.DataFrame(rows)


def test_competitor_roster_has_24_unique_models():
    assert len(DEFAULT_COMPETITOR_MODELS_24) == 24
    assert len(set(DEFAULT_COMPETITOR_MODELS_24)) == 24


def test_prepare_expanse_respects_history_and_slot_portions():
    candles = _build_candles(symbols=4, rows_per_symbol=10)
    fw = HRMTrainingFramework(
        screws=TrainingScrews(
            history_depth_portion=0.5,
            coinbase_slot_portion=0.5,
            random_seed=7,
        )
    )

    scoped = fw.prepare_expanse(candles)

    # 50% of 4 symbols -> 2 symbols.
    assert scoped["symbol"].nunique() == 2
    # 50% of 10 rows per chosen symbol -> 5 rows each.
    counts = scoped.groupby("symbol").size().tolist()
    assert sorted(counts) == [5, 5]


def test_weighted_objective_normalizes_weights():
    weights = ObjectiveWeights(pnl=2.0, flow_capture=0.0, convergence=1.0, stability=1.0)
    score = compute_weighted_objective(
        CandidateMetrics(pnl_norm=1.0, convergence=0.0, stability=0.0),
        weights,
    )
    assert score.objective == pytest.approx(0.5, rel=1e-6)


def test_default_framework_uses_pnl_first_weights():
    fw = HRMTrainingFramework()
    w_pnl, w_flow, w_conv, w_stability = fw.objective_weights.normalized()
    assert w_pnl > w_flow > w_conv > w_stability


def test_failfast_rejects_breaches():
    thresholds = FailFastThresholds(min_rows_per_symbol=100, max_gradient_norm=1.0)
    metrics = {
        "rows_per_symbol_min": 20.0,
        "gradient_norm": 2.5,
        "timestamp_monotonic": 1.0,
    }
    result = evaluate_failfast(metrics, thresholds)
    assert not result.ok
    assert len(result.reasons) >= 2


def test_judge_candidate_requires_positive_delta_and_failfast():
    fw = HRMTrainingFramework(
        objective_weights=ObjectiveWeights(
            pnl=0.6, flow_capture=0.1, convergence=0.2, stability=0.1
        ),
        failfast_thresholds=FailFastThresholds(),
    )

    baseline = CandidateMetrics(
        pnl_norm=0.50, convergence=0.50, stability=0.50, flow_capture=0.40
    )
    candidate = CandidateMetrics(
        pnl_norm=0.70, convergence=0.60, stability=0.55, flow_capture=0.60
    )

    good_failfast = {
        "rows_per_symbol_min": 1000.0,
        "nan_ratio": 0.0,
        "timestamp_monotonic": 1.0,
        "gradient_norm": 0.5,
        "loss": 0.8,
        "epoch_seconds": 20.0,
        "max_exposure": 0.7,
        "turnover": 0.1,
        "concentration": 0.15,
        "confidence": 0.8,
        "convergence": 0.65,
        "flow_capture": 0.65,
    }

    verdict = fw.judge_candidate(
        baseline=baseline, candidate=candidate, failfast_metrics=good_failfast
    )
    assert verdict.accepted
    assert verdict.objective_delta > 0

    bad_failfast = dict(good_failfast)
    bad_failfast["gradient_norm"] = 99.0
    verdict_bad = fw.judge_candidate(
        baseline=baseline, candidate=candidate, failfast_metrics=bad_failfast
    )
    assert not verdict_bad.accepted


def test_build_data_failfast_metrics_detects_non_monotonic_timestamps():
    candles = _build_candles(symbols=1, rows_per_symbol=4)
    # Break monotonic ordering for S1.
    candles.loc[2, "timestamp"], candles.loc[3, "timestamp"] = (
        candles.loc[3, "timestamp"],
        candles.loc[2, "timestamp"],
    )

    metrics = build_data_failfast_metrics(candles)
    assert metrics["rows_per_symbol_min"] == 4.0
    assert metrics["timestamp_monotonic"] == 0.0


def test_liquidity_flow_capture_rewards_routing_to_deep_liquid_pairs():
    pair_stats = pd.DataFrame(
        [
            {"symbol": "BTC-USD", "volume_24h": 1000.0, "spread": 0.001},
            {"symbol": "ETH-USD", "volume_24h": 500.0, "spread": 0.002},
            {"symbol": "DOGE-USD", "volume_24h": 30.0, "spread": 0.050},
        ]
    )

    high_quality = compute_liquidity_flow_capture(
        pair_stats_df=pair_stats,
        trade_control={"BTC-USD": 0.8, "ETH-USD": 0.2},
    )
    low_quality = compute_liquidity_flow_capture(
        pair_stats_df=pair_stats,
        trade_control={"DOGE-USD": 1.0},
    )

    assert 0.0 <= low_quality <= 1.0
    assert 0.0 <= high_quality <= 1.0
    assert high_quality > low_quality


def test_fractal_screws_grid_builds_expected_cross_product():
    grid = build_fractal_screws_grid(
        history_depths=(0.5, 1.0),
        slot_portions=(0.25, 0.75),
        seeds=(1, 2, 3),
    )
    assert len(grid) == 2 * 2 * 3
    assert all(0.0 <= g.history_depth_portion <= 1.0 for g in grid)
    assert all(0.0 <= g.coinbase_slot_portion <= 1.0 for g in grid)


def test_judge_fractal_winner_requires_all_scenarios_to_win():
    fw = HRMTrainingFramework(
        objective_weights=ObjectiveWeights.pnl_first(),
        failfast_thresholds=FailFastThresholds(),
    )

    ff_ok = {
        "rows_per_symbol_min": 1000.0,
        "nan_ratio": 0.0,
        "timestamp_monotonic": 1.0,
        "gradient_norm": 0.2,
        "loss": 0.5,
        "epoch_seconds": 10.0,
        "max_exposure": 0.6,
        "turnover": 0.05,
        "concentration": 0.15,
        "confidence": 0.9,
        "convergence": 0.7,
        "flow_capture": 0.8,
    }

    baseline = CandidateMetrics(pnl_norm=0.45, convergence=0.45, stability=0.50, flow_capture=0.45)
    strong = CandidateMetrics(pnl_norm=0.70, convergence=0.60, stability=0.55, flow_capture=0.70)
    weak = CandidateMetrics(pnl_norm=0.40, convergence=0.50, stability=0.50, flow_capture=0.40)

    verdicts_all_win = [
        fw.judge_candidate(baseline=baseline, candidate=strong, failfast_metrics=ff_ok),
        fw.judge_candidate(baseline=baseline, candidate=strong, failfast_metrics=ff_ok),
    ]
    fractal_all = judge_fractal_winner(verdicts_all_win)
    assert fractal_all.is_fractal_winner
    assert fractal_all.win_rate == 1.0

    verdicts_mixed = [
        fw.judge_candidate(baseline=baseline, candidate=strong, failfast_metrics=ff_ok),
        fw.judge_candidate(baseline=baseline, candidate=weak, failfast_metrics=ff_ok),
    ]
    fractal_mixed = judge_fractal_winner(verdicts_mixed)
    assert not fractal_mixed.is_fractal_winner
    assert fractal_mixed.win_rate < 1.0
