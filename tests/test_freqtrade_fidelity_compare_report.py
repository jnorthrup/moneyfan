from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_compare_report import build_fidelity_compare_markdown_report


def test_build_fidelity_compare_markdown_report_renders_deltas_and_tables():
    baseline = {
        "summary": {
            "dispatch_total": 10,
            "dispatch_fully_reconciled": 8,
            "records_with_realized_return": 8,
            "orphan_ack_count": 1,
            "orphan_fill_count": 2,
            "fidelity_metrics": {
                "mean_abs_pred_error_bps": 20.0,
                "rmse_pred_error_bps": 25.0,
                "directional_accuracy": 0.50,
                "pearson_pred_vs_realized_bps": 0.20,
                "mean_adverse_entry_slippage_bps": 4.0,
            },
        },
        "records": [
            {
                "signal_id": "sig-1",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "BTC/USDT",
                "dispatch_side": "long",
                "abs_pred_error_bps": 20.0,
                "adverse_entry_slippage_bps": 4.0,
                "realized_return_bps": 80.0,
                "pred_error_bps": -20.0,
            },
            {
                "signal_id": "sig-2",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "ETH/USDT",
                "dispatch_side": "short",
                "abs_pred_error_bps": 10.0,
                "adverse_entry_slippage_bps": 3.0,
                "realized_return_bps": -20.0,
                "pred_error_bps": 5.0,
            },
        ],
    }
    candidate = {
        "summary": {
            "dispatch_total": 11,
            "dispatch_fully_reconciled": 9,
            "records_with_realized_return": 9,
            "orphan_ack_count": 1,
            "orphan_fill_count": 1,
            "fidelity_metrics": {
                "mean_abs_pred_error_bps": 15.0,
                "rmse_pred_error_bps": 22.0,
                "directional_accuracy": 0.625,
                "pearson_pred_vs_realized_bps": 0.35,
                "mean_adverse_entry_slippage_bps": 3.5,
            },
        },
        "records": [
            {
                "signal_id": "sig-1",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "BTC/USDT",
                "dispatch_side": "long",
                "abs_pred_error_bps": 12.0,
                "adverse_entry_slippage_bps": 5.0,
                "realized_return_bps": 90.0,
                "pred_error_bps": -12.0,
            },
            {
                "signal_id": "sig-2",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "ETH/USDT",
                "dispatch_side": "short",
                "abs_pred_error_bps": 18.0,
                "adverse_entry_slippage_bps": 2.0,
                "realized_return_bps": -10.0,
                "pred_error_bps": 12.0,
            },
        ],
    }

    md = build_fidelity_compare_markdown_report(
        baseline_report=baseline,
        candidate_report=candidate,
        baseline_path="/tmp/baseline.json",
        candidate_path="/tmp/candidate.json",
        top_n=5,
    )

    assert "# HRM/Freqtrade Fidelity Compare Report" in md
    assert "`dispatch_total`" in md and "delta=+1" in md
    assert "mean_abs_pred_error_bps" in md and "winner: `candidate`" in md
    assert "directional_accuracy" in md and "winner: `candidate`" in md
    assert "Top 5 MAE Regressions (Candidate Worse)" in md
    assert "Top 5 MAE Improvements (Candidate Better)" in md
    assert "sig-1" in md and "sig-2" in md
    assert "/tmp/baseline.json" in md and "/tmp/candidate.json" in md
