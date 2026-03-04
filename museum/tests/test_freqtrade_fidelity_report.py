from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_report import build_fidelity_markdown_report, build_timestamped_report_path


def test_build_fidelity_markdown_report_renders_metrics_and_tables():
    reconciliation = {
        "schema": "moneyfan.hrm.freqtrade.fidelity_reconciliation.v1",
        "generated_at_utc": "2026-02-25T21:00:00Z",
        "summary": {
            "dispatch_total": 3,
            "dispatch_with_fill": 2,
            "dispatch_fully_reconciled": 2,
            "records_with_realized_return": 2,
            "orphan_ack_count": 1,
            "orphan_fill_count": 0,
            "parse_errors": {"dispatch_log": 0, "ack_log": 1, "fill_log": 0},
            "fidelity_metrics": {
                "mean_abs_pred_error_bps": 12.5,
                "rmse_pred_error_bps": 14.1,
                "directional_accuracy": 0.5,
                "pearson_pred_vs_realized_bps": 0.25,
                "mean_adverse_entry_slippage_bps": 3.2,
            },
        },
        "inputs": {
            "dispatch_log_path": "/tmp/dispatch.jsonl",
            "ack_log_path": "/tmp/ack.jsonl",
            "fill_log_path": "/tmp/fills.jsonl",
        },
        "records": [
            {
                "signal_id": "sig-a",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "BTC/USDT",
                "dispatch_side": "long",
                "confidence": 0.9,
                "predicted_return_bps": 100.0,
                "realized_return_bps": 80.0,
                "pred_error_bps": -20.0,
                "abs_pred_error_bps": 20.0,
                "adverse_entry_slippage_bps": 4.0,
                "ack_status": "webhook_forwarded",
                "dispatch_price": 50000.0,
                "entry_price": 50020.0,
            },
            {
                "signal_id": "sig-b",
                "reconcile_status": "ack_fill_matched",
                "dispatch_pair": "ETH/USDT",
                "dispatch_side": "short",
                "confidence": 0.7,
                "predicted_return_bps": -40.0,
                "realized_return_bps": -35.0,
                "pred_error_bps": 5.0,
                "abs_pred_error_bps": 5.0,
                "adverse_entry_slippage_bps": 2.0,
                "ack_status": "dry_run_forwarded",
                "dispatch_price": 3000.0,
                "entry_price": 2999.4,
            },
            {
                "signal_id": "sig-orphan",
                "reconcile_status": "orphan_ack",
                "ack_status": "webhook_forwarded",
            },
        ],
    }

    md = build_fidelity_markdown_report(
        reconciliation,
        reconciliation_json_path="/tmp/reconcile.json",
        top_n=2,
    )

    assert "# HRM/Freqtrade Fidelity Report" in md
    assert "`dispatch_total`: 3" in md
    assert "`dispatch_fully_reconciled`: 2" in md
    assert "`mean_abs_pred_error_bps`: 12.500 bps" in md
    assert "`directional_accuracy`: 50.00%" in md
    assert "## Top 2 Absolute Prediction Errors (Matched)" in md
    assert "sig-a" in md and "sig-b" in md
    assert "## Sample Orphans (Top 2)" in md
    assert "sig-orphan" in md
    assert "/tmp/reconcile.json" in md


def test_build_timestamped_report_path_preserves_suffixes():
    p = Path("/tmp/hrm_freqtrade_fidelity_report.md")
    out = build_timestamped_report_path(p, stamp="20260225_211500")
    assert str(out) == "/tmp/hrm_freqtrade_fidelity_report_20260225_211500.md"

    p2 = Path("/tmp/report.backup.md")
    out2 = build_timestamped_report_path(p2, stamp="20260225_211501")
    assert str(out2) == "/tmp/report_20260225_211501.backup.md"
