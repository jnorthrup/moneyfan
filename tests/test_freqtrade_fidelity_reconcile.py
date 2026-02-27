from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_reconcile import extract_fill_view, reconcile_from_paths


def test_extract_fill_view_supports_nested_freqtrade_style_trade():
    row = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "signal_id": "hrm-nested-1",
            "pair": "ETH/USDT",
            "is_short": True,
            "is_open": False,
            "open_rate": 100.0,
            "close_rate": 90.0,
            "profit_abs": 12.5,
        },
    }

    view = extract_fill_view(row)

    assert view["signal_id"] == "hrm-nested-1"
    assert view["pair"] == "ETH/USDT"
    assert view["side"] == "short"
    assert view["status"] == "closed"
    assert view["entry_price"] == 100.0
    assert view["exit_price"] == 90.0
    assert view["pnl_abs"] == 12.5
    # Computed from open/close and short direction.
    assert view["pnl_pct"] == 0.1


def test_reconcile_from_paths_joins_dispatch_ack_and_fill_and_computes_fidelity(tmp_path):
    dispatch_path = tmp_path / "hrm_fidelity_dispatch.jsonl"
    ack_path = tmp_path / "dispatch_ack.jsonl"
    fill_path = tmp_path / "fills.jsonl"

    dispatch_rows = [
        {
            "schema": "moneyfan.hrm.fidelity.dispatch.v1",
            "ts_utc": "2026-02-25T12:00:00Z",
            "signal_id": "sig-1",
            "iteration": 10,
            "instrument": {
                "symbol": "BTCUSDT",
                "pair": "BTC/USDT",
                "side": "long",
                "price": 50000.0,
                "price_timestamp": "2026-02-25T12:00:00Z",
            },
            "prediction": {
                "pred_fwd_return": 0.01,
                "confidence": 0.8,
                "score": 12.3,
                "score_mode": "net_effective_predicted_edge_bps",
                "net_effective_predicted_edge_bps": 22.0,
                "trade_head_calibration_loaded": True,
            },
            "risk": {
                "risk_tier": "normal",
                "veto_overridden": False,
            },
        },
        {
            "schema": "moneyfan.hrm.fidelity.dispatch.v1",
            "ts_utc": "2026-02-25T12:01:00Z",
            "signal_id": "sig-2",
            "iteration": 11,
            "instrument": {"symbol": "ETHUSDT", "pair": "ETH/USDT", "side": "short", "price": 3000.0},
            "prediction": {"pred_fwd_return": -0.005, "confidence": 0.7},
            "risk": {"risk_tier": "caution"},
        },
    ]
    ack_rows = [
        {"schema": "moneyfan.freqtrade.dispatch_ack.v1", "signal_id": "sig-1", "status": "webhook_forwarded", "mode": "webhook"},
        {"schema": "moneyfan.freqtrade.dispatch_ack.v1", "signal_id": "sig-2", "status": "webhook_forwarded", "mode": "webhook"},
        {"schema": "moneyfan.freqtrade.dispatch_ack.v1", "signal_id": "sig-orphan-ack", "status": "webhook_forwarded", "mode": "webhook"},
    ]
    fill_rows = [
        {
            "schema": "moneyfan.freqtrade.fill_event.v1",
            "signal_id": "sig-1",
            "pair": "BTC/USDT",
            "side": "long",
            "status": "closed",
            "entry_price": 50010.0,
            "exit_price": 50410.08,
            "pnl_pct": 0.008,
            "pnl_abs": 8.0,
        },
        {
            "schema": "moneyfan.freqtrade.fill_event.v1",
            "signal_id": "sig-orphan-fill",
            "pair": "SOL/USDT",
            "side": "long",
            "status": "closed",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "pnl_pct": 0.01,
        },
    ]

    dispatch_path.write_text("".join(json.dumps(r) + "\n" for r in dispatch_rows))
    ack_path.write_text("".join(json.dumps(r) + "\n" for r in ack_rows))
    fill_path.write_text("".join(json.dumps(r) + "\n" for r in fill_rows))

    report = reconcile_from_paths(
        dispatch_log_path=dispatch_path,
        ack_log_path=ack_path,
        fill_log_path=fill_path,
        exchange_target="coinbase_advanced",
        data_source="binance",
    )

    summary = report["summary"]
    assert summary["dispatch_total"] == 2
    assert summary["dispatch_with_ack"] == 2
    assert summary["dispatch_with_fill"] == 1
    assert summary["dispatch_fully_reconciled"] == 1
    assert summary["orphan_ack_count"] == 1
    assert summary["orphan_fill_count"] == 1
    assert summary["records_with_realized_return"] == 1

    fm = summary["fidelity_metrics"]
    assert fm["mean_abs_pred_error_bps"] == 20.0
    assert fm["rmse_pred_error_bps"] == 20.0
    assert fm["directional_accuracy"] == 1.0
    assert fm["pearson_pred_vs_realized_bps"] is None
    assert round(float(fm["mean_adverse_entry_slippage_bps"]), 6) == 2.0

    records = {r.get("signal_id"): r for r in report["records"] if r.get("signal_id")}
    assert records["sig-1"]["reconcile_status"] == "ack_fill_matched"
    assert records["sig-1"]["predicted_return_bps"] == 100.0
    assert records["sig-1"]["realized_return_bps"] == 80.0
    assert records["sig-1"]["pred_error_bps"] == -20.0
    assert records["sig-1"]["adverse_entry_slippage_bps"] == 2.0
    assert records["sig-2"]["reconcile_status"] == "ack_no_fill"
    assert records["sig-orphan-ack"]["reconcile_status"] == "orphan_ack"
    assert records["sig-orphan-fill"]["reconcile_status"] == "orphan_fill"
    assert report["inputs"]["exchange_target"] == "coinbase_advanced"
    assert report["inputs"]["data_source"] == "binance"
