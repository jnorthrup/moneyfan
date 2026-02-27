from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fidelity_pipeline import run_fidelity_pipeline


def test_run_fidelity_pipeline_composes_bridge_normalize_reconcile(tmp_path):
    handoff_path = tmp_path / "freqtrade_handoff.jsonl"
    bridge_state_path = tmp_path / "bridge_state.json"
    ack_log_path = tmp_path / "dispatch_ack.jsonl"
    dispatch_log_path = tmp_path / "hrm_fidelity_dispatch.jsonl"
    raw_fill_updates_path = tmp_path / "raw_trade_updates.jsonl"
    canonical_fill_events_path = tmp_path / "freqtrade_fill_events.jsonl"
    reconciliation_json_path = tmp_path / "reconcile.json"
    reconciliation_csv_path = tmp_path / "reconcile.csv"
    reject_log_path = tmp_path / "fill_rejects.jsonl"

    # Existing HRM dispatch + handoff produced by run.py/offload path
    dispatch_log_path.write_text(
        json.dumps(
            {
                "schema": "moneyfan.hrm.fidelity.dispatch.v1",
                "signal_id": "sig-pipe-1",
                "iteration": 5,
                "instrument": {"symbol": "BTCUSDT", "pair": "BTC/USDT", "side": "long", "price": 50000.0},
                "prediction": {"pred_fwd_return": 0.01, "confidence": 0.85, "trade_head_calibration_loaded": True},
                "risk": {"risk_tier": "normal"},
            }
        )
        + "\n"
    )
    handoff_path.write_text(
        json.dumps(
            {
                "schema": "moneyfan.freqtrade.handoff.v1",
                "signal_id": "sig-pipe-1",
                "pair": "BTC/USDT",
                "symbol": "BTCUSDT",
                "side": "long",
                "enter_long": 1,
                "enter_short": 0,
                "stake_fraction": 0.25,
                "stoploss": -0.02,
                "take_profit_pct": 0.03,
                "risk": {"risk_tier": "normal"},
                "model": {"confidence": 0.85, "pred_fwd_return": 0.01},
                "dispatch": {"iteration": 5, "source_mode": "paper", "source_broker_label": "freqtrade"},
            }
        )
        + "\n"
    )
    # Receiver-style raw ingest row
    raw_fill_updates_path.write_text(
        json.dumps(
            {
                "schema": "moneyfan.freqtrade.trade_update_ingest.v1",
                "received_at_utc": "2026-02-25T12:00:00Z",
                "source_path": "/trade-update",
                "payload": {
                    "schema": "freqtrade.trade_update",
                    "trade": {
                        "signal_id": "sig-pipe-1",
                        "pair": "BTC/USDT",
                        "is_short": False,
                        "is_open": False,
                        "open_rate": 50010.0,
                        "close_rate": 50410.08,
                        "profit_ratio": 0.008,
                        "id": "ft-1",
                    },
                },
            }
        )
        + "\n"
    )

    summary = run_fidelity_pipeline(
        handoff_path=handoff_path,
        bridge_state_path=bridge_state_path,
        ack_log_path=ack_log_path,
        dispatch_log_path=dispatch_log_path,
        raw_fill_updates_path=raw_fill_updates_path,
        canonical_fill_events_path=canonical_fill_events_path,
        reconciliation_json_path=reconciliation_json_path,
        reconciliation_csv_path=reconciliation_csv_path,
        webhook_url=None,
        normalizer_reject_log_path=reject_log_path,
        normalizer_dedupe=True,
        normalizer_reset_output=True,
        exchange_target="coinbase_advanced",
        data_source="binance",
    )

    assert summary["schema"] == "moneyfan.freqtrade.fidelity_pipeline_run.v1"
    assert summary["bridge"]["processed"] == 1
    assert summary["bridge"]["forwarded"] == 1
    assert summary["bridge"]["dry_run"] is True
    assert summary["normalize"]["events_written"] == 1
    assert summary["normalize"]["rejected_rows"] == 0
    assert summary["context"]["exchange_target"] == "coinbase_advanced"
    assert summary["context"]["data_source"] == "binance"

    rs = summary["reconcile_summary"]
    assert rs["dispatch_total"] == 1
    assert rs["dispatch_fully_reconciled"] == 1
    assert rs["records_with_realized_return"] == 1

    assert ack_log_path.exists()
    assert canonical_fill_events_path.exists()
    assert reconciliation_json_path.exists()
    assert reconciliation_csv_path.exists()

    report = json.loads(reconciliation_json_path.read_text())
    assert report["schema"] == "moneyfan.hrm.freqtrade.fidelity_reconciliation.v1"
    assert report["summary"]["dispatch_fully_reconciled"] == 1
    assert report["inputs"]["exchange_target"] == "coinbase_advanced"
    assert report["inputs"]["data_source"] == "binance"
    records = {r["signal_id"]: r for r in report["records"] if r.get("signal_id")}
    assert records["sig-pipe-1"]["reconcile_status"] == "ack_fill_matched"
