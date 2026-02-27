from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.freqtrade_fill_event_normalizer import canonicalize_fill_event, normalize_fill_jsonl


def test_canonicalize_fill_event_from_nested_trade_update():
    raw = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "signal_id": "sig-100",
            "pair": "BTC/USDT",
            "is_short": False,
            "status": "closed",
            "open_rate": 50000.0,
            "close_rate": 50500.0,
            "profit_abs": 12.34,
            "profit_ratio": 0.01,
            "id": 4242,
            "close_date": "2026-02-25T12:34:56Z",
        },
    }

    event = canonicalize_fill_event(raw)

    assert event["schema"] == "moneyfan.freqtrade.fill_event.v1"
    assert event["signal_id"] == "sig-100"
    assert event["pair"] == "BTC/USDT"
    assert event["side"] == "long"
    assert event["status"] == "closed"
    assert event["entry_price"] == 50000.0
    assert event["exit_price"] == 50500.0
    assert event["pnl_abs"] == 12.34
    assert event["pnl_pct"] == 0.01
    assert event["exchange_trade_id"] == 4242
    assert event["fill_ts_utc"] == "2026-02-25T12:34:56Z"
    assert event["source_schema"] == "freqtrade.trade_update"


def test_canonicalize_fill_event_from_receiver_ingest_wrapper():
    raw = {
        "schema": "moneyfan.freqtrade.trade_update_ingest.v1",
        "received_at_utc": "2026-02-25T20:00:00Z",
        "source_path": "/trade-update",
        "client_ip": "127.0.0.1",
        "payload": {
            "schema": "freqtrade.trade_update",
            "trade": {
                "signal_id": "sig-wrap-1",
                "pair": "ETH/USDT",
                "is_short": True,
                "is_open": False,
                "open_rate": 100.0,
                "close_rate": 95.0,
                "id": "t-wrap-1",
            },
        },
    }

    event = canonicalize_fill_event(raw)

    assert event["schema"] == "moneyfan.freqtrade.fill_event.v1"
    assert event["signal_id"] == "sig-wrap-1"
    assert event["pair"] == "ETH/USDT"
    assert event["side"] == "short"
    assert event["receiver_ingest"]["source_path"] == "/trade-update"
    assert event["receiver_ingest"]["client_ip"] == "127.0.0.1"


def test_normalize_fill_jsonl_writes_events_rejects_missing_signal_id_and_dedupes(tmp_path):
    in_path = tmp_path / "raw_fills.jsonl"
    out_path = tmp_path / "fill_events.jsonl"
    reject_path = tmp_path / "rejects.jsonl"

    valid_nested = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "signal_id": "sig-a",
            "pair": "ETH/USDT",
            "is_short": True,
            "is_open": False,
            "open_rate": 100.0,
            "close_rate": 95.0,
            "profit_abs": 5.0,
            "id": "t-1",
            "close_date": "2026-02-25T01:02:03Z",
        },
    }
    duplicate_same = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "signal_id": "sig-a",
            "pair": "ETH/USDT",
            "is_short": True,
            "status": "closed",
            "open_rate": 100.0,
            "close_rate": 95.0,
            "profit_abs": 5.0,
            "id": "t-1",
            "close_date": "2026-02-25T01:02:03Z",
        },
    }
    missing_signal = {
        "schema": "freqtrade.trade_update",
        "trade": {
            "pair": "SOL/USDT",
            "open_rate": 100.0,
            "close_rate": 101.0,
        },
    }

    with open(in_path, "w") as f:
        f.write(json.dumps(valid_nested) + "\n")
        f.write(json.dumps(duplicate_same) + "\n")
        f.write(json.dumps(missing_signal) + "\n")
        f.write("{invalid json}\n")

    summary = normalize_fill_jsonl(
        input_path=in_path,
        output_path=out_path,
        reject_log_path=reject_path,
        dedupe=True,
        reset_output=True,
    )

    assert summary["total_rows_seen"] == 4
    assert summary["events_written"] == 1
    assert summary["rejected_rows"] == 2
    assert summary["missing_signal_id"] == 1
    assert summary["parse_errors"] == 1
    assert summary["duplicates_skipped"] == 1
    assert summary["dedupe"] is True

    out_rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    assert len(out_rows) == 1
    assert out_rows[0]["schema"] == "moneyfan.freqtrade.fill_event.v1"
    assert out_rows[0]["signal_id"] == "sig-a"
    assert out_rows[0]["side"] == "short"
    assert out_rows[0]["pnl_pct"] == 0.05

    reject_rows = [json.loads(x) for x in reject_path.read_text().splitlines() if x.strip()]
    assert len(reject_rows) == 2
    reasons = {r["reason"] for r in reject_rows}
    assert reasons == {"normalization_error", "invalid_jsonl_record"}
