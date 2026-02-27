from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.pair_context_sampler_conformance import validate_muxer_sampler_conformance


def test_validate_muxer_sampler_conformance_passes_on_minimal_valid_rows():
    rows = [
        {"pair": "BTC/USDT", "symbol": "BTCUSDT", "ts_utc": "2026-02-26T00:00:00Z", "price": 50000.0},
        {"pair": "ETH/USDT", "symbol": "ETHUSDT", "ts_utc": "2026-02-26T00:00:01Z", "price": 3000.0},
    ]
    report = validate_muxer_sampler_conformance(
        rows,
        exchange_target="coinbase_advanced",
        data_source="binance",
        require_monotonic_ts=True,
    )
    assert report["result"] == "pass"
    assert report["summary"]["missing_required_columns"] == []
    assert report["summary"]["null_violations"] == 0
    assert report["summary"]["timestamp_violations"] == 0
    assert report["context"]["exchange_target"] == "coinbase_advanced"


def test_validate_muxer_sampler_conformance_fails_missing_required_and_bad_ts():
    rows = [
        {"pair": "BTC/USDT", "ts_utc": "bad-ts"},  # missing symbol + bad timestamp
        {"symbol": "ETHUSDT", "ts_utc": ""},  # missing pair + empty timestamp
    ]
    report = validate_muxer_sampler_conformance(rows)
    assert report["result"] == "fail"
    assert "symbol" in report["summary"]["missing_required_columns"] or report["summary"]["null_violations"] > 0
    assert report["summary"]["timestamp_violations"] >= 1


def test_validate_muxer_sampler_conformance_detects_non_monotonic_ts():
    rows = [
        {"pair": "BTC/USDT", "symbol": "BTCUSDT", "ts_utc": "2026-02-26T00:00:02Z"},
        {"pair": "BTC/USDT", "symbol": "BTCUSDT", "ts_utc": "2026-02-26T00:00:01Z"},
    ]
    report = validate_muxer_sampler_conformance(rows, require_monotonic_ts=True)
    assert report["result"] == "fail"
    assert report["summary"]["monotonic_timestamp_violations"] == 1
