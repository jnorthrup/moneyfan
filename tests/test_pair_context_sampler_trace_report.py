from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.pair_context_sampler_trace import build_pair_context_sampler_trace
from execution.pair_context_sampler_trace_report import (
    build_markdown_report,
    build_pair_context_sampler_trace_report,
)


def _md():
    return {
        "sampler_schema": "moneyfan.pair_context_sampler.v1",
        "sampler_version": "pcs_v1",
        "sampler_policy": "rank_weighted_without_replacement",
        "ranker_name": "ranker_a",
        "ranker_version": "rv1",
        "ranker_score_timestamp_policy": "point_in_time_only",
        "exchange_target": "coinbase_advanced",
        "data_source": "binance",
        "universe_filter_version": "uf_v1",
        "candidate_universe_size": 100,
    }


def test_build_pair_context_sampler_trace_report_summarizes_widths_and_versions():
    rows = [
        build_pair_context_sampler_trace(
            frame_id="f1",
            frame_ts_utc="2026-02-26T00:00:00Z",
            focal_pair="BTC/USDT",
            slot_pairs=["BTC/USDT", "ETH/USDT"],
            slot_mask=[1, 1],
            slot_features=[{}, {}],
            sampling_metadata=_md(),
            max_pair_width=4,
            slot_ordering="focal_then_rank_desc_pair_tiebreak",
        ),
        build_pair_context_sampler_trace(
            frame_id="f2",
            frame_ts_utc="2026-02-26T00:00:01Z",
            focal_pair="ETH/USDT",
            slot_pairs=["ETH/USDT", "BTC/USDT", "SOL/USDT"],
            slot_mask=[1, 1, 0],
            slot_features=[{}, {}, {}],
            sampling_metadata={**_md(), "ranker_version": "rv2"},
            max_pair_width=4,
        ),
    ]
    report = build_pair_context_sampler_trace_report(rows)
    s = report["summary"]
    assert s["rows_valid"] == 2
    assert s["pair_width_stats"]["min"] == 2.0
    assert s["pair_width_stats"]["max"] == 2.0
    assert report["distributions"]["ranker_versions"]["rv1"] == 1
    assert report["distributions"]["ranker_versions"]["rv2"] == 1
    assert report["distributions"]["exchange_targets"]["coinbase_advanced"] == 2
    assert report["distributions"]["data_sources"]["binance"] == 2


def test_build_markdown_report_includes_histogram_and_context_tables():
    report = build_pair_context_sampler_trace_report([])
    md = build_markdown_report(report, trace_path="/tmp/sampler_trace.jsonl")
    assert "# Pair-Context Sampler Trace Report" in md
    assert "/tmp/sampler_trace.jsonl" in md
    assert "## Pair Width Histogram" in md
    assert "## Exchange Targets" in md
