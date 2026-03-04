from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.pair_context_sampler_trace import (
    build_pair_context_sampler_trace,
    write_pair_context_sampler_traces,
)


def _sampling_metadata():
    return {
        "sampler_schema": "moneyfan.pair_context_sampler.v1",
        "sampler_version": "pcs_v1",
        "sampler_policy": "rank_weighted_without_replacement",
        "random_seed": 123,
        "seed_scope": "frame",
        "ranker_name": "exchange_pair_target_ranker",
        "ranker_version": "epr_v3",
        "ranker_score_timestamp_policy": "point_in_time_only",
        "exchange_target": "coinbase_advanced",
        "data_source": "binance",
        "universe_filter_version": "uf_v2",
        "candidate_universe_size": 42,
    }


def test_build_pair_context_sampler_trace_validates_mask_and_metadata():
    trace = build_pair_context_sampler_trace(
        frame_id="pcs-1",
        frame_ts_utc="2026-02-26T00:00:00Z",
        focal_pair="ETH/USDT",
        slot_pairs=["ETH/USDT", "BTC/USDT", "SOL/USDT"],
        slot_mask=[1, 1, 0],
        slot_features=[{"pair": "ETH/USDT"}, {"pair": "BTC/USDT"}, {"pair": "SOL/USDT"}],
        sampling_metadata=_sampling_metadata(),
        max_pair_width=4,
        slot_ordering="focal_then_rank_desc_pair_tiebreak",
        model_slot_order_invariant=False,
    )
    assert trace["schema"] == "moneyfan.pair_context_sampler.v1"
    assert trace["pair_width"] == 2
    assert trace["max_pair_width"] == 4
    assert trace["sampling_metadata"]["exchange_target"] == "coinbase_advanced"
    assert trace["sampling_metadata"]["data_source"] == "binance"


def test_build_pair_context_sampler_trace_rejects_bad_mask_length():
    try:
        build_pair_context_sampler_trace(
            frame_id="pcs-2",
            frame_ts_utc="2026-02-26T00:00:00Z",
            focal_pair="BTC/USDT",
            slot_pairs=["BTC/USDT", "ETH/USDT"],
            slot_mask=[1],
            slot_features=[],
            sampling_metadata=_sampling_metadata(),
        )
    except ValueError as e:
        assert "slot_mask length" in str(e)
    else:
        raise AssertionError("expected mask length validation error")


def test_write_pair_context_sampler_traces_writes_jsonl_summary(tmp_path: Path):
    out = tmp_path / "sampler_trace.jsonl"
    row = build_pair_context_sampler_trace(
        frame_id="pcs-3",
        frame_ts_utc="2026-02-26T00:00:01Z",
        focal_pair="BTC/USDT",
        slot_pairs=["BTC/USDT"],
        slot_mask=[1],
        slot_features=[{"pair": "BTC/USDT"}],
        sampling_metadata=_sampling_metadata(),
    )
    summary = write_pair_context_sampler_traces(out, [row], reset_output=True)
    assert summary["rows_written"] == 1
    payload = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert payload[0]["frame_id"] == "pcs-3"
