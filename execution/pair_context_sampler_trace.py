#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(row)) + "\n")


def _require_str(obj: Dict[str, Any], key: str) -> str:
    val = str(obj.get(key, "") or "").strip()
    if not val:
        raise ValueError(f"sampling_metadata.{key} is required")
    return val


def _validate_sampling_metadata(md: Dict[str, Any]) -> None:
    if not isinstance(md, dict):
        raise ValueError("sampling_metadata must be an object")
    for key in (
        "sampler_schema",
        "sampler_version",
        "sampler_policy",
        "ranker_name",
        "ranker_version",
        "ranker_score_timestamp_policy",
        "exchange_target",
        "data_source",
        "universe_filter_version",
    ):
        _require_str(md, key)
    if "candidate_universe_size" not in md:
        raise ValueError("sampling_metadata.candidate_universe_size is required")


def build_pair_context_sampler_trace(
    *,
    frame_id: str,
    frame_ts_utc: str,
    focal_pair: str,
    slot_pairs: Sequence[str],
    slot_mask: Sequence[int],
    slot_features: Any,
    sampling_metadata: Dict[str, Any],
    max_pair_width: Optional[int] = None,
    slot_ordering: str = "",
    model_slot_order_invariant: Optional[bool] = None,
) -> Dict[str, Any]:
    frame_id = str(frame_id or "").strip()
    frame_ts_utc = str(frame_ts_utc or "").strip()
    focal_pair = str(focal_pair or "").strip()
    if not frame_id:
        raise ValueError("frame_id is required")
    if not frame_ts_utc:
        raise ValueError("frame_ts_utc is required")
    if not focal_pair:
        raise ValueError("focal_pair is required")
    if not isinstance(slot_pairs, (list, tuple)) or not slot_pairs:
        raise ValueError("slot_pairs must be a non-empty sequence")
    if not isinstance(slot_mask, (list, tuple)):
        raise ValueError("slot_mask must be a sequence")
    if len(slot_mask) != len(slot_pairs):
        raise ValueError("slot_mask length must match slot_pairs length")
    if any(int(x) not in (0, 1) for x in slot_mask):
        raise ValueError("slot_mask values must be 0/1")
    pair_width = sum(int(x) for x in slot_mask)
    if pair_width <= 0:
        raise ValueError("slot_mask must include at least one present slot")
    if focal_pair not in [str(p) for p in slot_pairs]:
        raise ValueError("focal_pair must appear in slot_pairs")

    resolved_max = int(max_pair_width if max_pair_width is not None else len(slot_pairs))
    if resolved_max < len(slot_pairs):
        raise ValueError("max_pair_width cannot be smaller than slot_pairs length")

    _validate_sampling_metadata(sampling_metadata)

    trace: Dict[str, Any] = {
        "schema": "moneyfan.pair_context_sampler.v1",
        "generated_at_utc": utc_now_iso(),
        "frame_id": frame_id,
        "frame_ts_utc": frame_ts_utc,
        "focal_pair": focal_pair,
        "pair_width": int(pair_width),
        "max_pair_width": int(resolved_max),
        "slot_mask": [int(x) for x in slot_mask],
        "slot_pairs": [str(p) for p in slot_pairs],
        "slot_features": slot_features,
        "sampling_metadata": sampling_metadata,
    }
    if str(slot_ordering or "").strip():
        trace["slot_ordering"] = str(slot_ordering)
    if model_slot_order_invariant is not None:
        trace["model_slot_order_invariant"] = bool(model_slot_order_invariant)
    return trace


def write_pair_context_sampler_traces(path: Path, rows: Iterable[Dict[str, Any]], reset_output: bool = False) -> Dict[str, Any]:
    if reset_output and path.exists():
        path.unlink()
    count = 0
    for row in rows:
        append_jsonl(path, row)
        count += 1
    return {
        "schema": "moneyfan.pair_context_sampler_trace_write_summary.v1",
        "generated_at_utc": utc_now_iso(),
        "path": str(path),
        "rows_written": int(count),
        "reset_output": bool(reset_output),
    }
