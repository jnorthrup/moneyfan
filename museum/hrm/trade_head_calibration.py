from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_TRADE_HEAD_CALIBRATION_CANDIDATES = [
    Path("models/trained/hrm_trade_head_calibration.json"),
    Path("models/trained/trade_head_calibration.json"),
    Path("hrm/checkpoints/hrm_trade_head_calibration.json"),
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


@dataclass(frozen=True)
class TradeHeadCalibrationBin:
    conf_min: float
    conf_max: float
    count: int
    scale: float
    median_ratio: float
    mean_ratio: float


class TradeHeadCalibrator:
    """Lightweight post-hoc calibration for HRM trade magnitude heads."""

    def __init__(self, payload: Dict[str, Any], source_path: Optional[Path] = None):
        self.payload = payload
        self.source_path = source_path
        move_payload = payload.get("move_calibration") if isinstance(payload, dict) else None
        if not isinstance(move_payload, dict):
            raise ValueError("missing move_calibration")

        self.global_scale = float(_safe_float(move_payload.get("global_scale"), 1.0))
        self.fallback_scale = float(_safe_float(move_payload.get("fallback_scale"), self.global_scale))
        self.min_scale = float(_safe_float(move_payload.get("min_scale"), 0.01))
        self.max_scale = float(_safe_float(move_payload.get("max_scale"), 5.0))
        self.bins: List[TradeHeadCalibrationBin] = []

        for row in move_payload.get("bins", []) or []:
            if not isinstance(row, dict):
                continue
            self.bins.append(
                TradeHeadCalibrationBin(
                    conf_min=float(_safe_float(row.get("conf_min"), 0.0)),
                    conf_max=float(_safe_float(row.get("conf_max"), 1.0)),
                    count=int(row.get("count", 0) or 0),
                    scale=float(_safe_float(row.get("scale"), self.global_scale)),
                    median_ratio=float(_safe_float(row.get("median_ratio"), self.global_scale)),
                    mean_ratio=float(_safe_float(row.get("mean_ratio"), self.global_scale)),
                )
            )

    @classmethod
    def load(cls, path: Path | str) -> "TradeHeadCalibrator":
        p = Path(path)
        with open(p, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("invalid calibration payload")
        return cls(payload, source_path=p)

    def scale_for_confidence(self, confidence: float) -> float:
        conf = max(0.0, min(1.0, float(_safe_float(confidence, 0.0))))
        for b in self.bins:
            if conf >= b.conf_min and conf < b.conf_max:
                return float(max(self.min_scale, min(self.max_scale, b.scale)))
        return float(max(self.min_scale, min(self.max_scale, self.fallback_scale)))

    def calibrate_move_bps(self, raw_move_bps: float, confidence: float) -> float:
        raw = max(0.0, float(_safe_float(raw_move_bps, 0.0)))
        return float(raw * self.scale_for_confidence(confidence))

    def describe(self) -> Dict[str, Any]:
        return {
            "path": str(self.source_path) if self.source_path else None,
            "global_scale": self.global_scale,
            "fallback_scale": self.fallback_scale,
            "bins": [
                {
                    "conf_min": b.conf_min,
                    "conf_max": b.conf_max,
                    "count": b.count,
                    "scale": b.scale,
                }
                for b in self.bins
            ],
        }


def discover_trade_head_calibration_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(DEFAULT_TRADE_HEAD_CALIBRATION_CANDIDATES)
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return p
    return None


def _clipped_median(values: Sequence[float], lo: float, hi: float) -> float:
    if not values:
        return 1.0
    m = float(median(values))
    return float(max(lo, min(hi, m)))


def fit_trade_head_calibration_from_trade_rows(
    trade_rows: Iterable[Dict[str, Any]],
    *,
    confidence_bin_edges: Optional[Sequence[float]] = None,
    min_bin_count: int = 30,
    min_scale: float = 0.05,
    max_scale: float = 2.0,
) -> Dict[str, Any]:
    edges = list(confidence_bin_edges or [0.0, 0.35, 0.50, 0.65, 0.80, 1.01])
    if len(edges) < 2:
        raise ValueError("confidence_bin_edges must have at least 2 points")

    samples: List[Dict[str, float]] = []
    num_rows = 0
    for row in trade_rows:
        if not isinstance(row, dict):
            continue
        num_rows += 1
        pred_move_bps = _safe_float(row.get("predicted_move_bps"), 0.0)
        exposure = _safe_float(row.get("exposure"), 0.0)
        gross_ret = _safe_float(row.get("gross_ret"), 0.0)
        conf = max(0.0, min(1.0, _safe_float(row.get("confidence"), 0.0)))
        if pred_move_bps <= 0.0 or exposure <= 0.0:
            continue
        realized_signed_move = gross_ret / max(exposure, 1e-12)
        realized_move_bps = abs(realized_signed_move) * 10000.0
        if realized_move_bps < 0.0 or not math.isfinite(realized_move_bps):
            continue
        ratio = realized_move_bps / max(pred_move_bps, 1e-9)
        if not math.isfinite(ratio) or ratio < 0.0:
            continue
        samples.append(
            {
                "confidence": conf,
                "pred_move_bps": pred_move_bps,
                "realized_move_bps": realized_move_bps,
                "ratio": ratio,
            }
        )

    if not samples:
        raise ValueError("no valid calibration samples from trade rows")

    ratio_values = [float(s["ratio"]) for s in samples]
    global_scale = _clipped_median(ratio_values, min_scale, max_scale)

    bins: List[Dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_samples = [s for s in samples if float(s["confidence"]) >= float(lo) and float(s["confidence"]) < float(hi)]
        bin_ratios = [float(s["ratio"]) for s in bin_samples]
        if len(bin_ratios) >= int(min_bin_count):
            bin_scale = _clipped_median(bin_ratios, min_scale, max_scale)
            median_ratio = float(median(bin_ratios))
            mean_ratio = float(sum(bin_ratios) / max(len(bin_ratios), 1))
        else:
            bin_scale = global_scale
            median_ratio = float(median(bin_ratios)) if bin_ratios else global_scale
            mean_ratio = float(sum(bin_ratios) / len(bin_ratios)) if bin_ratios else global_scale
        bins.append(
            {
                "conf_min": float(lo),
                "conf_max": float(hi),
                "count": int(len(bin_ratios)),
                "scale": float(bin_scale),
                "median_ratio": float(median_ratio),
                "mean_ratio": float(mean_ratio),
            }
        )

    payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "fit_stats": {
            "trade_rows_seen": int(num_rows),
            "samples_used": int(len(samples)),
            "confidence_bin_edges": [float(x) for x in edges],
            "ratio_median_raw": float(median(ratio_values)),
            "ratio_mean_raw": float(sum(ratio_values) / max(len(ratio_values), 1)),
        },
        "move_calibration": {
            "method": "confidence_binned_median_scale",
            "global_scale": float(global_scale),
            "fallback_scale": float(global_scale),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "bins": bins,
        },
    }
    return payload

