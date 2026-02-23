from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from train import sample_stochastic_calendar_extent_df


def _sample_df(days: int = 40, rows_per_day: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=days * rows_per_day, freq=f"{24//rows_per_day}h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTCUSDT"] * len(ts),
            "close": np.linspace(100.0, 200.0, len(ts)),
        }
    )


def test_calendar_extent_sampling_applies_and_reports_span():
    np.random.seed(7)
    df = _sample_df(days=60, rows_per_day=6)

    sampled, meta = sample_stochastic_calendar_extent_df(df, min_days=7, max_days=21, min_rows=24)

    assert not sampled.empty
    assert meta["mode"] == "calendar_days"
    assert meta["applied"] is True
    assert 7 <= int(meta["span_days_requested"]) <= 21
    assert meta["rows_after"] == len(sampled)
    assert meta["rows_before"] == len(df)
    assert float(meta["available_span_days_total"]) > 0.0
    assert meta["extent_start"] is not None
    assert meta["extent_end"] is not None
    # Actual span can be shorter than requested near boundaries but should be positive.
    assert float(meta["span_days_actual"]) >= 0.0
    assert "span_days_target_met" in meta


def test_calendar_extent_sampling_disabled_is_noop():
    df = _sample_df(days=10, rows_per_day=4)

    sampled, meta = sample_stochastic_calendar_extent_df(df, min_days=0, max_days=0, min_rows=10)

    assert len(sampled) == len(df)
    assert meta["applied"] is False
    assert meta["mode"] == "disabled"


def test_calendar_extent_sampling_strict_flags_insufficient_history():
    df = _sample_df(days=1, rows_per_day=24)

    sampled, meta = sample_stochastic_calendar_extent_df(
        df,
        min_days=7,
        max_days=30,
        min_rows=10,
        strict_min_days=True,
    )

    assert len(sampled) == len(df)
    assert meta["applied"] is False
    assert meta["mode"] == "calendar_days"
    assert int(meta["span_days_requested"]) == 7
    assert float(meta["available_span_days_total"]) < 7.0
    assert str(meta["fallback_reason"]).startswith("available_span_below_min_days:")
