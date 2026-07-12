import sys
from unittest.mock import MagicMock

# Mock numpy before importing the module that uses it
mock_np = MagicMock()
# Add bool_ to the mock to satisfy pytest's internal checks
class MockBool: pass
mock_np.bool_ = MockBool
sys.modules['numpy'] = mock_np

import pytest
from codec_models.codec_24_zscore_stat_arb import _ema

def test_ema_constant_value():
    """Test that EMA of a constant array is the constant itself."""
    arr = [10.0, 10.0, 10.0, 10.0]
    result = _ema(arr, span=3)
    assert result == pytest.approx(10.0)

def test_ema_simple_sequence():
    """
    Test EMA with a simple sequence.
    For span=3, alpha = 2 / (3 + 1) = 0.5
    Sequence: [10, 20, 30]
    v0 = 10
    v1 = 0.5 * 20 + 0.5 * 10 = 15
    v2 = 0.5 * 30 + 0.5 * 15 = 22.5
    """
    arr = [10.0, 20.0, 30.0]
    result = _ema(arr, span=3)
    assert result == pytest.approx(22.5)

def test_ema_span_one():
    """
    Test EMA with span=1.
    alpha = 2 / (1 + 1) = 1.0
    EMA should be equal to the last element.
    """
    arr = [10.0, 20.0, 30.0, 40.0]
    result = _ema(arr, span=1)
    assert result == pytest.approx(40.0)

def test_ema_single_element():
    """Test EMA with a single element array."""
    arr = [10.0]
    result = _ema(arr, span=5)
    assert result == pytest.approx(10.0)

def test_ema_different_span():
    """
    Test EMA with span=9.
    alpha = 2 / (9 + 1) = 0.2
    Sequence: [100, 110]
    v0 = 100
    v1 = 0.2 * 110 + 0.8 * 100 = 22 + 80 = 102
    """
    arr = [100.0, 110.0]
    result = _ema(arr, span=9)
    assert result == pytest.approx(102.0)

def test_ema_empty_array():
    """Test EMA with an empty array (expected to raise IndexError per current implementation)."""
    arr = []
    with pytest.raises(IndexError):
        _ema(arr, span=5)

def test_ema_zero_span():
    """
    Test EMA with span=0.
    alpha = 2 / (0 + 1) = 2.0
    Sequence: [10, 20]
    v0 = 10
    v1 = 2.0 * 20 + (1 - 2.0) * 10 = 40 - 10 = 30
    """
    arr = [10.0, 20.0]
    result = _ema(arr, span=0)
    assert result == pytest.approx(30.0)

def test_ema_invalid_span_minus_one():
    """Test EMA with span=-1 (expected to raise ZeroDivisionError per current implementation)."""
    arr = [10.0, 20.0]
    with pytest.raises(ZeroDivisionError):
        _ema(arr, span=-1)
