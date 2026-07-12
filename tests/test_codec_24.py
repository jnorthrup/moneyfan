import sys
from unittest.mock import MagicMock

# Mock numpy before importing the codec
mock_np = MagicMock()
sys.modules['numpy'] = mock_np

# Mock ndarray behavior
class MockNDArray(list):
    def mean(self, axis=None):
        return sum(self) / len(self) if self else 0.0

    def std(self, axis=None):
        if not self:
            return 0.0
        m = self.mean()
        variance = sum((x - m)**2 for x in self) / len(self)
        return variance**0.5

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MockNDArray(super().__getitem__(item))
        return super().__getitem__(item)

    @property
    def shape(self):
        return (len(self),)

    def astype(self, dtype):
        return self

    def reshape(self, *args):
        return self

mock_np.ndarray = MockNDArray
mock_np.float32 = float
mock_np.array = lambda x, **kwargs: MockNDArray(x)
mock_np.mean = lambda x: (x.mean() if hasattr(x, 'mean') else (sum(x)/len(x) if len(x) > 0 else 0.0))
mock_np.std = lambda x: (x.std() if hasattr(x, 'std') else 0.0)
mock_np.dot = lambda x, y: sum(a*b for a, b in zip(x, y))
mock_np.sign = lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
mock_np.exp = lambda x: 2.718281828459045**x
mock_np.cumsum = lambda x: [sum(x[:i+1]) for i in range(len(x))]
mock_np.log = lambda x: 0.0 # dummy
mock_np.cov = lambda x, y: MockNDArray([[1.0, 0.0], [0.0, 1.0]]) # dummy
mock_np.var = lambda x: 1.0 # dummy
mock_np.tanh = lambda x: x # dummy

# Mock base_codec
mock_base = MagicMock()
sys.modules['codec_models.base_codec'] = mock_base
class MockBaseCodec:
    def __init__(self, config):
        self.config = config
    def validate_signal(self, confidence, direction):
        return confidence, direction
    def get_ohlcv(self, market_data, features):
        return [], [], [], []
    def record_instruments(self, **kwargs):
        pass
mock_base.BaseCodec = MockBaseCodec

# Mock mlx
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx.nn'] = MagicMock()

import pytest
# Now import the function to test from the codec
from codec_models.codec_24_zscore_stat_arb import _rolling_zscore

def test_rolling_zscore_standard():
    """Test z-score calculation with enough data and variance."""
    # prices = [1, 2, 3, 4, 5], window = 3
    # recent = [3, 4, 5]
    # mean = 4
    # variance = ((3-4)**2 + (4-4)**2 + (5-4)**2) / 3 = 2/3
    # std = (2/3)**0.5 = 0.816496580927726
    # z = (5 - 4) / 0.816496580927726 = 1.224744871...
    prices = MockNDArray([1, 2, 3, 4, 5])
    window = 3
    result = _rolling_zscore(prices, window)
    assert abs(result - 1.22474487) < 1e-6

def test_rolling_zscore_insufficient_data():
    """Test behavior when prices length is less than window."""
    # prices = [1, 2], window = 5
    # Should return 0.0
    prices = MockNDArray([1, 2])
    window = 5
    result = _rolling_zscore(prices, window)
    assert result == 0.0

def test_rolling_zscore_constant_prices():
    """Test behavior when all prices are the same (zero variance)."""
    prices = MockNDArray([10, 10, 10, 10])
    window = 4
    # mean = 10, std = 0
    # Should return 0.0 due to std < 1e-8 check
    result = _rolling_zscore(prices, window)
    assert result == 0.0

def test_rolling_zscore_near_zero_variance():
    """Test behavior when variance is extremely low but non-zero."""
    # Prices slightly different but std < 1e-8
    prices = MockNDArray([10.0, 10.0000000000001])
    window = 2
    result = _rolling_zscore(prices, window)
    assert result == 0.0
