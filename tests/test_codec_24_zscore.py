import sys
from unittest.mock import MagicMock

# 1. Mock numpy before ANY imports that might use it
mock_np = MagicMock()

class MockArray(list):
    def mean(self):
        if not self: return 0.0
        return sum(self) / len(self)

    def std(self):
        if not self: return 0.0
        mu = self.mean()
        variance = sum((x - mu) ** 2 for x in self) / len(self)
        return variance ** 0.5

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return MockArray(result)
        return result

    @property
    def shape(self):
        return (len(self),)

mock_np.ndarray = MockArray
mock_np.array = lambda x, **kwargs: MockArray(x)
mock_np.float32 = float

# Mock other numpy functions used in the module or its dependencies
mock_np.mean.side_effect = lambda x: MockArray(x).mean()
mock_np.std.side_effect = lambda x: MockArray(x).std()
mock_np.var.side_effect = lambda x: MockArray(x).std() ** 2
mock_np.exp = MagicMock()
mock_np.cumsum = MagicMock()
mock_np.log = MagicMock()
mock_np.cov = MagicMock()
mock_np.dot = MagicMock()
mock_np.sign = lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
mock_np.tanh = MagicMock()
mock_np.zeros = lambda shape, **kwargs: MockArray([0.0] * (shape[0] if isinstance(shape, tuple) else shape))

sys.modules['numpy'] = mock_np

# 2. Mock mlx to avoid import errors
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx.nn'] = MagicMock()

# 3. Now import the function to test
from codec_models.codec_24_zscore_stat_arb import _rolling_zscore

def test_rolling_zscore_basic():
    """Test z-score calculation with a simple known sequence."""
    # prices = [1, 2, 3], window = 3
    # mu = 2, std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.81649658
    # last price = 3
    # z = (3 - 2) / (0.81649658 + 1e-8) ≈ 1.22474487
    prices = MockArray([1, 2, 3])
    z = _rolling_zscore(prices, 3)
    assert abs(z - 1.22474487) < 1e-6

def test_rolling_zscore_constant():
    """Test that constant input returns 0.0 (handles zero std dev)."""
    prices = MockArray([1, 1, 1])
    z = _rolling_zscore(prices, 3)
    assert abs(z) < 1e-7

def test_rolling_zscore_short_window():
    """Test that window is adjusted when prices length is smaller than window."""
    # len(prices) = 3, window = 10 -> window becomes 3
    prices = MockArray([1, 2, 3])
    z = _rolling_zscore(prices, 10)
    assert abs(z - 1.22474487) < 1e-6

def test_rolling_zscore_minimum_history():
    """Test with very short history."""
    # len(prices) = 1, window = 5 -> window becomes max(2, 1) = 2
    # seg = prices[-2:] = [1]
    # mu = 1, std = 0
    # z = (1 - 1) / (0 + 1e-8) = 0
    prices = MockArray([1])
    z = _rolling_zscore(prices, 5)
    assert z == 0.0

def test_rolling_zscore_sliding_window():
    """Test that it only uses the last 'window' bars."""
    # prices = [100, 100, 100, 1, 2, 3], window = 3
    # The '100's should be ignored.
    # mu of [1, 2, 3] is 2, std is sqrt(2/3)
    prices = MockArray([100, 100, 100, 1, 2, 3])
    z = _rolling_zscore(prices, 3)
    assert abs(z - 1.22474487) < 1e-6
