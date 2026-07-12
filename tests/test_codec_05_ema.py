
import sys
from unittest.mock import MagicMock

# Mock numpy and scipy before they are imported by the codec
mock_np = MagicMock()
mock_sp = MagicMock()
mock_sp_signal = MagicMock()

sys.modules['numpy'] = mock_np
sys.modules['scipy'] = mock_sp
sys.modules['scipy.signal'] = mock_sp_signal

# Also need to mock mlx since it's imported in base_codec or codec_05
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx.nn'] = MagicMock()

import numpy as np
from codec_models.codec_05_pairs_trading import Codec05

def test_ema_logic():
    # Setup mock behavior for numpy
    # prices[0] access
    mock_prices = MagicMock()
    mock_prices.__len__.return_value = 100
    mock_prices.__getitem__.return_value = 10.0 # prices[0]

    # Mock np.zeros
    mock_np.zeros.return_value = [0.0] * 100

    # Mock np.array for zi
    mock_np.array.side_effect = lambda x: x

    codec = Codec05()

    # We want to check if it calls lfilter when HAS_SCIPY is True
    import codec_models.codec_05_pairs_trading as codec_mod
    codec_mod.HAS_SCIPY = True

    market_data = {'price': 10.0}
    features = [0.01] * 64

    # Mock get_ohlcv to return our mock_prices
    codec.get_ohlcv = MagicMock(return_value=(mock_prices, mock_prices, mock_prices, mock_prices))
    # Mock validate_signal
    codec.validate_signal = MagicMock(return_value=(0.5, 1.0))
    # Mock _ar1_beta
    codec_mod._ar1_beta = MagicMock(return_value=-0.1)

    # Mock spread.mean() and spread.std()
    mock_spread = MagicMock()
    mock_spread.__getitem__.return_value = mock_spread # spread[-self.window:]
    mock_spread.mean.return_value = 0.0
    mock_spread.std.return_value = 1.0
    mock_spread.__sub__.return_value = mock_spread

    # Mock lfilter to return (mock_spread, None)
    codec_mod.lfilter.return_value = (mock_spread, None)

    codec.forward(market_data, features)

    # Verify lfilter was called
    assert codec_mod.lfilter.called
    print("lfilter call verified")

if __name__ == "__main__":
    try:
        test_ema_logic()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
