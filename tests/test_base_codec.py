
import sys
from unittest.mock import MagicMock

# Mock numpy before importing BaseCodec to avoid ModuleNotFoundError
mock_np = MagicMock()
sys.modules['numpy'] = mock_np

import pytest
from codec_models.base_codec import BaseCodec

class MockCodec(BaseCodec):
    """Concrete implementation of BaseCodec for testing."""
    def forward(self, tick_context, indicator_vec):
        return 0.0, 0.0

    def online_adapter(self, batch_data, learning_rate=1e-3):
        pass

@pytest.fixture
def codec():
    return MockCodec({"name": "test_codec"})

@pytest.mark.parametrize("conviction, direction, expected_conviction, expected_direction", [
    # Happy path: within range
    (0.5, 0.5, 0.5, 0.5),
    (0.1, -0.9, 0.1, -0.9),

    # Boundary values
    (0.0, -1.0, 0.0, -1.0),
    (1.0, 1.0, 1.0, 1.0),

    # Clipping upper bound
    (1.1, 1.1, 1.0, 1.0),
    (2.0, 5.0, 1.0, 1.0),

    # Clipping lower bound
    (-0.1, -1.1, 0.0, -1.0),
    (-5.0, -2.0, 0.0, -1.0),

    # Mixed clipping
    (1.5, -2.0, 1.0, -1.0),
    (-0.5, 2.0, 0.0, 1.0),

    # Type conversion
    ("0.5", "0.5", 0.5, 0.5),
    (1, -1, 1.0, -1.0),
    (0, 0, 0.0, 0.0),
])
def test_validate_signal(codec, conviction, direction, expected_conviction, expected_direction):
    """Test that validate_signal correctly clips and validates inputs."""
    c, d = codec.validate_signal(conviction, direction)
    assert c == expected_conviction
    assert d == expected_direction
    assert isinstance(c, float)
    assert isinstance(d, float)

def test_validate_signal_invalid_input(codec):
    """Test behavior with non-numeric inputs that can't be converted to float."""
    with pytest.raises(ValueError):
        codec.validate_signal("not a number", 0.5)
    with pytest.raises(ValueError):
        codec.validate_signal(0.5, "not a number")
