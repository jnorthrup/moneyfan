"""
Tests for the create_hrm_model factory function in Torch HRM.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    # This is to allow structural verification even if torch is missing
    torch = None

import pytest
from hrm.torch_hrm import (
    TorchHrmModule,
    TorchHrmConfig,
    create_hrm_model,
)

@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestHrmFactory:
    """Test the create_hrm_model factory function."""

    def test_create_hrm_model_types(self):
        """Test that the factory returns correct types."""
        model, config = create_hrm_model()
        assert isinstance(model, TorchHrmModule)
        assert isinstance(config, TorchHrmConfig)

    def test_create_hrm_model_config_kwargs(self):
        """Test that config_kwargs are correctly applied."""
        custom_dim = 128
        model, config = create_hrm_model(embedding_dim=custom_dim)
        assert config.embedding_dim == custom_dim
        assert model.config.embedding_dim == custom_dim

    def test_create_hrm_model_eval_mode(self):
        """Test that the model is created in eval mode."""
        model, _ = create_hrm_model()
        assert not model.training

    def test_create_hrm_model_device(self):
        """Test that the model is on the correct device."""
        # Using CPU as it's always available for these tests
        model, _ = create_hrm_model(device="cpu")
        # Check one of the parameters to verify device
        param = next(model.parameters())
        assert param.device.type == "cpu"

    def test_create_hrm_model_determinism(self):
        """Test that the same seed produces identical weights."""
        seed = 123
        model1, _ = create_hrm_model(seed=seed)
        model2, _ = create_hrm_model(seed=seed)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_create_hrm_model_different_seeds(self):
        """Test that different seeds produce different weights."""
        model1, _ = create_hrm_model(seed=42)
        model2, _ = create_hrm_model(seed=43)

        # At least some parameters should be different
        different = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2):
                different = True
                break
        assert different, "Models with different seeds should have different weights"
