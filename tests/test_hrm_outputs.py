"""
Focused Tests for Torch HRM Output Shape and Loss Sanity

Tests for verifying output shapes and loss sanity in the HRM module.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from typing import Tuple

from hrm.torch_hrm import (
    TorchHrmModule,
    TorchHrmConfig,
    create_hrm_model,
    get_device,
)


class TestHrmOutputShape:
    """Test output shapes of the HRM model."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return get_device()
    
    @pytest.fixture
    def model(self, device):
        """Create a test model."""
        model, _ = create_hrm_model(
            device=device,
            seed=42,
            num_entities=100,
            num_relations=10,
            embedding_dim=32,
            num_hash_buckets=64,
            hidden_dim=64,
        )
        return model
    
    @pytest.fixture
    def sample_input(self, device):
        """Create sample input tensors."""
        batch_size = 4
        seq_len = 8
        
        head_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        relation_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
        tail_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        
        return head_ids, relation_ids, tail_ids
    
    def test_forward_output_shape(self, model, sample_input):
        """Test that forward pass returns correct output shape."""
        head_ids, relation_ids, tail_ids = sample_input
        
        output = model(head_ids, relation_ids, tail_ids)
        
        expected_shape = (head_ids.shape[0], head_ids.shape[1], 1)
        
        assert output.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {output.shape}"
        )
    
    def test_embedding_output_shape(self, model, sample_input):
        """Test that embedding layer returns correct shapes."""
        head_ids, relation_ids, _ = sample_input
        
        head_emb, relation_emb = model.get_embeddings(head_ids, relation_ids)
        
        assert head_emb.shape == (head_ids.shape[0], head_ids.shape[1], 32)
        assert relation_emb.shape == (relation_ids.shape[0], relation_ids.shape[1], 32)
    
    def test_different_batch_sizes(self, model, device):
        """Test that model handles different batch sizes correctly."""
        seq_len = 4
        embedding_dim = 32
        
        for batch_size in [1, 2, 8, 16]:
            head_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
            relation_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
            tail_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
            
            output = model(head_ids, relation_ids, tail_ids)
            
            assert output.shape == (batch_size, seq_len, 1), (
                f"Failed for batch_size={batch_size}"
            )
    
    def test_different_seq_lengths(self, model, device):
        """Test that model handles different sequence lengths correctly."""
        batch_size = 4
        embedding_dim = 32
        
        for seq_len in [1, 4, 16, 32]:
            head_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
            relation_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
            tail_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
            
            output = model(head_ids, relation_ids, tail_ids)
            
            assert output.shape == (batch_size, seq_len, 1), (
                f"Failed for seq_len={seq_len}"
            )


class TestHrmLossSanity:
    """Test loss sanity of the HRM model."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return get_device()
    
    @pytest.fixture
    def model(self, device):
        """Create a test model."""
        model, _ = create_hrm_model(
            device=device,
            seed=42,
            num_entities=100,
            num_relations=10,
            embedding_dim=32,
            num_hash_buckets=64,
            hidden_dim=64,
        )
        return model
    
    @pytest.fixture
    def sample_input(self, device):
        """Create sample input tensors."""
        batch_size = 4
        seq_len = 8
        
        head_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        relation_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
        tail_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        
        return head_ids, relation_ids, tail_ids
    
    def test_loss_is_finite(self, model, sample_input):
        """Test that loss is finite (not NaN or Inf)."""
        head_ids, relation_ids, tail_ids = sample_input
        
        model.eval()
        with torch.no_grad():
            output = model(head_ids, relation_ids, tail_ids)
            
            # Simple MSE loss
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)
            
            assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    
    def test_loss_in_reasonable_range(self, model, sample_input):
        """Test that loss is in a reasonable range."""
        head_ids, relation_ids, tail_ids = sample_input
        
        model.eval()
        with torch.no_grad():
            output = model(head_ids, relation_ids, tail_ids)
            
            # Target with small values
            target = torch.zeros_like(output)
            loss = torch.nn.functional.mse_loss(output, target)
            
            # Loss should be finite
            assert torch.isfinite(loss), f"Loss is not finite: {loss}"
            # Loss should be non-negative
            assert loss >= 0, f"Loss is negative: {loss}"
    
    def test_gradient_flow(self, model, sample_input):
        """Test that gradients flow properly."""
        head_ids, relation_ids, tail_ids = sample_input
        
        model.train()
        
        # Forward pass
        output = model(head_ids, relation_ids, tail_ids)
        
        # Dummy loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    def test_loss_decreases_with_training(self, model, device):
        """Test that loss decreases with training steps."""
        head_ids = torch.randint(0, 100, (4, 8), device=device)
        relation_ids = torch.randint(0, 10, (4, 8), device=device)
        tail_ids = torch.randint(0, 100, (4, 8), device=device)
        
        # Fixed target
        target = torch.zeros(4, 8, 1, device=device)
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        losses = []
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model(head_ids, relation_ids, tail_ids)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should generally decrease
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
        )
    
    def test_deterministic_output(self, device):
        """Test that model produces deterministic output with same seed."""
        # Create two models with same seed
        model1, _ = create_hrm_model(device=device, seed=42)
        model2, _ = create_hrm_model(device=device, seed=42)
        
        # Same input
        head_ids = torch.randint(0, 100, (2, 4), device=device)
        relation_ids = torch.randint(0, 10, (2, 4), device=device)
        tail_ids = torch.randint(0, 100, (2, 4), device=device)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(head_ids, relation_ids, tail_ids)
            output2 = model2(head_ids, relation_ids, tail_ids)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2), (
            "Model outputs differ with same seed"
        )


class TestHrmDevice:
    """Test device selection and compatibility."""
    
    def test_cpu_device(self):
        """Test model works on CPU."""
        model, _ = create_hrm_model(device="cpu", seed=42)
        
        head_ids = torch.randint(0, 100, (2, 4))
        relation_ids = torch.randint(0, 10, (2, 4))
        tail_ids = torch.randint(0, 100, (2, 4))
        
        output = model(head_ids, relation_ids, tail_ids)
        
        assert output.device.type == "cpu"
        assert output.shape == (2, 4, 1)
    
    def test_mps_device_available(self):
        """Test MPS device availability."""
        if torch.backends.mps.is_available():
            model, _ = create_hrm_model(device="mps", seed=42)
            
            head_ids = torch.randint(0, 100, (2, 4), device="mps")
            relation_ids = torch.randint(0, 10, (2, 4), device="mps")
            tail_ids = torch.randint(0, 100, (2, 4), device="mps")
            
            output = model(head_ids, relation_ids, tail_ids)
            
            assert output.device.type == "mps"
            assert output.shape == (2, 4, 1)
        else:
            pytest.skip("MPS not available")


def run_tests():
    """Run all tests and return results."""
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    run_tests()
