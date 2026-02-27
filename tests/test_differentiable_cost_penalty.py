import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import HRMConfig, MLXBasketTrainer, MLXHierarchicalCodec
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_trade_loss_with_cost_penalty():
    # Setup config with zero cost weight first
    config = HRMConfig(
        input_dim=10,
        hidden_dim=8,
        trade_head_weight=1.0,
        cost_turnover_weight=0.0,
        energy_roundtrip_cost_bps=100.0 # 1% cost
    )
    
    trainer = MLXBasketTrainer(config)
    
    # Create dummy data
    B, T, F = 1, 5, 10
    bar_features = mx.random.normal((B, T, F))
    realized_returns = mx.array([[0.02]]) # 2% return (raw)
    
    # Get loss with zero cost
    (loss_no_cost, _), _ = trainer._trade_loss_and_grad(
        trainer.model, bar_features, realized_returns
    )
    
    # Now enable cost penalty
    config.cost_turnover_weight = 1.0
    # Re-initialize trainer internals if needed (or just use the loss_fn directly if it captures config by reference)
    # The loss_fn captures model.config, so it should see the update if model.config is the same object.
    
    (loss_with_cost, _), _ = trainer._trade_loss_and_grad(
        trainer.model, bar_features, realized_returns
    )
    
    print(f"Loss No Cost: {loss_no_cost.item()}")
    print(f"Loss With Cost: {loss_with_cost.item()}")
    
    # Loss should be HIGHER (less negative) with cost penalty because we subtract cost from alpha.
    # loss = -(alpha - cost) = -alpha + cost.
    assert loss_with_cost.item() > loss_no_cost.item()

@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_trade_loss_alpha_scaling():
    config = HRMConfig(
        input_dim=10,
        hidden_dim=8,
        trade_head_weight=2.0,
        cost_turnover_weight=0.0
    )
    trainer = MLXBasketTrainer(config)
    
    B, T, F = 1, 5, 10
    bar_features = mx.random.normal((B, T, F))
    realized_returns = mx.array([[0.02]])
    
    (loss_h2, _), _ = trainer._trade_loss_and_grad(
        trainer.model, bar_features, realized_returns
    )
    
    config.trade_head_weight = 1.0
    (loss_h1, _), _ = trainer._trade_loss_and_grad(
        trainer.model, bar_features, realized_returns
    )
    
    # loss_h2 should be twice loss_h1 (roughly, minus the mean over batch)
    np.testing.assert_allclose(loss_h2.item(), 2.0 * loss_h1.item(), rtol=1e-5)
