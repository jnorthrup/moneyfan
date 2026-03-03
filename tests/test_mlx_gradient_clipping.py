import numpy as np
import pytest

from hrm.hierarchical_codec_mlx import HAS_MLX, HRMConfig, MLXBasketTrainer


@pytest.mark.skipif(not HAS_MLX, reason="MLX is required for this regression test")
def test_pretrain_step_gradient_clipping_accepts_nested_gradient_tree():
    import mlx.core as mx

    trainer = MLXBasketTrainer(
        HRMConfig(
            n_codec_outputs=4,
            input_dim=20,
            hidden_dim=64,
            regime_attn_layers=2,
            tactical_attn_layers=2,
            n_heads=4,
        )
    )

    batch = mx.array(np.random.randn(1, 16, 20).astype(np.float32))
    loss, memory = trainer.pretrain_step(
        batch,
        memory=None,
        clip_gradients=True,
        max_gradient_norm=1.0,
    )

    # Materialize lazy tensors so this test fails if nested gradient handling regresses.
    mx.eval(loss, *memory)
    assert np.isfinite(float(loss.item()))
    assert memory is not None
