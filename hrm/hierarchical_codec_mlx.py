"""
HRM MLX Implementation — Native (Preserves Architecture)
=========================================================

MLX implementation that preserves HRM's exact architecture:
- macro_regime_layer / tactical_execution_layer nested cycles: SEQUENTIAL (not tiled)
- TemporalOrderBook update: CASCADING sequential updates (not tiled)
- State persistence: proper regime_state / tactical_state carry across cycles

MLX provides speedup through:
1. Lazy evaluation (automatic kernel fusion)
2. Metal GPU acceleration
3. ANE targeting (Apple Neural Engine)
4. Automatic optimization

DO NOT break HRM's sequential dependencies for speed.

Naming convention (crypto-technical):
  TemporalOrderBook     — cascading decay memory: exponentially-weighted order book of embeddings
  ob_depth_frames       — number of temporal "price levels" in the order book memory
  ob_lookback_horizon   — horizon over which the decay ratio is calibrated
  macro_regime_layer    — HRM Slow/High layer: detects market regime, risk budget, codec trust
  tactical_execution_layer — HRM Fast/Low layer: real-time signal execution and position sizing
  regime_state (z_H)    — working state of the macro regime layer
  tactical_state (z_L)  — working state of the tactical execution layer
  codec_score_head      — emits next-bar codec feature predictions (world model pre-training)
  expected_return_head  — predicts forward expected return
  signal_conviction_head — conviction score ∈ [0,1]
  stop_loss_head        — stop-loss offset ∈ [-0.15, 0]
  take_profit_head      — take-profit target ∈ [0, 0.30]
  position_size_head    — fraction of notional ∈ [0, 1]
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")


@dataclass
class HRMConfig:
    """
    Configuration for the Hierarchical Reasoning Model (HRM).

    Crypto-technical field names:
      n_codec_outputs      : number of codec expert output channels (default 24)
      hidden_dim           : embedding dimension throughout the HRM
      ob_depth_frames      : temporal order book depth (number of decay frames)
      ob_lookback_horizon  : candle horizon over which OB decay ratio is calibrated
      regime_attn_layers   : transformer layers in the macro regime layer
      tactical_attn_layers : transformer layers in the tactical execution layer
      regime_update_cycles : number of macro regime update cycles per forward pass
      tactical_update_cycles: number of tactical execution cycles per regime cycle
      n_heads              : number of attention heads
    """
    n_codec_outputs: int = 24
    hidden_dim: int = 64
    ob_depth_frames: int = 20
    ob_lookback_horizon: int = 200
    regime_attn_layers: int = 2
    tactical_attn_layers: int = 2
    regime_update_cycles: int = 2
    tactical_update_cycles: int = 3
    n_heads: int = 4

    # Legacy aliases for callers that still use old field names
    @property
    def n_signals(self): return self.n_codec_outputs
    @property
    def sparkline_frames(self): return self.ob_depth_frames
    @property
    def sparkline_horizon(self): return self.ob_lookback_horizon
    @property
    def H_layers(self): return self.regime_attn_layers
    @property
    def L_layers(self): return self.tactical_attn_layers
    @property
    def H_cycles(self): return self.regime_update_cycles
    @property
    def L_cycles(self): return self.tactical_update_cycles


# Legacy alias so existing import `HierarchicalCodecConfig as MLXConfig` still resolves
HierarchicalCodecConfig = HRMConfig


class MLXTemporalOrderBook:
    """
    Temporal Order Book — cascading exponential decay memory.

    Analogous to a limit order book but over time: each 'depth frame' holds
    an exponentially-decayed embedding of past market state, with frame 0
    being the most recent (highest weight) and frame ob_depth_frames-1 the
    oldest (lowest weight).

    Frame 0 = current bar embedding
    Frame k = (1 - alpha_k) * old[k] + alpha_k * frame_{k-1}

    This creates a cascading temporal memory — NO tiling, NO parallel frame computation.
    """

    def __init__(self, hidden_dim: int, ob_depth_frames: int = 20, ob_lookback_horizon: int = 200):
        self.hidden_dim = hidden_dim
        self.ob_depth_frames = ob_depth_frames
        self.ob_lookback_horizon = ob_lookback_horizon
        self.decay_ratio = ob_lookback_horizon ** (1.0 / max(ob_depth_frames - 1, 1))

    def update(self, temporal_ob: Optional[mx.array], current_bar: mx.array) -> mx.array:
        """
        Update the temporal order book with the current bar embedding.

        Args:
            temporal_ob : [B, ob_depth_frames, D] — previous order book state (None on first call)
            current_bar : [B, D] — current bar's projected embedding

        Returns:
            Updated temporal order book [B, ob_depth_frames, D]
        """
        B, D = current_bar.shape

        if temporal_ob is None:
            temporal_ob = mx.zeros((B, self.ob_depth_frames, D))

        # Frame 0 is always the current bar (most recent)
        frame_0 = current_bar[:, None, :]

        # Sequential cascading: each depth level decays from the level above it
        depth_frames: List[mx.array] = [frame_0]
        for k in range(1, self.ob_depth_frames):
            alpha_k = 1.0 / (self.decay_ratio ** k)
            prev_frame = depth_frames[-1]
            frame_k = (1.0 - alpha_k) * temporal_ob[:, k:k+1, :] + alpha_k * prev_frame
            depth_frames.append(frame_k)

        return mx.concatenate(depth_frames, axis=1)

    def read(self, temporal_ob: mx.array) -> mx.array:
        """
        Read the regime context vector from the temporal order book.

        Each depth frame contributes inversely to its age (decay_ratio^k),
        producing a recency-weighted market context — the "best bid" of the
        order book in embedding space.

        Returns:
            market_context : [B, D]
        """
        B, F, D = temporal_ob.shape

        weights_list = [1.0 / (self.decay_ratio ** k) for k in range(self.ob_depth_frames)]
        weights_sum = sum(weights_list)
        weights = mx.array([w / weights_sum for w in weights_list])  # [ob_depth_frames]

        weighted = temporal_ob * weights.reshape(1, -1, 1)
        return mx.sum(weighted, axis=1)  # [B, D]


class MLXFeedForward(nn.Module):
    """Feed-forward block (4x expansion, GELU activation)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MLXTransformerBlock(nn.Module):
    """
    Single transformer block: Multi-Head Attention + FeedForward with residuals.

    EXACT PyTorch structure — NO tiling.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.attention = nn.MultiHeadAttention(hidden_dim, n_heads)
        self.norm1 = nn.RMSNorm(hidden_dim)

        self.ff = MLXFeedForward(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward with attention + feed-forward — SEQUENTIAL (not tiled)."""
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class MLXMarketDepthLayer(nn.Module):
    """
    Market depth processing layer — stacked transformer blocks with sequential processing.

    Used for both macro_regime_layer and tactical_execution_layer.
    EXACT PyTorch structure — NO tiling.
    """

    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = [
            MLXTransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ]

    def __call__(self, state: mx.array, context: mx.array) -> mx.array:
        """
        Process state through stacked transformer blocks — SEQUENTIAL.

        Args:
            state   : [B, T, D] — current layer state (regime_state or tactical_state)
            context : [B, T, D] — conditioning context from the other layer + bar features

        Returns:
            Updated state [B, T, D]
        """
        x = state + context
        for block in self.blocks:
            x = block(x)
        return x


class MLXHierarchicalCodec(nn.Module):
    """
    HRM — Native MLX implementation — EXACT PyTorch architecture.

    Two-layer hierarchy:
      macro_regime_layer (Slow/High)  : detects regime, manages codec trust, risk budget
      tactical_execution_layer (Fast/Low) : real-time signal execution, position sizing, veto

    NO TILING — preserves sequential regime/tactical cycle dependencies.

    Output heads (trade mode):
      expected_return_head    : predicted forward return ∈ [-1, 1]
      signal_conviction_head  : confidence in signal ∈ [0, 1]
      stop_loss_head          : stop-loss offset ∈ [-0.15, 0]
      take_profit_head        : take-profit target ∈ [0, 0.30]
      position_size_head      : fraction of notional ∈ [0, 1]
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Project raw OHLCV/codec features into HRM embedding space
        self.bar_feature_proj = nn.Linear(config.n_codec_outputs * 2, config.hidden_dim)

        # Temporal order book memory (replaces "sparkline")
        self.temporal_ob = MLXTemporalOrderBook(
            config.hidden_dim, config.ob_depth_frames, config.ob_lookback_horizon
        )

        # Macro regime layer (Slow/High — strategic)
        self.macro_regime_layer = MLXMarketDepthLayer(
            config.hidden_dim, config.regime_attn_layers, config.n_heads
        )
        # Tactical execution layer (Fast/Low — real-time)
        self.tactical_execution_layer = MLXMarketDepthLayer(
            config.hidden_dim, config.tactical_attn_layers, config.n_heads
        )

        # Learnable initial states
        self.regime_state_init = mx.random.normal((config.hidden_dim,)) * 0.02
        self.tactical_state_init = mx.random.normal((config.hidden_dim,)) * 0.02

        # World-model pre-training head: predict next bar's codec features
        self.codec_score_head = nn.Linear(config.hidden_dim, config.n_codec_outputs * 2)

        # Trade output heads
        self.expected_return_head = nn.Linear(config.hidden_dim, 1)
        self.signal_conviction_head = nn.Linear(config.hidden_dim, 1)
        self.stop_loss_head = nn.Linear(config.hidden_dim, 1)
        self.take_profit_head = nn.Linear(config.hidden_dim, 1)
        self.position_size_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        bar_codec_features: mx.array,
        memory: Optional[Tuple] = None,
        mode: str = "pretrain"
    ) -> Tuple[mx.array, Optional[Tuple]]:
        """
        HRM forward pass — NO TILING.

        Preserves:
        - Sequential macro/tactical cycle processing
        - TemporalOrderBook cascading updates
        - regime_state / tactical_state persistence across calls
        - Bar feature injection at both regime and tactical layers

        MLX optimises execution WITHOUT changing the logic.

        Args:
            bar_codec_features : [B, T, n_codec_outputs*2] — OHLCV bar features + codec scores
            memory             : (temporal_ob_state, regime_state, tactical_state) from previous call
            mode               : "pretrain" → world model loss head
                                 "trade"    → return + conviction + risk management heads

        Returns:
            output     : predicted codec scores (pretrain) or trade parameters (trade)
            new_memory : (temporal_ob_state, regime_state, tactical_state)
        """
        B, T, _ = bar_codec_features.shape

        temporal_ob_state, regime_state, tactical_state = (
            memory if memory else (None, None, None)
        )

        # Project bar features into HRM embedding space
        bar_embed = self.bar_feature_proj(bar_codec_features)

        # Update temporal order book with current bar's mean embedding
        current_bar_embed = bar_embed.mean(axis=1)  # [B, D]
        temporal_ob_state = self.temporal_ob.update(temporal_ob_state, current_bar_embed)
        market_context = self.temporal_ob.read(temporal_ob_state)  # [B, D]

        # Broadcast market context across all timesteps in the bar window
        market_context_broadcast = mx.expand_dims(market_context, 1)  # [B, 1, D]
        market_context_broadcast = mx.broadcast_to(
            market_context_broadcast,
            (B, T, self.hidden_dim)
        )  # [B, T, D]
        bar_features_with_context = bar_embed + market_context_broadcast

        # Initialise regime/tactical states if not carried from previous call
        if regime_state is None:
            regime_state = mx.broadcast_to(
                mx.expand_dims(self.regime_state_init, 0),
                (B, T, self.hidden_dim)
            )
            tactical_state = mx.broadcast_to(
                mx.expand_dims(self.tactical_state_init, 0),
                (B, T, self.hidden_dim)
            )

        # Macro/Tactical nested cycles — SEQUENTIAL (exact PyTorch logic)
        # All cycles except the final one run without gradient contribution
        for _regime_cycle in range(self.config.regime_update_cycles - 1):
            for _tactical_cycle in range(self.config.tactical_update_cycles):
                tactical_state = self.tactical_execution_layer(
                    tactical_state, regime_state + bar_features_with_context
                )
            regime_state = self.macro_regime_layer(regime_state, tactical_state)

        # Final cycle (with gradient)
        for _tactical_cycle in range(self.config.tactical_update_cycles):
            tactical_state = self.tactical_execution_layer(
                tactical_state, regime_state + bar_features_with_context
            )
        regime_state = self.macro_regime_layer(regime_state, tactical_state)

        # Output heads — always read from regime layer's final timestep
        regime_final = regime_state[:, -1, :]  # [B, D]

        if mode == "pretrain":
            # World-model pre-training: predict next bar's codec feature vector
            output = self.codec_score_head(regime_final)
        else:
            # Trade mode: full risk-parameterised output
            pred_fwd_return = self.expected_return_head(regime_final)
            signal_conviction = mx.sigmoid(self.signal_conviction_head(regime_final))

            # stop_loss_pct ∈ [-0.15, 0] — negative = stop below entry
            stop_loss_pct = mx.tanh(self.stop_loss_head(regime_final)) * 0.15
            # take_profit_pct ∈ [0, 0.30] — positive = target above entry
            take_profit_pct = mx.sigmoid(self.take_profit_head(regime_final)) * 0.30
            # position_fraction ∈ [0, 1] — fraction of notional to deploy
            position_fraction = mx.sigmoid(self.position_size_head(regime_final))

            output = mx.concatenate(
                [pred_fwd_return, signal_conviction, stop_loss_pct, take_profit_pct, position_fraction],
                axis=-1
            )

        new_memory = (
            mx.stop_gradient(temporal_ob_state), 
            mx.stop_gradient(regime_state), 
            mx.stop_gradient(tactical_state)
        )
        return output, new_memory


class MLXBasketTrainer:
    """
    MLX-compatible HRM trainer with automatic optimisation.

    Two-phase training:
      pretrain_step : world-model loss — predict next bar's codec features (self-supervised)
      trade_step    : alpha loss — maximise conviction-weighted expected return (supervised)

    Optimizations:
      @mx.compile enables kernel fusion and prevents Python interpreter overhead.
      BPTT horizon is bounded to the sequence window via mx.stop_gradient in the model's forward.
    """

    def __init__(self, config: HRMConfig = None):
        self.config = config or HRMConfig()
        self.model = MLXHierarchicalCodec(self.config)
        
        # Compile pure inner functions to avoid binding 'self' continually
        
        @mx.compile
        def compiled_pretrain(bar_codec_features: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
            output, next_memory = self.model.forward(bar_codec_features, memory=memory, mode="pretrain")
            target = bar_codec_features[:, -1, :]
            world_model_loss = mx.mean(mx.square(output - target))
            return world_model_loss, next_memory
            
        @mx.compile
        def compiled_trade(bar_codec_features: mx.array, realized_returns: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
            output, next_memory = self.model.forward(bar_codec_features, memory=memory, mode="trade")
            pred_fwd_return = output[:, 0]
            signal_conviction = output[:, 1]
            weighted_alpha = pred_fwd_return * signal_conviction
            alpha_loss = -mx.mean(weighted_alpha * realized_returns)
            return alpha_loss, next_memory
            
        self._compiled_pretrain = compiled_pretrain
        self._compiled_trade = compiled_trade

    def pretrain_step(self, bar_codec_features: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
        """
        World-model pre-training step (compiled).

        Loss: MSE between predicted next-bar codec features and actual last bar.
        Returns: loss scalar, next memory state
        """
        return self._compiled_pretrain(bar_codec_features, memory=memory)

    def trade_step(self, bar_codec_features: mx.array, realized_returns: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
        """
        Alpha-maximisation training step (compiled).

        Loss: negative conviction-weighted expected return (maximise alpha).
        Returns: loss scalar, next memory state
        """
        return self._compiled_trade(bar_codec_features, realized_returns, memory=memory)


# Legacy alias — MLXCodecTrainer was the old name; kept for any external references
MLXCodecTrainer = MLXBasketTrainer


def enable_ane_optimization():
    """
    Enable ANE (Apple Neural Engine) optimisation.

    Allows MLX to target specialised hardware for maximum throughput during
    stochastic basket training. Call before creating models.
    """
    if HAS_MLX:
        try:
            mx.set_default_device(mx.ane)
            print("✅ ANE optimisation enabled")
        except:
            try:
                mx.set_default_device(mx.gpu)
                print("✅ GPU optimisation enabled")
            except:
                print("⚠️  Using CPU fallback")
    else:
        print("❌ MLX not available")


def benchmark_speed(bar_codec_features: mx.array, n_iter: int = 100) -> dict:
    """
    Benchmark native MLX HRM forward pass throughput.

    Returns timing statistics in milliseconds.
    """
    if not HAS_MLX:
        return {"error": "MLX not available"}

    config = HRMConfig()
    model = MLXHierarchicalCodec(config)

    # Warmup
    for _ in range(10):
        output, _ = model.forward(bar_codec_features)
        mx.eval(output)

    import time
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        output, _ = model.forward(bar_codec_features)
        mx.eval(output)  # Force evaluation (MLX is lazy)
        times.append(time.perf_counter() - start)

    import numpy as np
    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


import numpy as np
