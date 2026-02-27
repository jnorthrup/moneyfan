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
from typing import Any, Optional, Tuple, List

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    optim = None
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
    input_dim: int = 92
    hidden_dim: int = 64
    ob_depth_frames: int = 20
    ob_lookback_horizon: int = 200
    ob_decay_mode: str = "exponential"
    ob_hyperbolic_tau: float = 32.0
    ob_tactical_near_frames: int = 64
    regime_attn_layers: int = 2
    tactical_attn_layers: int = 2
    regime_update_cycles: int = 2
    tactical_update_cycles: int = 3
    n_heads: int = 4
    use_mechanical_veto: bool = False
    replay_coalescing: bool = False
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_momentum: float = 0.95
    optimizer_nesterov: bool = True
    muon_ns_steps: int = 5
    energy_discount_gamma: float = 0.99
    energy_roundtrip_cost_bps: float = 16.0
    energy_churn_penalty: float = 0.0
    energy_target_clip: float = 0.25
    
    # Objective weights (for differentiable penalties)
    world_model_weight: float = 1.0
    trade_head_weight: float = 1.0
    cost_turnover_weight: float = 0.0

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

    def __init__(
        self,
        hidden_dim: int,
        ob_depth_frames: int = 20,
        ob_lookback_horizon: int = 200,
        decay_mode: str = "exponential",
        hyperbolic_tau: float = 32.0,
    ):
        self.hidden_dim = hidden_dim
        self.ob_depth_frames = ob_depth_frames
        self.ob_lookback_horizon = ob_lookback_horizon
        self.decay_ratio = ob_lookback_horizon ** (1.0 / max(ob_depth_frames - 1, 1))
        self.decay_mode = decay_mode
        self.hyperbolic_tau = hyperbolic_tau

    def _alpha_for_frame(self, k: int) -> float:
        if self.decay_mode == "hyperbolic":
            tau = max(self.hyperbolic_tau, 1.0)
            # Slower falloff than exponential: preserves long-tail memory while
            # keeping the first band responsive for execution features.
            alpha = 1.0 / (1.0 + (k / tau))
            return float(min(1.0, max(0.01, alpha)))
        return float(1.0 / (self.decay_ratio ** k))

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
            alpha_k = self._alpha_for_frame(k)
            prev_frame = depth_frames[-1]
            frame_k = (1.0 - alpha_k) * temporal_ob[:, k:k+1, :] + alpha_k * prev_frame
            depth_frames.append(frame_k)

        return mx.concatenate(depth_frames, axis=1)

    def _frame_weights(self) -> mx.array:
        if self.decay_mode == "hyperbolic":
            weights_list = [
                1.0 / (1.0 + (k / max(self.hyperbolic_tau, 1e-6)))
                for k in range(self.ob_depth_frames)
            ]
        else:
            weights_list = [1.0 / (self.decay_ratio ** k) for k in range(self.ob_depth_frames)]

        weights_sum = sum(weights_list)
        return mx.array([w / weights_sum for w in weights_list])

    def _read_with_weights(self, temporal_ob: mx.array, weights: mx.array) -> mx.array:
        weighted = temporal_ob * weights.reshape(1, -1, 1)
        return mx.sum(weighted, axis=1)  # [B, D]

    def read_tactical(self, temporal_ob: mx.array, near_frames: int = 64) -> mx.array:
        """
        Tactical read seam for near-horizon context.

        Trunk behavior remains conservative and backwards compatible:
        it uses a normalized near-band weighting over the existing frame tensor.
        """
        active = min(max(1, near_frames), self.ob_depth_frames)
        if self.decay_mode == "hyperbolic":
            # Sharper recency bias than regime read to protect execution crispness.
            tactical_tau = max(4.0, self.hyperbolic_tau * 0.25)
            near_weights = [1.0 / (1.0 + (k / tactical_tau)) for k in range(active)]
        else:
            near_weights = [1.0 / (k + 1.0) for k in range(active)]
        near_weights += [0.0] * (self.ob_depth_frames - active)
        weights_sum = sum(near_weights) or 1.0
        weights = mx.array([w / weights_sum for w in near_weights])
        return self._read_with_weights(temporal_ob, weights)

    def read_regime(self, temporal_ob: mx.array) -> mx.array:
        """
        Read the regime context vector from the temporal order book.

        Each depth frame contributes inversely to its age (decay_ratio^k),
        producing a recency-weighted market context — the "best bid" of the
        order book in embedding space.

        Returns:
            market_context : [B, D]
        """
        if self.decay_mode != "hyperbolic":
            return self._read_with_weights(temporal_ob, self._frame_weights())

        # Banded weighting so far memory survives the near-band dominance.
        f = self.ob_depth_frames
        near_end = min(64, f)
        mid_end = min(192, f)

        weights = [0.0] * f
        if near_end > 0:
            near = [1.0 / (1.0 + (k / max(self.hyperbolic_tau * 0.5, 1.0))) for k in range(near_end)]
            s = sum(near) or 1.0
            for i, w in enumerate(near):
                weights[i] = 0.45 * (w / s)

        if mid_end > near_end:
            mid_len = mid_end - near_end
            mid = [1.0 / (1.0 + (k / max(self.hyperbolic_tau, 1.0))) for k in range(mid_len)]
            s = sum(mid) or 1.0
            for i, w in enumerate(mid, start=near_end):
                weights[i] = 0.35 * (w / s)

        if f > mid_end:
            far_len = f - mid_end
            far = [1.0 / (1.0 + (k / max(self.hyperbolic_tau * 2.0, 1.0))) for k in range(far_len)]
            s = sum(far) or 1.0
            for i, w in enumerate(far, start=mid_end):
                weights[i] = 0.20 * (w / s)

        wsum = sum(weights) or 1.0
        return self._read_with_weights(temporal_ob, mx.array([w / wsum for w in weights]))

    def read(self, temporal_ob: mx.array) -> mx.array:
        """Backward-compatible alias for regime context read."""
        return self.read_regime(temporal_ob)


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
        self.bar_feature_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Temporal order book memory (replaces "sparkline")
        self.temporal_ob = MLXTemporalOrderBook(
            config.hidden_dim,
            config.ob_depth_frames,
            config.ob_lookback_horizon,
            decay_mode=config.ob_decay_mode,
            hyperbolic_tau=config.ob_hyperbolic_tau,
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

        # World-model pre-training head: predict next bar's full feature vector (92 channels)
        self.codec_score_head = nn.Linear(config.hidden_dim, config.input_dim)

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
        B, T, F = bar_codec_features.shape

        # ── Input shape guard ─────────────────────────────────────────────────
        # Validate before the first matmul so callers get a descriptive error
        # (not the cryptic MLX [addmm] message) when weights were built with a
        # different input_dim than the incoming feature tensor.
        expected_dim = self.config.input_dim
        if F != expected_dim:
            raise ValueError(
                f"[HRM] Input feature dimension mismatch: got {F} features but "
                f"bar_feature_proj was built for {expected_dim}. "
                "Re-initialize the model or load compatible weights."
            )

        temporal_ob_state, regime_state, tactical_state = (
            memory if memory else (None, None, None)
        )

        # Project bar features into HRM embedding space
        bar_embed = self.bar_feature_proj(bar_codec_features)

        # Update temporal order book with current bar's mean embedding
        current_bar_embed = bar_embed.mean(axis=1)  # [B, D]
        temporal_ob_state = self.temporal_ob.update(temporal_ob_state, current_bar_embed)
        regime_context = self.temporal_ob.read_regime(temporal_ob_state)  # [B, D]
        tactical_context = self.temporal_ob.read_tactical(
            temporal_ob_state,
            near_frames=self.config.ob_tactical_near_frames,
        )  # [B, D]

        # Broadcast split contexts across all timesteps in the bar window
        regime_context_broadcast = mx.broadcast_to(
            mx.expand_dims(regime_context, 1),
            (B, T, self.hidden_dim),
        )
        tactical_context_broadcast = mx.broadcast_to(
            mx.expand_dims(tactical_context, 1),
            (B, T, self.hidden_dim),
        )
        bar_features_with_regime_context = bar_embed + regime_context_broadcast
        bar_features_with_tactical_context = bar_embed + tactical_context_broadcast

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
                    tactical_state, regime_state + bar_features_with_tactical_context
                )
            regime_state = self.macro_regime_layer(
                regime_state, tactical_state + bar_features_with_regime_context
            )

        # Final cycle (with gradient)
        for _tactical_cycle in range(self.config.tactical_update_cycles):
            tactical_state = self.tactical_execution_layer(
                tactical_state, regime_state + bar_features_with_tactical_context
            )
        regime_state = self.macro_regime_layer(
            regime_state, tactical_state + bar_features_with_regime_context
        )

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
      energy_step   : energy-routing proxy loss — regress discounted net alpha score from trade outputs

    Optimizations:
      MLX lazy evaluation + fused graph execution.
      Optimizer updates can be coalesced via auto_eval=False + flush_updates().
      BPTT horizon is bounded to the sequence window via mx.stop_gradient in the model's forward.
    """

    def __init__(self, config: HRMConfig = None):
        self.config = config or HRMConfig()
        self.model = MLXHierarchicalCodec(self.config)
        self.optimizer_name = (self.config.optimizer_name or "adamw").strip().lower()
        self.optimizer = self._build_optimizer()
        
        # Compile pure inner functions to avoid binding 'self' continually
        
        def _split_trade_outputs(output: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
            return (
                output[:, 0],
                output[:, 1],
                output[:, 2],
                output[:, 3],
                output[:, 4],
            )

        def pretrain_loss_fn(model: "MLXHierarchicalCodec", bar_codec_features: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
            output, next_memory = model.forward(bar_codec_features, memory=memory, mode="pretrain")
            target = bar_codec_features[:, -1, :]
            world_model_loss = mx.mean(mx.square(output - target))
            return world_model_loss, next_memory

        def trade_loss_fn(model: "MLXHierarchicalCodec", bar_codec_features: mx.array, realized_returns: mx.array, memory: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
            output, next_memory = model.forward(bar_codec_features, memory=memory, mode="trade")
            pred_fwd_return, signal_conviction, stop_loss_pct, take_profit_pct, position_fraction = _split_trade_outputs(output)

            # PyTorch parity: compute realized PnL with SL/TP clamping and position sizing.
            # Conviction remains in the objective as a gate on realized exposure.
            realized = realized_returns.reshape((-1, 1))
            entry = mx.ones_like(realized)
            exit_price = entry * (1.0 + realized)

            pred_dir = mx.sign(pred_fwd_return).reshape((-1, 1))
            active_mask = mx.abs(pred_dir)
            conviction = signal_conviction.reshape((-1, 1))
            sl = mx.maximum(mx.abs(stop_loss_pct).reshape((-1, 1)), 1e-4)
            tp = mx.maximum(take_profit_pct.reshape((-1, 1)), 1e-4)
            size = mx.clip(position_fraction.reshape((-1, 1)), 0.0, 1.0)

            long_exit = mx.clip(exit_price, entry * (1.0 - sl), entry * (1.0 + tp))
            short_exit = mx.clip(exit_price, entry * (1.0 - tp), entry * (1.0 + sl))
            exit_final = mx.where(pred_dir > 0, long_exit, short_exit)

            raw_pnl = (exit_final - entry) * size * conviction * 100.0
            final_pnl = mx.where(pred_dir > 0, raw_pnl, -raw_pnl)
            
            # Differentiable cost penalty: penalize predicted exposure to encourage sparse, high-conviction signals.
            # (Roundtrip cost in bps * effective size)
            roundtrip_cost = float(getattr(model.config, "energy_roundtrip_cost_bps", 16.0)) / 10000.0
            cost_penalty = roundtrip_cost * size * conviction * 100.0 # scale to PnL units (100x return)
            
            # Combine terms using objective weights
            # Note: final_pnl is (+) for profit, (-) for loss. We want to maximize (alpha - cost).
            # So loss = -(alpha - cost) = -alpha + cost
            weighted_alpha = final_pnl * float(getattr(model.config, "trade_head_weight", 1.0))
            weighted_cost = cost_penalty * float(getattr(model.config, "cost_turnover_weight", 0.0))
            
            net_pnl = weighted_alpha - weighted_cost
            alpha_loss = -mx.mean(net_pnl * active_mask)
            return alpha_loss, next_memory

        energy_discount_gamma = float(max(0.0, min(1.0, getattr(self.config, "energy_discount_gamma", 0.99))))
        energy_roundtrip_cost = float(max(0.0, getattr(self.config, "energy_roundtrip_cost_bps", 16.0))) / 10000.0
        energy_churn_penalty = float(max(0.0, getattr(self.config, "energy_churn_penalty", 0.0)))
        energy_target_clip = float(max(1e-6, getattr(self.config, "energy_target_clip", 0.25)))

        def energy_loss_fn(
            model: "MLXHierarchicalCodec",
            bar_codec_features: mx.array,
            realized_returns: mx.array,
            memory: Optional[Tuple] = None,
        ) -> Tuple[mx.array, Tuple]:
            """
            Energy-routing proxy objective (training-only):
            learn a scalar discounted net-alpha score from the existing trade heads.

            No new runtime head is introduced yet to preserve checkpoint compatibility.
            The scalar score is synthesized from the trade outputs and trained against a
            discounted/costed realized-return proxy target.
            """
            output, next_memory = model.forward(bar_codec_features, memory=memory, mode="trade")
            pred_fwd_return, signal_conviction, _stop_loss_pct, _take_profit_pct, position_fraction = _split_trade_outputs(output)

            realized = realized_returns.reshape((-1, 1))
            conviction = signal_conviction.reshape((-1, 1))
            size = mx.clip(position_fraction.reshape((-1, 1)), 0.0, 1.0)

            # Scalar routing score proxy: predicted discounted net alpha from trade outputs.
            pred_alpha = mx.tanh(pred_fwd_return).reshape((-1, 1)) * conviction * size
            pred_net_alpha = (pred_alpha * energy_discount_gamma) - ((energy_roundtrip_cost + energy_churn_penalty) * size)

            # Target is a clipped discounted realized alpha proxy, costed on the same size proxy.
            target_realized = mx.clip(realized, -energy_target_clip, energy_target_clip)
            target_net_alpha = (target_realized * energy_discount_gamma) - ((energy_roundtrip_cost + energy_churn_penalty) * size)

            # Energy = -net_alpha. Minimize energy regression error.
            pred_energy = -pred_net_alpha
            target_energy = -target_net_alpha
            energy_loss = mx.mean(mx.square(pred_energy - target_energy))
            return energy_loss, next_memory

        self._pretrain_loss_and_grad = mx.value_and_grad(pretrain_loss_fn)
        self._trade_loss_and_grad = mx.value_and_grad(trade_loss_fn)
        self._energy_loss_and_grad = mx.value_and_grad(energy_loss_fn)

    def _build_optimizer(self) -> Any:
        if not HAS_MLX or optim is None:
            return None

        lr = float(self.config.learning_rate)
        wd = float(self.config.weight_decay)
        b1 = float(self.config.optimizer_beta1)
        b2 = float(self.config.optimizer_beta2)

        name = self.optimizer_name
        if name == "adam":
            return optim.Adam(learning_rate=lr, betas=[b1, b2])
        if name == "adamw":
            return optim.AdamW(learning_rate=lr, betas=[b1, b2], weight_decay=wd)
        if name == "lion":
            # Lion typically prefers lower lr and higher wd than AdamW; leave explicit tuning to config.
            return optim.Lion(
                learning_rate=lr,
                betas=[b1, min(b2, 0.999)],
                weight_decay=wd,
            )
        if name == "muon":
            # Muon is strongest on matrix-like hidden weights. Route 0D/1D params
            # (biases, scalar heads, norms) to AdamW as a stable fallback.
            muon_opt = optim.Muon(
                learning_rate=lr,
                momentum=float(self.config.optimizer_momentum),
                weight_decay=wd,
                nesterov=bool(self.config.optimizer_nesterov),
                ns_steps=int(self.config.muon_ns_steps),
            )
            fallback = optim.AdamW(learning_rate=lr, betas=[b1, b2], weight_decay=wd)
            return optim.MultiOptimizer(
                [muon_opt, fallback],
                filters=[lambda _path, weight: getattr(weight, "ndim", 0) >= 2],
            )

        raise ValueError(
            f"Unsupported optimizer_name={self.config.optimizer_name!r}. "
            "Expected one of: adam, adamw, lion, muon"
        )

    def _eval_training_state(self, *values: mx.array, memory: Optional[Tuple] = None):
        eval_args: List[Any] = [*values]
        if memory is not None:
            eval_args.extend(list(memory))
        if self.optimizer is not None:
            eval_args.append(self.model.parameters())
            eval_args.append(self.optimizer.state)
        mx.eval(*eval_args)

    def clip_gradients(self, grads: Dict[str, mx.array], max_norm: float = 1.0) -> Dict[str, mx.array]:
        """
        Clip gradients to a maximum norm to prevent gradient explosion during BPTT.

        Args:
            grads: Dictionary of gradients from mx.value_and_grad
            max_norm: Maximum L2 norm for the gradients

        Returns:
            Clipped gradients
        """
        total_norm_sq = sum(mx.square(g).sum() for g in grads.values() if g is not None)
        total_norm = mx.sqrt(total_norm_sq) + 1e-8
        scale = mx.minimum(max_norm / total_norm, mx.array(1.0))
        return {k: v * scale for k, v in grads.items()}

    def pretrain_step(
        self,
        bar_codec_features: mx.array,
        memory: Optional[Tuple] = None,
        auto_eval: bool = True,
        clip_gradients: bool = False,
        max_gradient_norm: float = 1.0,
        scale: float = 1.0,
    ) -> Tuple[mx.array, Tuple]:
        """
        World-model pre-training step with optimizer update.

        Loss: MSE between predicted next-bar codec features and actual last bar.
        Returns: loss scalar, next memory state
        """
        (world_model_loss, next_memory), grads = self._pretrain_loss_and_grad(
            self.model, bar_codec_features, memory
        )
        
        if scale != 1.0:
            world_model_loss = world_model_loss * scale
            grads = {k: v * scale if v is not None else None for k, v in grads.items()}

        if self.optimizer is not None:
            if clip_gradients:
                grads = self.clip_gradients(grads, max_gradient_norm)
            self.optimizer.update(self.model, grads)
        if auto_eval:
            self._eval_training_state(world_model_loss, memory=next_memory)
        return world_model_loss, next_memory

    def trade_step(
        self,
        bar_codec_features: mx.array,
        realized_returns: mx.array,
        memory: Optional[Tuple] = None,
        auto_eval: bool = True,
        clip_gradients: bool = False,
        max_gradient_norm: float = 1.0,
        scale: float = 1.0,
    ) -> Tuple[mx.array, Tuple]:
        """
        Alpha-maximisation training step with optimizer update.

        Loss: negative conviction-weighted expected return (maximise alpha).
        Returns: loss scalar, next memory state
        """
        (alpha_loss, next_memory), grads = self._trade_loss_and_grad(
            self.model, bar_codec_features, realized_returns, memory
        )
        
        if scale != 1.0:
            alpha_loss = alpha_loss * scale
            grads = {k: v * scale if v is not None else None for k, v in grads.items()}

        if self.optimizer is not None:
            if clip_gradients:
                grads = self.clip_gradients(grads, max_gradient_norm)
            self.optimizer.update(self.model, grads)
        if auto_eval:
            self._eval_training_state(alpha_loss, memory=next_memory)
        return alpha_loss, next_memory

    def energy_step(
        self,
        bar_codec_features: mx.array,
        realized_returns: mx.array,
        memory: Optional[Tuple] = None,
        auto_eval: bool = True,
        clip_gradients: bool = False,
        max_gradient_norm: float = 1.0,
        scale: float = 1.0,
    ) -> Tuple[mx.array, Tuple]:
        """
        Energy-routing proxy training step with optimizer update.

        Loss: MSE between predicted and realized discounted net-alpha energy proxies.
        Returns: loss scalar, next memory state
        """
        (energy_loss, next_memory), grads = self._energy_loss_and_grad(
            self.model, bar_codec_features, realized_returns, memory
        )
        
        if scale != 1.0:
            energy_loss = energy_loss * scale
            grads = {k: v * scale if v is not None else None for k, v in grads.items()}

        if self.optimizer is not None:
            if clip_gradients:
                grads = self.clip_gradients(grads, max_gradient_norm)
            self.optimizer.update(self.model, grads)
        if auto_eval:
            self._eval_training_state(energy_loss, memory=next_memory)
        return energy_loss, next_memory

    def flush_updates(self, *values: mx.array, memory: Optional[Tuple] = None):
        """Force materialization of any queued optimizer/model updates."""
        self._eval_training_state(*values, memory=memory)


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
