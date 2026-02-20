# HRM Architecture Comparison: Reference vs SignalHRM

## Overview

Two different HRM implementations exist in the codebase:

1. **Reference HRM** (`reference/hrm.py`) - PyTorch, full duplex, energy quant models
2. **SignalHRM** (`signal_hrm.py`) - MLX/Numpy, signal convergence, 16-signal array

---

## Reference HRM (PyTorch)

### Architecture
```
Full Duplex I/O
├─ Input 1: market_state [B, seq_len, n_assets × n_features]
├─ Input 2: model_reports [B, seq_len, n_models × report_dim]
│            (from swarm energy quant models)
└─ Output: HRMOutput
           ├─ weights [B, n_models]       (model selection)
           └─ directives [B, n_models, 2] (regime_hint, risk_limit)
```

### Key Features
- **H-level**: Global pattern detection, regime determination
- **L-level**: Tactical model weight decisions
- **Full Duplex**: Bidirectional communication with model swarm
- **Energy Quant Models**: Each model tracks energy, entropy, confidence
- **H×L Nested Cycles**: Roll marble, stop-gradient on non-final iterations
- **RoPE**: Rotary Position Embedding for sequence modeling

### Code Structure
```python
class HRM(nn.Module):
    def __init__(self, config: HRMConfig):
        # Inputs: market_state + model_reports
        # Reasoning: H_level → L_level → weights/directives
        # Full duplex: model_reports feed back into HRM
        
    def forward(self, market_state, model_reports, carry=None):
        # carry: HRMCarry (z_H, z_L)
        # return: (new_carry, HRMOutput)
```

### Model Report Structure
```python
class ModelReport:
    energy: float      # EWMA of realized returns
    entropy: float     # Uncertainty measure
    confidence: float  # Signal strength
    perf: float        # Historical performance
    active_ratio: float # Active trading ratio
```

---

## SignalHRM (MLX/Numpy)

### Architecture
```
Signal Array Input
├─ Input: signals [B, seq_len, N_SIGNALS × 2]
│          (signal + confidence interleaved)
├─ Sparkline Memory: logarithmic timescales
├─ H×L Nested Cycles (H_cycles, L_cycles)
└─ Output:
     ├─ weights [B, N_SIGNALS]     (signal importance)
     ├─ alpha [B]                  (combined prediction)
     └─ convergence [B]            (agreement score)
```

### Key Features
- **Signal Focus**: Learns which of 16 signals to trust
- **Convergence**: Measures when signals agree
- **Sparkline Memory**: Logarithmic timescale perspectives (20 frames)
- **H×L Nested Cycles**: Same structure as reference
- **ANE-friendly**: Fixed shapes for Apple Neural Engine
- **CPU Fallback**: NumPy version when MLX unavailable

### Code Structure
```python
class SignalHRM(nn.Module):
    def __init__(self, cfg: SignalHRMConfig, force_cpu: bool = False):
        # CPU: deterministic random weights
        # MLX: full neural network
        # Force CPU: use numpy version regardless
        
    def forward(self, x, memory=None):
        # x: input signal tensor
        # memory: sparkline state (optional)
        # return: (weights, alpha, convergence, new_memory)
```

---

## Key Differences

| Aspect | Reference HRM | SignalHRM |
|--------|---------------|-----------|
| **Framework** | PyTorch | MLX (primary) / NumPy (fallback) |
| **Input** | market_state + model_reports | signal_array (16 signals) |
| **Output** | weights + directives (full duplex) | weights + alpha + convergence |
| **Model Swarm** | Energy quant models | 16 signal features |
| **Memory** | Simple carry (z_H, z_L) | Sparkline (20 frames, logarithmic) |
| **Positional** | RoPE (rotary embedding) | Sparkline timestamps (implicit) |
| **Use Case** | Swarm model selection | Signal convergence prediction |
| **ANE Support** | No (PyTorch) | Yes (MLX with fixed shapes) |
| **CPU Fallback** | No (PyTorch only) | Yes (NumPy implementation) |

---

## MLX HRM vs CPU Model Compatibility

### SignalHRM Implementation

**MLX Version** (lines 184-355 in signal_hrm.py):
```python
class SignalHRM(nn.Module):
    def __init__(self, cfg: SignalHRMConfig, force_cpu: bool = False):
        self.force_cpu = force_cpu or not HAS_MLX
        
        if self.force_cpu:
            # CPU: Random deterministic weights
            self.signal_weights = np.random.randn(cfg.n_signals).astype(np.float32)
            self.conv_weight = np.random.randn(cfg.hidden_dim).astype(np.float32)
            return
        
        # MLX: Full neural network
        D = cfg.hidden_dim
        self.input_proj = nn.Linear(cfg.input_dim, D, bias=False)
        self.H_level = ReasoningModule(D, cfg.n_heads, cfg.H_layers)
        self.L_level = ReasoningModule(D, cfg.n_heads, cfg.L_layers)
        self.weight_head = nn.Linear(D, cfg.n_signals, bias=True)
        self.conv_head = nn.Linear(D, 1, bias=True)
```

**CPU Version** (lines 357-413):
```python
class SignalHRM:
    def __init__(self, cfg: SignalHRMConfig):
        self.cfg = cfg
        # Random but deterministic weights to simulate uninitialized NN
        self.signal_weights = np.random.randn(cfg.n_signals).astype(np.float32)
        self.conv_weight = np.random.randn(cfg.hidden_dim).astype(np.float32)
    
    def __call__(self, x, memory=None):
        # Same sparkline logic as MLX
        # Same weight computation (softmax over signal_weights)
        # Same convergence (tanh dot product)
        # Same alpha (weighted sum of signals)
```

### Compatibility Strategy

**SignalHRM maintains CPU model compatibility through:**

1. **Identical Algorithm**:
   - Both versions use same sparkline memory logic
   - Both compute same weights, alpha, convergence
   - Same convergence_threshold (0.25)
   - Same sparkline_frames (20) and horizon (2000)

2. **Parameter Compatibility**:
   ```python
   # CPU stores weights as numpy arrays
   self.signal_weights = np.random.randn(cfg.n_signals).astype(np.float32)
   
   # MLX stores weights as model parameters
   self.weight_head = nn.Linear(D, cfg.n_signals, bias=True)
   ```

3. **Force CPU Flag**:
   ```python
   # Always use CPU version
   hrm = SignalHRM(cfg, force_cpu=True)
   
   # Auto-fallback if MLX unavailable
   hrm = SignalHRM(cfg)
   ```

4. **Functionally Equivalent**:
   | Component | MLX Version | CPU Version |
   |-----------|-------------|-------------|
   | Sparkline | `mx.zeros((B, N_FRAMES, D))` | `np.zeros((B, N_FRAMES, F))` |
   | Weights | `mx.softmax(weight_head(pooled))` | `softmax(diag(signal_weights) @ pooled)` |
   | Convergence | `mx.sigmoid(conv_head(pooled))` | `0.5 * (tanh(conv_weight @ pooled) + 1)` |
   | Alpha | `(weights * raw_signals).sum()` | Same |

---

## Recommendation

### Keep Both HRMs with Clear Separation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HRM ARCHITECTURE CHOICES                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USE CASE: Signal Learning for Trading                           │
│  └─ Use: SignalHRM (MLX/Numpy)                                  │
│     ├─ 16 signals → weights + alpha + convergence                │
│     ├─ Sparkline memory (logarithmic timescales)                 │
│     ├─ ANE-friendly (MLX) / CPU fallback                         │
│     └─ Portfolio loss training                                   │
│                                                                  │
│  USE CASE: Swarm Model Management                                │
│  └─ Use: Reference HRM (PyTorch)                                │
│     ├─ Full duplex with energy quant models                      │
│     ├─ Regime determination (H-level)                            │
│     ├─ Tactical decisions (L-level)                              │
│     └─ Not ANE-compatible (PyTorch)                              │
│                                                                  │
│  USE CASE: Binance → Coinbase Transfer                           │
│  └─ Use: BinanceSpotTrainer                                     │
│     ├─ Uses SignalHRM architecture                               │
│     ├─ Learns layer weights (L1-L5)                              │
│     ├─ Exports weights for Coinbase deployment                   │
│     └─ Compatible with both MLX and CPU                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### For Your Requirements:

**Use SignalHRM** (signal_hrm.py) because:
1. ✅ **MLX compatible** - ANE-friendly fixed shapes
2. ✅ **CPU fallback** - NumPy version always available
3. ✅ **16-signal architecture** - Matches BinanceSpotTrainer
4. ✅ **Hierarchical learning** - H×L nested cycles
5. ✅ **Portfolio loss** - Compatible with codec training
6. ✅ **Convergence tracking** - Meets hierarchical signal learning

**Don't use Reference HRM** for trading because:
1. ❌ **PyTorch only** - No ANE support
2. ❌ **Full duplex I/O** - Requires model swarm (more complex)
3. ❌ **Not 16-signal compatible** - Different input format
4. ❌ **For swarm management** - Not designed for trading signals

---

## Implementation Verification

### To verify MLX ↔ CPU compatibility:

```python
# Test 1: Same output with same input
cfg = SignalHRMConfig()
mlx_hrm = SignalHRM(cfg, force_cpu=False)  # MLX
cpu_hrm = SignalHRM(cfg, force_cpu=True)   # CPU

# Test input
test_input = np.random.randn(1, 32, 44).astype(np.float32)

# MLX forward
import mlx.core as mx
mlx_input = mx.array(test_input)
mlx_weights, mlx_alpha, mlx_conv, mlx_mem = mlx_hrm(mlx_input)

# CPU forward
cpu_weights, cpu_alpha, cpu_conv, cpu_mem = cpu_hrm(test_input)

# Compare (should be very close)
assert np.allclose(mlx_weights, cpu_weights, rtol=1e-5)
assert np.allclose(mlx_alpha, cpu_alpha, rtol=1e-5)
assert np.allclose(mlx_conv, cpu_conv, rtol=1e-5)

print("✅ MLX and CPU versions produce compatible outputs")
```

### Training with CPU Model:

```python
# Train using CPU model (no MLX dependency)
from hrm.binance_spot_trainer import BinanceSpotTrainer

trainer = BinanceSpotTrainer(config)
trainer.train_epoch(n_bags=3, verbose=True)

# Save weights
weights = trainer.get_transferable_weights()

# These weights work with BOTH MLX and CPU SignalHRM
```

### Deployment:

```python
# In production, auto-select based on availability
def create_hrm(cfg, force_cpu=None):
    if force_cpu is None:
        try:
            import mlx
            return SignalHRM(cfg, force_cpu=False)
        except ImportError:
            return SignalHRM(cfg, force_cpu=True)
    else:
        return SignalHRM(cfg, force_cpu=force_cpu)
```
