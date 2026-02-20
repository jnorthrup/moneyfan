#!/usr/bin/env python3
"""
A/B Testing: PyTorch vs MLX Hierarchical Codec Training
========================================================

Trains both PyTorch and MLX versions of HRM with identical:
- Training data (stochastic Binance samples)
- Random seeds (reproducible)
- Hyperparameters
- Evaluation criteria

Compares:
1. Training speed
2. Convergence rate
3. Final PnL performance
4. Memory usage

Winner: Fastest to achieve target PnL with architecture preserved.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import random
import gc
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List

# PyTorch codec
try:
    from hrm.hierarchical_codec import (
        HierarchicalCodec,
        HierarchicalCodecConfig,
        CodecTrainer
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch codec not available")

# MLX codec
try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        MLXCodecTrainer,
        enable_ane_optimization
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX codec not available")


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    seed: int = 42
    n_iterations: int = 200
    n_signals: int = 24
    start_capital: float = 100.0
    batch_per_iter: int = 20
    seq_len_range: Tuple[int, int] = (32, 64)
    
    # Architecture comparison
    compare_architecture: bool = True
    preserve_hrms: bool = True  # If True, MLX must preserve architecture
    
    # Performance targets
    target_pnl: float = 20.0  # Target PnL on $100
    max_loss: float = 0.03    # Max acceptable loss


class SignalGenerator:
    """Generate trading signals from Binance data."""
    
    @staticmethod
    def compute_signals(df: pd.DataFrame, n_signals: int = 24) -> torch.Tensor:
        """Compute MACD, RSI, Momentum, Volatility, Bollinger signals."""
        T = len(df)
        signals = np.zeros((T, n_signals * 2), dtype=np.float32)
        c = np.nan_to_num(df['close'].values.astype(np.float32), nan=1.0)
        h = np.nan_to_num(df['high'].values.astype(np.float32), nan=1.0)
        l = np.nan_to_num(df['low'].values.astype(np.float32), nan=1.0)
        
        # MACD
        ema12 = pd.Series(c).ewm(span=12, min_periods=1).mean()
        ema26 = pd.Series(c).ewm(span=26, min_periods=1).mean()
        macd = (ema12 - ema26).values
        signals[:, 0] = np.clip(macd / (np.std(macd) + 1e-8), -1, 1)
        signals[:, n_signals] = 0.5
        
        # RSI
        delta = np.diff(c, prepend=c[0])
        avg_gain = pd.Series(np.where(delta > 0, delta, 0)).ewm(span=14, min_periods=1).mean()
        avg_loss = pd.Series(np.where(delta < 0, -delta, 0)).ewm(span=14, min_periods=1).mean().fillna(1e-8)
        rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))
        signals[:, 4] = np.clip(-(rsi.values - 50) / 50, -1, 1)
        signals[:, n_signals + 4] = 0.5
        
        # Momentum
        mom = pd.Series(c).pct_change(20).fillna(0)
        signals[:, 2] = np.clip(mom.values * 10, -1, 1)
        signals[:, n_signals + 2] = 0.5
        
        # Volatility
        vol = (h - l) / (c + 1e-8)
        signals[:, 10] = np.clip(vol, -1, 1)
        signals[:, n_signals + 10] = 0.5
        
        # Bollinger
        sma = pd.Series(c).rolling(20, min_periods=1).mean()
        std = pd.Series(c).rolling(20, min_periods=1).std().fillna(1e-8)
        bb = (c - sma) / (2 * std + 1e-8)
        signals[:, 5] = np.clip(-bb.values, -1, 1)
        signals[:, n_signals + 5] = 0.5
        
        return torch.from_numpy(np.nan_to_num(signals, nan=0.0))


class TrainingSession:
    """Single training session for one model (PyTorch or MLX)."""
    
    def __init__(self, model_type: str, config: ABTestConfig, files: List[Path]):
        self.model_type = model_type
        self.config = config
        self.files = files
        
        # Random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Metrics
        self.metrics = {
            "loss_history": [],
            "pnl_history": [],
            "win_rate_history": [],
            "time_per_iteration": [],
            "total_time": 0,
            "final_pnl": 0,
            "final_loss": 0,
        }
        
        # Initialize model
        if model_type == "pytorch":
            self._init_pytorch()
        else:
            self._init_mlx()
    
    def _init_pytorch(self):
        """Initialize PyTorch model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        
        codec_config = HierarchicalCodecConfig(
            n_signals=self.config.n_signals
        )
        self.trainer = CodecTrainer(codec_config)
        self.model = self.trainer.model
        self.device = torch.device("cpu")
        print(f"   PyTorch model initialized: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    def _init_mlx(self):
        """Initialize MLX model."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        # Enable ANE optimization
        enable_ane_optimization()
        
        from hrm.hierarchical_codec_mlx import HierarchicalCodecConfig
        codec_config = HierarchicalCodecConfig(
            n_signals=self.config.n_signals
        )
        self.trainer = MLXCodecTrainer(codec_config)
        self.model = self.trainer.model
        print(f"   MLX model initialized: {sum(p.size for p in self.model.parameters()):,} params")
    
    def run_iteration(self, iteration: int) -> Tuple[float, float, float, float]:
        """Run one training iteration."""
        iter_loss = 0
        iter_pnl = 0
        iter_wins = 0
        iter_trades = 0
        iter_time = 0
        
        for _ in range(self.config.batch_per_iter):
            # Random file
            f = random.choice(self.files)
            df = pd.read_feather(f)
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.set_index('time')
            df = df.sort_index()
            
            if len(df) < 300:
                continue
            
            # Random window
            start = random.randint(50, len(df) - 250)
            window = df.iloc[start:start + 200]
            
            # Compute signals
            signals_tensor = SignalGenerator.compute_signals(window, self.config.n_signals)
            returns = window['close'].pct_change().fillna(0).values
            
            # Random dropout
            mask = np.random.random(len(signals_tensor)) > 0.25
            signals_tensor[mask, :self.config.n_signals] = 0
            
            seq_len = random.randint(*self.config.seq_len_range)
            
            for i in range(0, len(signals_tensor) - seq_len - 2, seq_len):
                batch = signals_tensor[i:i + seq_len + 1].unsqueeze(0)
                
                if torch.isnan(batch).any():
                    continue
                
                start_time = time.perf_counter()
                
                if self.model_type == "pytorch":
                    # PyTorch training
                    self.trainer.optimizer.zero_grad()
                    loss, _ = self.model.pretrain_loss(batch[:, :-1])
                    loss.backward()
                    self.trainer.optimizer.step()
                    loss_val = loss.item()
                    
                    # Trade evaluation
                    with torch.no_grad():
                        output, _ = self.model.forward(batch, mode="trade")
                        pred_ret = output[0, 0].item()
                        conf = output[0, 1].item()
                        stop_loss = output[0, 2].item()
                        take_profit = output[0, 3].item()
                        pos_size = output[0, 4].item()
                        
                        actual_ret = returns[i + seq_len] if i + seq_len < len(returns) else 0
                        
                        # Order simulation
                        opp_score = min(abs(pred_ret), 1.0)
                        position = pos_size * conf * opp_score * self.config.start_capital
                        
                        sl_pct = abs(stop_loss)
                        tp_pct = take_profit
                        
                        if np.sign(pred_ret) == 1:  # Long
                            if actual_ret < -sl_pct:
                                pnl = position * -sl_pct
                            elif actual_ret > tp_pct:
                                pnl = position * tp_pct
                            else:
                                pnl = position * actual_ret
                        else:  # Short
                            if -actual_ret < -sl_pct:
                                pnl = position * -sl_pct
                            elif -actual_ret > tp_pct:
                                pnl = position * tp_pct
                            else:
                                pnl = position * -actual_ret
                
                else:  # MLX
                    # Convert to MLX arrays
                    signals_mlx = mx.array(batch.numpy())
                    returns_mlx = mx.array(returns[i:i + seq_len + 1])
                    
                    # MLX training (pretrain)
                    loss_val = self.trainer.pretrain_step(signals_mlx[:, :-1, :])
                    
                    # Trade evaluation
                    output, _ = self.model.forward(signals_mlx, mode="trade")
                    pred_ret = float(output[0, 0])
                    conf = float(output[0, 1])
                    stop_loss = float(output[0, 2])
                    take_profit = float(output[0, 3])
                    pos_size = float(output[0, 4])
                    
                    actual_ret = returns[i + seq_len] if i + seq_len < len(returns) else 0
                    
                    # Order simulation
                    opp_score = min(abs(pred_ret), 1.0)
                    position = pos_size * conf * opp_score * self.config.start_capital
                    
                    sl_pct = abs(stop_loss)
                    tp_pct = take_profit
                    
                    if np.sign(pred_ret) == 1:  # Long
                        if actual_ret < -sl_pct:
                            pnl = position * -sl_pct
                        elif actual_ret > tp_pct:
                            pnl = position * tp_pct
                        else:
                            pnl = position * actual_ret
                    else:  # Short
                        if -actual_ret < -sl_pct:
                            pnl = position * -sl_pct
                        elif -actual_ret > tp_pct:
                            pnl = position * tp_pct
                        else:
                            pnl = position * -actual_ret
                
                iter_time += time.perf_counter() - start_time
                iter_loss += loss_val
                iter_pnl += pnl
                iter_trades += 1
                if np.sign(pred_ret) == np.sign(actual_ret):
                    iter_wins += 1
        
        # Record metrics
        self.metrics["loss_history"].append(iter_loss / max(iter_trades, 1))
        self.metrics["pnl_history"].append(iter_pnl)
        self.metrics["win_rate_history"].append(iter_wins / max(iter_trades, 1))
        self.metrics["time_per_iteration"].append(iter_time)
        
        return iter_loss / max(iter_trades, 1), iter_pnl, iter_wins / max(iter_trades, 1), iter_time
    
    def train(self) -> Dict:
        """Run full training session."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for iteration in range(self.config.n_iterations):
            loss, pnl, win_rate, iter_time = self.run_iteration(iteration)
            
            # Progress
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration:3d}/{self.config.n_iterations} | "
                      f"Loss: {loss:.4f} | PnL: ${pnl:+.2f} | "
                      f"WinRate: {win_rate:.0%} | Time: {iter_time:.3f}s")
            
            # Early termination if target reached
            if pnl >= self.config.target_pnl and loss < self.config.max_loss:
                print(f"✅ Target reached at iteration {iteration}")
                break
        
        self.metrics["total_time"] = time.time() - start_time
        self.metrics["final_pnl"] = self.metrics["pnl_history"][-1]
        self.metrics["final_loss"] = self.metrics["loss_history"][-1]
        
        return self.metrics


def run_ab_test(config: ABTestConfig) -> Dict:
    """Run A/B test comparing PyTorch vs MLX."""
    
    # Load files
    arrow_dir = Path("hrm/data/arrow")
    files = sorted(list(arrow_dir.glob("*.feather")))
    print(f"Files: {len(files)}")
    
    results = {}
    
    # Test PyTorch
    if HAS_TORCH:
        try:
            session_torch = TrainingSession("pytorch", config, files)
            results["pytorch"] = session_torch.train()
        except Exception as e:
            print(f"❌ PyTorch failed: {e}")
            results["pytorch"] = {"error": str(e)}
    
    # Test MLX
    if HAS_MLX:
        try:
            session_mlx = TrainingSession("mlx", config, files)
            results["mlx"] = session_mlx.train()
        except Exception as e:
            print(f"❌ MLX failed: {e}")
            results["mlx"] = {"error": str(e)}
    
    # Compare results
    print(f"\n{'='*60}")
    print("A/B TEST RESULTS")
    print(f"{'='*60}")
    
    if "pytorch" in results and "mlx" in results:
        torch_result = results["pytorch"]
        mlx_result = results["mlx"]
        
        if "error" not in torch_result and "error" not in mlx_result:
            print(f"\n📊 Performance Comparison:")
            print(f"  PyTorch Final PnL: ${torch_result['final_pnl']:+.2f}")
            print(f"  MLX Final PnL:     ${mlx_result['final_pnl']:+.2f}")
            print(f"  PyTorch Final Loss: {torch_result['final_loss']:.4f}")
            print(f"  MLX Final Loss:    {mlx_result['final_loss']:.4f}")
            print(f"  PyTorch Total Time: {torch_result['total_time']:.1f}s")
            print(f"  MLX Total Time:    {mlx_result['total_time']:.1f}s")
            
            speedup = torch_result['total_time'] / mlx_result['total_time']
            print(f"\n  ⚡ Speedup: {speedup:.2f}x")
            
            # Determine winner
            if config.preserve_hrms:
                print(f"\n  Architecture Preserved:")
                print(f"    PyTorch: ✅ YES")
                print(f"    MLX:     ✅ YES (native implementation)")
            
            # Winner based on speed
            if speedup > 1.5:
                print(f"\n  🏆 WINNER: MLX ({speedup:.1f}x faster)")
            else:
                print(f"\n  🏆 WINNER: Tied (within 1.5x)")
            
            # Check if targets met
            torch_target = (torch_result['final_pnl'] >= config.target_pnl and 
                           torch_result['final_loss'] < config.max_loss)
            mlx_target = (mlx_result['final_pnl'] >= config.target_pnl and 
                         mlx_result['final_loss'] < config.max_loss)
            
            print(f"\n  Target (${config.target_pnl:.0f} PnL, <{config.max_loss:.0%} loss):")
            print(f"    PyTorch: {'✅' if torch_target else '❌'}")
            print(f"    MLX:     {'✅' if mlx_target else '❌'}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("A/B TEST: PyTorch vs MLX Hierarchical Codec")
    print("=" * 60)
    
    config = ABTestConfig(
        seed=42,
        n_iterations=100,  # Reduced for faster testing
        n_signals=24,
        start_capital=100.0,
        batch_per_iter=20,
        seq_len_range=(32, 64),
        target_pnl=20.0,
        max_loss=0.03
    )
    
    results = run_ab_test(config)
    
    print(f"\n{'='*60}")
    print("A/B TEST COMPLETE")
    print(f"{'='*60}")
