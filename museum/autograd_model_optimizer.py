#!/usr/bin/env python3
"""
Autograd-Based Continuous Model Sizing Optimizer
=================================================

Uses MLX autograd to optimize model configuration by:
1. Running training with current model size
2. Analyzing pretraining loss variability (CV)
3. Computing gradients of stability metrics w.r.t. model parameters
4. Adjusting model size using gradient descent
5. Repeating until optimal configuration found

Usage:
    python autograd_model_optimizer.py
    python autograd_model_optimizer.py --quick
"""

import subprocess
import time
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import argparse

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available - using gradient-free optimization")


@dataclass
class ModelParameters:
    """Model parameters to optimize"""
    hidden_dim: float = 64.0
    regime_layers: float = 2.0
    tactical_layers: float = 2.0
    attention_heads: float = 4.0
    
    def to_config(self) -> Dict:
        """Convert to configuration dictionary"""
        return {
            "hidden_dim": int(max(32, min(self.hidden_dim, 512))),
            "regime_attn_layers": int(max(1, min(self.regime_layers, 6))),
            "tactical_attn_layers": int(max(1, min(self.tactical_layers, 6))),
            "n_heads": int(max(2, min(self.attention_heads, 16))),
            "n_codec_outputs": 24
        }
    
    def to_string(self) -> str:
        cfg = self.to_config()
        return f"h{cfg['hidden_dim']}_r{cfg['regime_attn_layers']}_t{cfg['tactical_attn_layers']}_h{cfg['n_heads']}"
    
    def to_array(self) -> mx.array:
        """Convert to MLX array for autograd"""
        return mx.array([
            self.hidden_dim,
            self.regime_layers,
            self.tactical_layers,
            self.attention_heads
        ])
    
    @classmethod
    def from_array(cls, arr: mx.array) -> 'ModelParameters':
        """Create from MLX array"""
        arr_np = np.array(arr)
        return cls(
            hidden_dim=float(arr_np[0]),
            regime_layers=float(arr_np[1]),
            tactical_layers=float(arr_np[2]),
            attention_heads=float(arr_np[3])
        )


@dataclass
class TrainingMetrics:
    """Metrics from a training run"""
    mean_loss: float
    cv: float
    stability: float
    episodes: int
    min_loss: float
    max_loss: float
    
    def stability_score(self) -> mx.array:
        """Compute stability score as MLX array"""
        # Target: CV < 0.4, mean_loss < 200
        cv_penalty = mx.maximum(self.cv - 0.4, 0.0) * 100
        loss_penalty = mx.maximum(self.mean_loss - 200, 0.0) * 0.1
        stability = mx.maximum(100.0 - cv_penalty - loss_penalty, 0.0)
        return stability
    
    def variability_penalty(self) -> mx.array:
        """Penalty for high variability (CV > 0.4)"""
        return mx.maximum(self.cv - 0.4, 0.0)
    
    def loss_penalty(self) -> mx.array:
        """Penalty for high mean loss (> 200)"""
        return mx.maximum(self.mean_loss - 200, 0.0) * 0.01


class AutogradModelOptimizer:
    """Optimizes model parameters using MLX autograd"""
    
    def __init__(self, episodes: int = 30, quick: bool = False, lr: float = 0.1):
        self.episodes_per_test = episodes if not quick else 15
        self.learning_rate = lr
        
        # Current model parameters (starting point)
        self.params = ModelParameters(
            hidden_dim=64.0,
            regime_layers=2.0,
            tactical_layers=2.0,
            attention_heads=4.0
        )
        
        # Best parameters found
        self.best_params = None
        self.best_stability = 0.0
        
        # History for tracking progress
        self.history: List[Dict] = []
    
    def calculate_variability_metrics(self, losses: List[float]) -> Dict:
        """Calculate variability metrics from loss list"""
        if not losses:
            return {'mean': 0, 'cv': 1.0, 'stability': 0}
        
        arr = np.array(losses)
        mean_loss = float(np.mean(arr))
        std_loss = float(np.std(arr))
        cv = std_loss / mean_loss if mean_loss > 0 else 1.0
        
        # Stability score (0-100)
        stability = max(0, 100 - (cv * 100) - (min(np.max(arr) / 100, 50)))
        
        return {
            'mean': mean_loss,
            'cv': cv,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'stability': stability
        }
    
    def extract_losses_from_output(self, output: str) -> List[float]:
        """Extract pred_loss values from training output"""
        pattern = r'pred_loss=([\d.]+)'
        matches = re.findall(pattern, output)
        return [float(m) for m in matches]
    
    def run_training_with_config(self, config: Dict) -> TrainingMetrics:
        """Run training with specific configuration and return metrics"""
        # Create config file
        config_file = f"temp_config_{int(time.time())}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training
        cmd = [
            "python", "train.py",
            "--episodes", str(self.episodes_per_test),
            "--notional", "100",
            "--pretrain-only",
            "--fully-stochastic-pair-sampling",
            "--weights-path", "models/trained/hrm_latest_weights.npz"
        ]
        
        try:
            # Set environment variable for config override
            env = dict(subprocess.os.environ)
            env['MODEL_CONFIG_OVERRIDE'] = config_file
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
            
            process.wait()
            output = ''.join(output_lines)
            
            # Clean up config file
            Path(config_file).unlink(missing_ok=True)
            
            # Extract losses
            losses = self.extract_losses_from_output(output)
            
            if not losses:
                return TrainingMetrics(
                    mean_loss=1000.0,
                    cv=1.0,
                    stability=0.0,
                    episodes=0,
                    min_loss=1000.0,
                    max_loss=1000.0
                )
            
            metrics = self.calculate_variability_metrics(losses)
            
            return TrainingMetrics(
                mean_loss=metrics['mean'],
                cv=metrics['cv'],
                stability=metrics['stability'],
                episodes=len(losses),
                min_loss=metrics['min'],
                max_loss=metrics['max']
            )
            
        except Exception as e:
            print(f"Error running training: {e}")
            return TrainingMetrics(
                mean_loss=1000.0,
                cv=1.0,
                stability=0.0,
                episodes=0,
                min_loss=1000.0,
                max_loss=1000.0
            )
    
    def compute_loss(self, params_array: mx.array) -> mx.array:
        """Compute loss for autograd optimization"""
        params = ModelParameters.from_array(params_array)
        config = params.to_config()
        
        print(f"Testing: {params.to_string()}")
        
        metrics = self.run_training_with_config(config)
        
        # Store in history
        self.history.append({
            'params': params.to_string(),
            'mean_loss': metrics.mean_loss,
            'cv': metrics.cv,
            'stability': metrics.stability,
            'episodes': metrics.episodes
        })
        
        # Compute loss (negative stability for maximization)
        stability = metrics.stability_score()
        loss = -stability  # We want to maximize stability
        
        print(f"   Mean loss: {metrics.mean_loss:.1f}")
        print(f"   CV: {metrics.cv:.3f}")
        print(f"   Stability: {metrics.stability:.1f}/100")
        print(f"   Autograd loss: {loss.item():.3f}")
        
        # Track best
        if stability > self.best_stability:
            self.best_stability = stability
            self.best_params = params
            print(f"   🏆 NEW BEST: {params.to_string()}")
        
        return loss
    
    def optimize_with_autograd(self, max_iterations: int = 10, patience: int = 3):
        """Optimize model parameters using MLX autograd"""
        if not HAS_MLX:
            print("MLX not available - using gradient-free optimization")
            return self.optimize_gradient_free(max_iterations)
        
        print("🔄 AUTOGRAID-BASED MODEL OPTIMIZATION")
        print("="*70)
        print(f"Max iterations: {max_iterations}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Episodes per test: {self.episodes_per_test}")
        print("="*70)
        
        # Initialize parameters
        params_array = self.params.to_array()
        params_array = mx.array(params_array, dtype=mx.float32)
        
        # Create optimizer (using simple gradient descent)
        # Note: MLX doesn't have built-in optimizers like PyTorch
        # We'll implement manual gradient descent
        
        best_value = float('-inf')
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # Compute loss and gradients
            loss_fn = self.compute_loss
            loss_value, grads = mx.value_and_grad(loss_fn)(params_array)
            
            print(f"\n📊 GRADIENT INFORMATION:")
            print(f"   Loss: {loss_value.item():.3f}")
            print(f"   Gradients: {grads}")
            
            # Gradient descent update
            params_array = params_array - self.learning_rate * grads

            # Clip parameters to reasonable ranges (PER-PARAMETER clipping)
            # Each mx.clip() applies to the ENTIRE array, so we must use numpy for per-index clipping
            import numpy as np
            params_np = np.array(params_array)
            params_np[0] = np.clip(params_np[0], 32, 512)      # hidden_dim: 32-512
            params_np[1] = np.clip(params_np[1], 1, 6)          # regime_layers: 1-6
            params_np[2] = np.clip(params_np[2], 1, 6)          # tactical_layers: 1-6
            params_np[3] = np.clip(params_np[3], 2, 16)         # heads: 2-16
            params_array = mx.array(params_np)
            
            # Check for improvement
            current_value = -loss_value.item()
            if current_value > best_value:
                best_value = current_value
                no_improvement_count = 0
                print(f"✅ IMPROVEMENT: Stability increased to {current_value:.1f}")
            else:
                no_improvement_count += 1
                print(f"⚠️  NO IMPROVEMENT: {no_improvement_count}/{patience}")
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"\n🎯 OPTIMIZATION CONVERGED (no improvement for {patience} iterations)")
                break
            
            # Convert to config for next iteration
            self.params = ModelParameters.from_array(params_array)
            print(f"Next config: {self.params.to_string()}")
        
        # Print final results
        self.print_final_results()
        
        return self.params
    
    def optimize_gradient_free(self, max_iterations: int = 10):
        """Optimize without autograd using random search"""
        print("🔄 GRADIENT-FREE OPTIMIZATION (MLX not available)")
        print("="*70)
        
        # Define search space
        configs_to_test = [
            {"hidden_dim": 96, "regime_attn_layers": 3, "tactical_attn_layers": 3, "n_heads": 6},
            {"hidden_dim": 128, "regime_attn_layers": 2, "tactical_attn_layers": 2, "n_heads": 6},
            {"hidden_dim": 128, "regime_attn_layers": 3, "tactical_attn_layers": 3, "n_heads": 8},
            {"hidden_dim": 192, "regime_attn_layers": 2, "tactical_attn_layers": 2, "n_heads": 8},
            {"hidden_dim": 256, "regime_attn_layers": 3, "tactical_attn_layers": 3, "n_heads": 8},
        ]
        
        results = []
        
        for i, config in enumerate(configs_to_test[:max_iterations], 1):
            print(f"\nTEST {i}/{min(len(configs_to_test), max_iterations)}: {config}")
            
            metrics = self.run_training_with_config(config)
            results.append({
                'config': config,
                'metrics': metrics,
                'stability': metrics.stability
            })
            
            # Track best
            if metrics.stability > self.best_stability:
                self.best_stability = metrics.stability
                self.best_params = config
                print(f"🏆 NEW BEST: {config}")
            
            # Early stopping if optimal found
            if metrics.mean_loss < 200 and metrics.cv < 0.4 and metrics.stability > 60:
                print(f"✅ OPTIMAL CONFIGURATION FOUND!")
                break
        
        # Print results
        self.print_final_results()
        
        return results
    
    def print_final_results(self):
        """Print final optimization results"""
        print(f"\n{'='*70}")
        print("🎯 FINAL OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        if not self.history:
            print("No optimization history available")
            return
        
        # Print history
        print(f"\n📊 OPTIMIZATION HISTORY ({len(self.history)} tests):")
        for i, h in enumerate(self.history, 1):
            print(f"{i}. {h['params']}")
            print(f"   Mean: {h['mean_loss']:.1f}, CV: {h['cv']:.3f}, Stability: {h['stability']:.1f}")
        
        # Print best
        if self.best_params:
            print(f"\n🏆 BEST CONFIGURATION FOUND:")
            if isinstance(self.best_params, ModelParameters):
                best_str = self.best_params.to_string()
            else:
                best_str = f"h{self.best_params['hidden_dim']}_r{self.best_params['regime_attn_layers']}_t{self.best_params['tactical_attn_layers']}_h{self.best_params['n_heads']}"
            
            print(f"   {best_str}")
            print(f"   Stability: {self.best_stability:.1f}/100")
            
            if self.best_stability > 60:
                print(f"\n✅ OPTIMAL CONFIGURATION!")
            elif self.best_stability > 40:
                print(f"\n⚠️  ACCEPTABLE CONFIGURATION (consider refinement)")
            else:
                print(f"\n❌ NO OPTIMAL CONFIGURATION FOUND - need more testing")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Use best config for full training")
        print(f"   2. Run 1000 episodes to verify")
        print(f"   3. Adjust if needed")
        
        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Autograd-based model optimizer')
    parser.add_argument('--quick', action='store_true', help='Quick test (15 episodes)')
    parser.add_argument('--episodes', type=int, default=30, help='Episodes per test')
    parser.add_argument('--iterations', type=int, default=10, help='Max optimization iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for gradient descent')
    
    args = parser.parse_args()
    
    optimizer = AutogradModelOptimizer(
        episodes=args.episodes,
        quick=args.quick,
        lr=args.lr
    )
    
    if HAS_MLX:
        result = optimizer.optimize_with_autograd(max_iterations=args.iterations)
    else:
        result = optimizer.optimize_gradient_free(max_iterations=args.iterations)
    
    return result


if __name__ == "__main__":
    main()