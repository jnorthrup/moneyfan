#!/usr/bin/env python3
"""
Continuous Model Size Optimizer
=================================

Automatically tests different model configurations until optimal size is found.
This system runs training cycles with different hidden dimensions and layer counts
to find the configuration that achieves optimal variability (CV < 0.4).

Usage:
    python continuous_model_optimizer.py
    python continuous_model_optimizer.py --quick-test (for fast testing)
"""

import subprocess
import time
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import sys


@dataclass
class ModelConfig:
    """Model configuration to test"""
    hidden_dim: int
    regime_layers: int
    tactical_layers: int
    attention_heads: int
    codec_outputs: int = 24
    
    def to_string(self) -> str:
        return f"hidden={self.hidden_dim},regime={self.regime_layers},tactical={self.tactical_layers},heads={self.attention_heads}"
    
    def get_file_suffix(self) -> str:
        return f"h{self.hidden_dim}_r{self.regime_layers}_t{self.tactical_layers}_h{self.attention_heads}"


@dataclass
class TestResult:
    """Results from a test run"""
    config: ModelConfig
    episodes: int
    mean_loss: float
    cv: float
    min_loss: float
    max_loss: float
    stability_score: float
    success: bool
    error_message: str = ""
    
    def is_optimal(self) -> bool:
        """Check if this config meets optimal criteria"""
        return (self.mean_loss < 200 and 
                self.cv < 0.4 and 
                self.stability_score > 60)
    
    def is_acceptable(self) -> bool:
        """Check if this config is acceptable"""
        return (self.mean_loss < 300 and 
                self.cv < 0.5 and 
                self.stability_score > 40)


class ContinuousModelOptimizer:
    """Continuously tests model configurations until optimal size is found"""
    
    def __init__(self, episodes_per_test: int = 50, quick_test: bool = False):
        self.episodes_per_test = episodes_per_test if not quick_test else 20
        self.quick_test = quick_test
        
        # Configurations to test (from underpowered to potentially overpowered)
        self.configs_to_test = [
            # Baseline (current)
            ModelConfig(hidden_dim=64, regime_layers=2, tactical_layers=2, attention_heads=4),
            
            # Moderate improvements
            ModelConfig(hidden_dim=96, regime_layers=2, tactical_layers=2, attention_heads=6),
            ModelConfig(hidden_dim=64, regime_layers=3, tactical_layers=3, attention_heads=8),
            
            # Significant improvements
            ModelConfig(hidden_dim=128, regime_layers=2, tactical_layers=2, attention_heads=6),
            ModelConfig(hidden_dim=128, regime_layers=3, tactical_layers=3, attention_heads=8),
            ModelConfig(hidden_dim=192, regime_layers=2, tactical_layers=2, attention_heads=8),
            
            # Large configurations
            ModelConfig(hidden_dim=256, regime_layers=3, tactical_layers=3, attention_heads=8),
            ModelConfig(hidden_dim=256, regime_layers=4, tactical_layers=4, attention_heads=8),
        ]
        
        self.results: List[TestResult] = []
        self.best_result: Optional[TestResult] = None
        
    def calculate_variability_metrics(self, losses: List[float]) -> Dict:
        """Calculate variability metrics from loss list"""
        if not losses:
            return {'mean': 0, 'cv': 1, 'stability': 0}
        
        arr = np.array(losses)
        mean_loss = float(np.mean(arr))
        std_loss = float(np.std(arr))
        cv = std_loss / mean_loss if mean_loss > 0 else 1.0
        
        # Stability score (0-100)
        stability_score = max(0, 100 - (cv * 100) - (min(np.max(arr) / 100, 50)))
        
        return {
            'mean': mean_loss,
            'cv': cv,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'stability': stability_score
        }
    
    def extract_losses_from_output(self, output: str) -> List[float]:
        """Extract pred_loss values from training output"""
        pattern = r'pred_loss=([\d.]+)'
        matches = re.findall(pattern, output)
        return [float(m) for m in matches]
    
    def run_training_test(self, config: ModelConfig) -> TestResult:
        """Run a training test with specific configuration"""
        print(f"\n{'='*70}")
        print(f"🧪 TESTING: {config.to_string()}")
        print(f"{'='*70}")
        print(f"Episodes: {self.episodes_per_test}")
        
        # Create custom configuration file for this test
        config_file = f"test_config_{config.get_file_suffix()}.json"
        config_data = {
            "hidden_dim": config.hidden_dim,
            "regime_attn_layers": config.regime_layers,
            "tactical_attn_layers": config.tactical_layers,
            "n_heads": config.attention_heads,
            "n_codec_outputs": config.codec_outputs
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Run training command
        cmd = [
            "python", "train.py",
            "--episodes", str(self.episodes_per_test),
            "--notional", "100",
            "--pretrain-only",
            "--fully-stochastic-pair-sampling",
            "--weights-path", "models/trained/hrm_latest_weights.npz"
        ]
        
        try:
            # Run the command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                print(line, end='')
            
            process.wait()
            output = ''.join(output_lines)
            
            # Extract losses
            losses = self.extract_losses_from_output(output)
            
            if not losses:
                return TestResult(
                    config=config,
                    episodes=0,
                    mean_loss=0,
                    cv=0,
                    min_loss=0,
                    max_loss=0,
                    stability_score=0,
                    success=False,
                    error_message="No pred_loss found in output"
                )
            
            # Calculate metrics
            metrics = self.calculate_variability_metrics(losses)
            
            result = TestResult(
                config=config,
                episodes=len(losses),
                mean_loss=metrics['mean'],
                cv=metrics['cv'],
                min_loss=metrics['min'],
                max_loss=metrics['max'],
                stability_score=metrics['stability'],
                success=True
            )
            
            # Clean up config file
            Path(config_file).unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            # Clean up config file if exists
            Path(config_file).unlink(missing_ok=True)
            
            return TestResult(
                config=config,
                episodes=0,
                mean_loss=0,
                cv=0,
                min_loss=0,
                max_loss=0,
                stability_score=0,
                success=False,
                error_message=str(e)
            )
    
    def print_test_result(self, result: TestResult, test_num: int):
        """Print formatted test result"""
        print(f"\n📊 TEST {test_num} RESULTS:")
        print(f"   Config: {result.config.to_string()}")
        print(f"   Episodes: {result.episodes}")
        
        if not result.success:
            print(f"   ❌ FAILED: {result.error_message}")
            return
        
        print(f"   Mean Loss: {result.mean_loss:.1f}")
        print(f"   CV: {result.cv:.3f}")
        print(f"   Range: {result.min_loss:.1f} - {result.max_loss:.1f}")
        print(f"   Stability: {result.stability_score:.1f}/100")
        
        if result.is_optimal():
            print(f"   🏆 OPTIMAL - Meets all criteria!")
        elif result.is_acceptable():
            print(f"   ✅ ACCEPTABLE - Could use refinement")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT")
    
    def should_stop_testing(self, result: TestResult) -> bool:
        """Determine if we should stop testing based on results"""
        if result.is_optimal():
            return True
        
        # If we've tested all configs, stop
        if len(self.results) >= len(self.configs_to_test):
            return True
        
        # If we've tested 3 configs and have an acceptable result
        if len(self.results) >= 3 and any(r.is_acceptable() for r in self.results):
            return True
        
        return False
    
    def print_final_recommendation(self):
        """Print final recommendation based on all tests"""
        print(f"\n{'='*70}")
        print("🎯 FINAL RECOMMENDATION")
        print(f"{'='*70}")
        
        if not self.results:
            print("No test results available")
            return
        
        # Sort results by stability score (descending)
        sorted_results = sorted(self.results, key=lambda x: x.stability_score, reverse=True)
        
        print(f"\n📈 TEST SUMMARY ({len(self.results)} configurations tested):")
        
        for i, result in enumerate(sorted_results, 1):
            status = "🏆 OPTIMAL" if result.is_optimal() else "✅ ACCEPTABLE" if result.is_acceptable() else "❌ NEEDS WORK"
            print(f"{i}. {status} - {result.config.to_string()}")
            print(f"   CV: {result.cv:.3f}, Mean: {result.mean_loss:.1f}, Stability: {result.stability_score:.1f}")
        
        # Best result
        best = sorted_results[0]
        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        print(f"   {best.config.to_string()}")
        print(f"   CV: {best.cv:.3f} (target: <0.4)")
        print(f"   Mean loss: {best.mean_loss:.1f} (target: <200)")
        print(f"   Stability: {best.stability_score:.1f}/100 (target: >60)")
        
        # Compare with current baseline (hidden_dim=64)
        baseline = next((r for r in self.results if r.config.hidden_dim == 64 and r.config.regime_layers == 2), None)
        if baseline:
            improvement = ((baseline.stability_score - best.stability_score) / baseline.stability_score) * 100
            print(f"\n📊 IMPROVEMENT OVER BASELINE (hidden_dim=64):")
            if improvement > 0:
                print(f"   ❌ Current config is better: {improvement:.1f}% worse")
            else:
                print(f"   ✅ Improvement: {abs(improvement):.1f}% better")
        
        print(f"\n💡 NEXT STEPS:")
        if best.is_optimal():
            print(f"   1. ✅ Use configuration: {best.config.to_string()}")
            print(f"   2. Run full training with this config")
            print(f"   3. Consider this as optimal model size")
        else:
            print(f"   1. ⚠️ Found acceptable config, but more testing may help")
            print(f"   2. Consider testing larger configurations")
            print(f"   3. Or add regularization to prevent overfitting")
        
        print(f"\n{'='*70}")
    
    def run_continuous_optimization(self):
        """Run continuous optimization until optimal size is found"""
        print("🔄 CONTINUOUS MODEL SIZING OPTIMIZER")
        print("="*70)
        print(f"Testing {len(self.configs_to_test)} configurations")
        print(f"Episodes per test: {self.episodes_per_test}")
        print(f"Target: CV < 0.4, Mean loss < 200")
        print("="*70)
        
        # First, stop any currently running training
        print("Stopping any existing training processes...")
        subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
        time.sleep(2)
        
        # Run tests for each configuration
        for i, config in enumerate(self.configs_to_test, 1):
            print(f"\n⏳ Test {i}/{len(self.configs_to_test)}")
            
            result = self.run_training_test(config)
            self.results.append(result)
            
            self.print_test_result(result, i)
            
            # Check if we should stop
            if self.should_stop_testing(result):
                if result.is_optimal():
                    print(f"\n✅ OPTIMAL CONFIGURATION FOUND!")
                else:
                    print(f"\n✅ ACCEPTABLE CONFIGURATION FOUND (testing stopped early)")
                break
            
            # Small pause between tests
            if i < len(self.configs_to_test):
                print(f"\n⏳ Waiting 30 seconds before next test...")
                time.sleep(30)
        
        # Print final recommendation
        self.print_final_recommendation()
        
        return self.results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous model sizing optimizer')
    parser.add_argument('--quick-test', action='store_true', help='Run quick tests (20 episodes each)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per test (default: 50)')
    
    args = parser.parse_args()
    
    optimizer = ContinuousModelOptimizer(
        episodes_per_test=args.episodes,
        quick_test=args.quick_test
    )
    
    results = optimizer.run_continuous_optimization()
    return results


if __name__ == "__main__":
    main()