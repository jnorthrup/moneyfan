#!/usr/bin/env python3
"""
Continuous Optimizer Daemon - Runs Until Optimal Size Found
============================================================

Automatically runs model size tests until optimal configuration is found.
This daemon will:
1. Stop any running training
2. Test configurations
3. Analyze results
4. Repeat until optimal size is found

Usage:
    python continuous_optimizer_daemon.py
    python continuous_optimizer_daemon.py --max-runs 5
    python continuous_optimizer_daemon.py --quick-test
"""

import subprocess
import time
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse
import json


@dataclass
class ModelConfig:
    hidden_dim: int
    regime_layers: int
    tactical_layers: int
    attention_heads: int
    learning_rate: float = 1e-4
    
    def to_string(self) -> str:
        return f"h{self.hidden_dim}_r{self.regime_layers}_t{self.tactical_layers}_h{self.attention_heads}"
    
    def to_command(self, episodes: int) -> List[str]:
        """Generate command line for this config"""
        # For now, we'll modify train.py to accept these as arguments
        # Actually, we'll use a different approach: modify the model config file
        return [
            "python", "train.py",
            "--episodes", str(episodes),
            "--notional", "100",
            "--pretrain-only",
            "--fully-stochastic-pair-sampling",
            "--weights-path", "models/trained/hrm_latest_weights.npz",
            "--learning-rate", str(self.learning_rate)
        ]


@dataclass
class TestResult:
    config: ModelConfig
    episodes: int
    mean_loss: float
    cv: float
    stability: float
    success: bool
    error: str = ""
    
    def is_optimal(self) -> bool:
        """Check if meets optimal criteria"""
        return (self.mean_loss < 200 and 
                self.cv < 0.4 and 
                self.stability > 60)
    
    def is_acceptable(self) -> bool:
        """Check if meets acceptable criteria"""
        return (self.mean_loss < 300 and 
                self.cv < 0.5 and 
                self.stability > 40)


class ContinuousOptimizerDaemon:
    def __init__(self, episodes: int = 30, max_runs: int = 10, quick: bool = False):
        self.episodes_per_test = episodes if not quick else 15
        self.max_runs = max_runs
        self.quick = quick
        
        # Configurations to test (ordered from most likely to improve to least)
        self.configs_to_test = [
            # Focus on the most promising improvements first
            ModelConfig(96, 3, 3, 6),    # Moderate improvement - likely best
            ModelConfig(128, 3, 3, 8),   # Significant improvement
            ModelConfig(64, 3, 3, 8),    # More layers, same width
            ModelConfig(128, 2, 2, 6),   # Wider but fewer layers
            ModelConfig(192, 2, 2, 8),   # Much wider
            ModelConfig(256, 3, 3, 8),   # Very large
        ]
        
        self.results: List[TestResult] = []
        self.best_result: Optional[TestResult] = None
        
    def calculate_metrics(self, losses: List[float]) -> Dict:
        """Calculate variability metrics"""
        if not losses:
            return {'mean': 0, 'cv': 1, 'stability': 0}
        
        arr = np.array(losses)
        mean = float(np.mean(arr))
        cv = float(np.std(arr) / mean) if mean > 0 else 1.0
        stability = max(0, 100 - (cv * 100) - (min(np.max(arr) / 100, 50)))
        
        return {
            'mean': mean,
            'cv': cv,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'stability': stability
        }
    
    def extract_losses(self, output: str) -> List[float]:
        """Extract pred_loss from training output"""
        pattern = r'pred_loss=([\d.]+)'
        return [float(m) for m in re.findall(pattern, output)]
    
    def run_config_test(self, config: ModelConfig) -> TestResult:
        """Run a test with specific configuration"""
        print(f"\n{'='*70}")
        print(f"🧪 TESTING: {config.to_string()}")
        print(f"{'='*70}")
        
        # Create temporary model config file
        config_file = f"temp_config_{config.to_string()}.json"
        config_data = {
            "hidden_dim": config.hidden_dim,
            "regime_attn_layers": config.regime_layers,
            "tactical_attn_layers": config.tactical_layers,
            "n_heads": config.attention_heads,
            "n_codec_outputs": 24
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Modify train.py temporarily to use this config
        # For now, we'll use environment variable or command line
        cmd = config.to_command(self.episodes_per_test)
        
        # Set environment variable for custom config
        env = {}
        env['MODEL_CONFIG_OVERRIDE'] = config_file
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**subprocess.os.environ, **env}
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                print(line, end='')
            
            process.wait()
            output = ''.join(output_lines)
            
            # Clean up config file
            Path(config_file).unlink(missing_ok=True)
            
            losses = self.extract_losses(output)
            
            if not losses:
                return TestResult(config, 0, 0, 0, 0, False, "No pred_loss found")
            
            metrics = self.calculate_metrics(losses)
            
            return TestResult(
                config,
                len(losses),
                metrics['mean'],
                metrics['cv'],
                metrics['stability'],
                True
            )
            
        except Exception as e:
            Path(config_file).unlink(missing_ok=True)
            return TestResult(config, 0, 0, 0, 0, False, str(e))
    
    def should_continue_testing(self) -> bool:
        """Determine if we should continue testing"""
        if not self.results:
            return True
        
        # Check if we have an optimal result
        optimal_found = any(r.is_optimal() for r in self.results)
        if optimal_found:
            return False
        
        # Check if we've reached max runs
        if len(self.results) >= self.max_runs:
            return False
        
        # Check if we have an acceptable result after testing 3+ configs
        if len(self.results) >= 3 and any(r.is_acceptable() for r in self.results):
            return False
        
        return True
    
    def print_progress(self):
        """Print current progress"""
        print(f"\n{'='*70}")
        print(f"📊 PROGRESS UPDATE")
        print(f"{'='*70}")
        print(f"Tests completed: {len(self.results)}/{self.max_runs}")
        print(f"Optimal found: {'✅' if any(r.is_optimal() for r in self.results) else '❌'}")
        
        if self.results:
            # Find best result
            valid_results = [r for r in self.results if r.success]
            if valid_results:
                best = max(valid_results, key=lambda x: x.stability)
                print(f"Best config so far: {best.config.to_string()}")
                print(f"Best stability: {best.stability:.1f}/100")
        
        print(f"{'='*70}")
    
    def run_continuous_optimization(self):
        """Run continuous optimization until optimal size found"""
        print("🔄 CONTINUOUS MODEL OPTIMIZATION DAEMON")
        print("="*70)
        print(f"Target: CV < 0.4, Mean loss < 200")
        print(f"Episodes per test: {self.episodes_per_test}")
        print(f"Maximum test runs: {self.max_runs}")
        print("="*70)
        
        # Stop any running training
        print("\n🛑 Stopping any existing training...")
        subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
        time.sleep(2)
        
        test_count = 0
        
        while self.should_continue_testing():
            test_count += 1
            
            # Get next config to test
            if test_count <= len(self.configs_to_test):
                config = self.configs_to_test[test_count - 1]
            else:
                # If we've tested all configs but still not optimal,
                # try variations of the best result
                if self.results:
                    best = max([r for r in self.results if r.success], key=lambda x: x.stability)
                    # Try increasing hidden_dim
                    new_config = ModelConfig(
                        hidden_dim=min(best.config.hidden_dim + 64, 256),
                        regime_layers=best.config.regime_layers,
                        tactical_layers=best.config.tactical_layers,
                        attention_heads=best.config.attention_heads + 2
                    )
                    config = new_config
                else:
                    break
            
            print(f"\n🎯 Test {test_count}: {config.to_string()}")
            
            result = self.run_config_test(config)
            self.results.append(result)
            
            if result.success:
                print(f"\n✅ Test completed")
                print(f"   Mean loss: {result.mean_loss:.1f}")
                print(f"   CV: {result.cv:.3f}")
                print(f"   Stability: {result.stability:.1f}/100")
                
                if result.is_optimal():
                    print(f"   🏆 OPTIMAL CONFIGURATION FOUND!")
                    self.best_result = result
                    break
                elif result.is_acceptable():
                    print(f"   ✅ ACCEPTABLE - Good configuration")
                else:
                    print(f"   ❌ NEEDS IMPROVEMENT")
            else:
                print(f"\n❌ Test failed: {result.error}")
            
            # Pause between tests (except last)
            if self.should_continue_testing():
                print(f"\n⏳ Waiting 60 seconds before next test...")
                time.sleep(60)
            
            # Print progress every 3 tests
            if test_count % 3 == 0:
                self.print_progress()
        
        # Print final results
        self.print_final_results()
        
        return self.results
    
    def print_final_results(self):
        """Print final results and recommendation"""
        print(f"\n{'='*70}")
        print("🎯 FINAL RESULTS & RECOMMENDATION")
        print(f"{'='*70}")
        
        if not self.results:
            print("No test results available")
            return
        
        # Sort by stability
        valid_results = [r for r in self.results if r.success]
        valid_results.sort(key=lambda x: x.stability, reverse=True)
        
        print(f"\n📊 TESTED CONFIGURATIONS ({len(valid_results)} successful):")
        for i, r in enumerate(valid_results, 1):
            status = "🏆 OPTIMAL" if r.is_optimal() else "✅ ACCEPTABLE" if r.is_acceptable() else "❌ NEEDS WORK"
            print(f"{i}. {status} - {r.config.to_string()}")
            print(f"   CV: {r.cv:.3f}, Mean: {r.mean_loss:.1f}, Stability: {r.stability:.1f}")
        
        if valid_results:
            best = valid_results[0]
            print(f"\n🎯 RECOMMENDED CONFIGURATION:")
            print(f"   {best.config.to_string()}")
            print(f"   CV: {best.cv:.3f} (target: <0.4)")
            print(f"   Mean loss: {best.mean_loss:.1f} (target: <200)")
            print(f"   Stability: {best.stability:.1f}/100 (target: >60)")
            
            if best.is_optimal():
                print(f"\n✅ OPTIMAL CONFIGURATION FOUND!")
                print(f"   Use this configuration for full training")
            else:
                print(f"\n⚠️  No optimal configuration found")
                print(f"   Consider testing even larger models or adding regularization")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Use recommended config for full training")
        print(f"   2. Run 1000 episodes with optimal config")
        print(f"   3. Verify stability remains good")
        
        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Continuous model optimizer daemon')
    parser.add_argument('--episodes', type=int, default=30, help='Episodes per test')
    parser.add_argument('--max-runs', type=int, default=10, help='Maximum test runs')
    parser.add_argument('--quick-test', action='store_true', help='Quick test (15 episodes)')
    
    args = parser.parse_args()
    
    daemon = ContinuousOptimizerDaemon(
        episodes=args.episodes,
        max_runs=args.max_runs,
        quick=args.quick_test
    )
    
    results = daemon.run_continuous_optimization()
    return results


if __name__ == "__main__":
    main()