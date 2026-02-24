#!/usr/bin/env python3
"""
Continuous Model Size Optimizer - Finds Optimal Configuration
==============================================================

Automatically runs training with different model sizes until optimal configuration is found.
This script will:
1. Stop any running training
2. Test different model configurations
3. Analyze variability (CV) and mean loss
4. Continue until CV < 0.4 and mean loss < 200
5. Report optimal configuration

Usage:
    python continuous_optimizer.py --quick
    python continuous_optimizer.py --max-tests 10
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
    regime_layers: int = 2
    tactical_layers: int = 2
    attention_heads: int = 4
    
    def to_string(self) -> str:
        return f"h{self.hidden_dim}_r{self.regime_layers}_t{self.tactical_layers}_h{self.attention_heads}"
    
    def to_command(self, episodes: int) -> List[str]:
        """Generate training command"""
        return [
            "python", "train.py",
            "--episodes", str(episodes),
            "--notional", "100",
            "--pretrain-only",
            "--fully-stochastic-pair-sampling",
            # Don't load existing weights - train from scratch
            "--hidden-dim", str(self.hidden_dim),
            "--regime-layers", str(self.regime_layers),
            "--tactical-layers", str(self.tactical_layers),
            "--attention-heads", str(self.attention_heads)
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
        return (self.mean_loss < 200 and self.cv < 0.4 and self.stability > 60)
    
    def is_acceptable(self) -> bool:
        return (self.mean_loss < 300 and self.cv < 0.5 and self.stability > 40)


class ContinuousOptimizer:
    def __init__(self, episodes: int = 20, max_tests: int = 8, quick: bool = False):
        self.episodes_per_test = episodes if not quick else 15
        self.max_tests = max_tests
        
        # Configurations to test (ordered by expected improvement)
        self.configs = [
            ModelConfig(96, 2, 2, 4),   # Moderate increase
            ModelConfig(96, 3, 3, 6),   # More layers + heads
            ModelConfig(128, 2, 2, 4),  # Larger width
            ModelConfig(128, 3, 3, 6),  # Large + more layers
            ModelConfig(192, 2, 2, 6),  # Much larger
            ModelConfig(256, 3, 3, 8),  # Very large
            ModelConfig(384, 3, 3, 8),  # Extremely large
            ModelConfig(512, 4, 4, 8),  # Maximum reasonable size
        ]
        
        self.results: List[TestResult] = []
        self.best_result: Optional[TestResult] = None
    
    def extract_losses(self, output: str) -> List[float]:
        """Extract pred_loss from training output"""
        pattern = r'pred_loss=([\d.]+)'
        return [float(m) for m in re.findall(pattern, output)]
    
    def calculate_metrics(self, losses: List[float]) -> Dict:
        """Calculate variability metrics"""
        if not losses:
            return {'mean': 0, 'cv': 1, 'stability': 0}
        
        arr = np.array(losses)
        mean = float(np.mean(arr))
        cv = float(np.std(arr) / mean) if mean > 0 else 1.0
        stability = max(0, 100 - (cv * 100) - (min(np.max(arr) / 100, 50)))
        
        return {'mean': mean, 'cv': cv, 'stability': stability}
    
    def run_test(self, config: ModelConfig) -> TestResult:
        """Run a training test with given configuration"""
        print(f"\n{'='*70}")
        print(f"🧪 TESTING: {config.to_string()}")
        print(f"{'='*70}")
        
        cmd = config.to_command(self.episodes_per_test)
        
        try:
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
            return TestResult(config, 0, 0, 0, 0, False, str(e))
    
    def analyze_result(self, result: TestResult) -> str:
        """Analyze test result"""
        if not result.success:
            return f"❌ Test failed: {result.error}"
        
        if result.is_optimal():
            return "🏆 OPTIMAL - Perfect configuration!"
        elif result.is_acceptable():
            return "✅ ACCEPTABLE - Good configuration"
        elif result.mean_loss < 500 and result.cv < 0.6:
            return "⚠️  BORDERLINE - May need more capacity"
        else:
            return "❌ UNDERPOWERED - Needs larger model"
    
    def should_stop(self) -> bool:
        """Check if we should stop testing"""
        if not self.results:
            return False
        
        # Check if we have optimal result
        if any(r.is_optimal() for r in self.results):
            return True
        
        # Check if we've reached max tests
        if len(self.results) >= self.max_tests:
            return True
        
        # Check if we have acceptable result after testing 3+ configs
        if len(self.results) >= 3 and any(r.is_acceptable() for r in self.results):
            return True
        
        return False
    
    def run(self):
        """Run continuous optimization"""
        print("🔄 CONTINUOUS MODEL OPTIMIZATION")
        print("="*70)
        print(f"Episodes per test: {self.episodes_per_test}")
        print(f"Max tests: {self.max_tests}")
        print(f"Target: CV < 0.4, Mean loss < 200")
        print("="*70)
        
        # Stop any running training
        print("\n🛑 Stopping any existing training...")
        subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
        time.sleep(2)
        
        # Run tests
        for i, config in enumerate(self.configs, 1):
            print(f"\n⏳ Test {i}/{len(self.configs)}")
            
            result = self.run_test(config)
            self.results.append(result)
            
            analysis = self.analyze_result(result)
            print(f"\n📊 RESULT: {analysis}")
            
            if result.success:
                print(f"   Mean: {result.mean_loss:.1f}, CV: {result.cv:.3f}, Stability: {result.stability:.1f}")
            
            # Check if we should stop
            if self.should_stop():
                if result.is_optimal():
                    print(f"\n✅ OPTIMAL CONFIGURATION FOUND!")
                else:
                    print(f"\n✅ ACCEPTABLE CONFIGURATION FOUND")
                break
            
            # Pause between tests
            if i < len(self.configs) and not self.should_stop():
                print(f"\n⏳ Waiting 60 seconds...")
                time.sleep(60)
        
        # Print final results
        self.print_final_results()
        
        return self.results
    
    def print_final_results(self):
        """Print final results"""
        print(f"\n{'='*70}")
        print("🎯 FINAL RESULTS")
        print(f"{'='*70}")
        
        if not self.results:
            print("No test results")
            return
        
        # Sort by stability
        valid_results = [r for r in self.results if r.success]
        valid_results.sort(key=lambda x: x.stability, reverse=True)
        
        print(f"\n📊 TESTED CONFIGURATIONS ({len(valid_results)} successful):")
        for i, r in enumerate(valid_results, 1):
            status = "🏆" if r.is_optimal() else "✅" if r.is_acceptable() else "❌"
            print(f"{status} {i}. {r.config.to_string()}")
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
                print(f"   Use this for full training (1000 episodes)")
            else:
                print(f"\n⚠️  No optimal configuration found")
                print(f"   Current config may be borderline")
                print(f"   Consider adding regularization or data augmentation")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Use recommended config for full training")
        print(f"   2. Run with: --hidden-dim {best.config.hidden_dim} --regime-layers {best.config.regime_layers} --tactical-layers {best.config.tactical_layers} --attention-heads {best.config.attention_heads}")
        
        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Continuous model optimizer')
    parser.add_argument('--quick', action='store_true', help='Quick test (15 episodes)')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per test')
    parser.add_argument('--max-tests', type=int, default=8, help='Maximum number of tests')
    
    args = parser.parse_args()
    
    optimizer = ContinuousOptimizer(
        episodes=args.episodes,
        max_tests=args.max_tests,
        quick=args.quick
    )
    
    results = optimizer.run()
    return results


if __name__ == "__main__":
    main()