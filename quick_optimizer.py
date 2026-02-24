#!/usr/bin/env python3
"""
Quick Model Size Optimizer - Minimal Test Version
==================================================

Quickly tests 3 key configurations to find optimal model size.
Each test runs 20 episodes for rapid feedback.

Usage:
    python quick_optimizer.py
"""

import subprocess
import re
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    hidden_dim: int
    regime_layers: int
    tactical_layers: int
    attention_heads: int
    
    def to_string(self) -> str:
        return f"hidden={self.hidden_dim},regime={self.regime_layers},tactical={self.tactical_layers},heads={self.attention_heads}"


@dataclass
class TestResult:
    config: ModelConfig
    mean_loss: float
    cv: float
    stability: float
    success: bool
    error: str = ""


class QuickOptimizer:
    def __init__(self):
        # 3 key configurations to test
        self.configs = [
            ModelConfig(64, 2, 2, 4),    # Current (baseline)
            ModelConfig(96, 3, 3, 6),    # Moderate improvement
            ModelConfig(128, 3, 3, 8),   # Recommended improvement
        ]
    
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
    
    def run_test(self, config: ModelConfig, episodes: int = 20) -> TestResult:
        """Run a quick test with given config"""
        print(f"\n🧪 Testing: {config.to_string()}")
        print(f"Running {episodes} episodes...")
        
        cmd = [
            "python", "train.py",
            "--episodes", str(episodes),
            "--notional", "100",
            "--pretrain-only",
            "--fully-stochastic-pair-sampling",
            "--weights-path", "models/trained/hrm_latest_weights.npz"
        ]
        
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
                return TestResult(config, 0, 0, 0, False, "No pred_loss found")
            
            metrics = self.calculate_metrics(losses)
            return TestResult(
                config,
                metrics['mean'],
                metrics['cv'],
                metrics['stability'],
                True
            )
            
        except Exception as e:
            return TestResult(config, 0, 0, 0, False, str(e))
    
    def analyze_result(self, result: TestResult) -> str:
        """Analyze test result and provide recommendation"""
        if not result.success:
            return f"❌ Test failed: {result.error}"
        
        if result.mean_loss < 200 and result.cv < 0.4 and result.stability > 60:
            return "✅ OPTIMAL - Perfect configuration!"
        elif result.mean_loss < 300 and result.cv < 0.5 and result.stability > 40:
            return "✅ ACCEPTABLE - Good configuration"
        elif result.mean_loss < 400 and result.cv < 0.6 and result.stability > 30:
            return "⚠️ BORDERLINE - May need adjustment"
        else:
            return "❌ UNDERPOWERED - Needs larger model"
    
    def run(self):
        """Run quick optimization"""
        print("⚡ QUICK MODEL SIZE OPTIMIZER")
        print("="*60)
        print("Testing 3 configurations with 20 episodes each")
        print("Target: CV < 0.4, Mean loss < 200")
        print("="*60)
        
        # Stop existing training
        subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
        time.sleep(2)
        
        results = []
        
        for config in self.configs:
            result = self.run_test(config, episodes=20)
            results.append(result)
            
            analysis = self.analyze_result(result)
            print(f"\n📊 RESULT: {analysis}")
            
            if result.success:
                print(f"   Mean: {result.mean_loss:.1f}, CV: {result.cv:.3f}, Stability: {result.stability:.1f}")
            
            # Small pause between tests
            if config != self.configs[-1]:
                print(f"\n⏳ Waiting 30 seconds...")
                time.sleep(30)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*60}")
        
        # Sort by stability (best first)
        valid_results = [r for r in results if r.success]
        valid_results.sort(key=lambda x: x.stability, reverse=True)
        
        for i, r in enumerate(valid_results, 1):
            status = "🏆" if i == 1 else "✅" if r.stability > 40 else "⚠️"
            print(f"{status} {i}. {r.config.to_string()}")
            print(f"   CV: {r.cv:.3f}, Mean: {r.mean_loss:.1f}, Stability: {r.stability:.1f}")
        
        if valid_results:
            best = valid_results[0]
            print(f"\n🎯 RECOMMENDATION: Use {best.config.to_string()}")
            
            if best.config.hidden_dim > 64:
                print(f"   ✅ Current model (64) needs increase to {best.config.hidden_dim}")
            else:
                print(f"   ✅ Current model (64) is adequate")
        
        return results


def main():
    optimizer = QuickOptimizer()
    results = optimizer.run()
    return results


if __name__ == "__main__":
    main()