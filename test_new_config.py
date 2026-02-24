#!/usr/bin/env python3
"""
Test new model configuration with different hidden dimensions
==============================================================

Quick test to see if the new hidden-dim argument works in train.py
"""

import subprocess
import time
import re
import numpy as np


def test_config(hidden_dim: int, episodes: int = 10):
    """Test a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing hidden_dim={hidden_dim}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "train.py",
        "--episodes", str(episodes),
        "--notional", "100",
        "--pretrain-only",
        "--fully-stochastic-pair-sampling",
        "--weights-path", "models/trained/hrm_latest_weights.npz",
        "--hidden-dim", str(hidden_dim),
        "--regime-layers", "2",
        "--tactical-layers", "2",
        "--attention-heads", "4"
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
        
        # Extract losses
        pattern = r'pred_loss=([\d.]+)'
        matches = re.findall(pattern, output)
        losses = [float(m) for m in matches]
        
        if not losses:
            return None
        
        # Calculate metrics
        arr = np.array(losses)
        mean_loss = float(np.mean(arr))
        std_loss = float(np.std(arr))
        cv = std_loss / mean_loss if mean_loss > 0 else 1.0
        
        print(f"\n📊 RESULTS: hidden_dim={hidden_dim}")
        print(f"   Episodes: {len(losses)}")
        print(f"   Mean loss: {mean_loss:.1f}")
        print(f"   CV: {cv:.3f}")
        
        return {
            'hidden_dim': hidden_dim,
            'mean_loss': mean_loss,
            'cv': cv,
            'episodes': len(losses)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("🔄 TESTING NEW MODEL CONFIGURATION ARGUMENTS")
    print("="*60)
    
    # Stop any running training
    subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
    time.sleep(2)
    
    # Test different configurations
    configs = [64, 96, 128, 192, 256]
    results = []
    
    for config in configs:
        result = test_config(config, episodes=10)
        if result:
            results.append(result)
        
        # Small pause between tests
        if config != configs[-1]:
            print(f"\n⏳ Waiting 30 seconds...")
            time.sleep(30)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("No successful tests")
        return
    
    # Sort by CV (lower is better)
    results.sort(key=lambda x: x['cv'])
    
    for i, r in enumerate(results, 1):
        print(f"{i}. hidden_dim={r['hidden_dim']}")
        print(f"   Mean: {r['mean_loss']:.1f}, CV: {r['cv']:.3f}")
        
        if r['cv'] < 0.4 and r['mean_loss'] < 200:
            print(f"   ✅ OPTIMAL")
        elif r['cv'] < 0.5 and r['mean_loss'] < 300:
            print(f"   ✅ ACCEPTABLE")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT")
    
    # Best config
    best = results[0]
    print(f"\n🎯 RECOMMENDATION: Use hidden_dim={best['hidden_dim']}")
    
    if best['hidden_dim'] > 64:
        print(f"   ✅ Current model (64) needs increase to {best['hidden_dim']}")
    else:
        print(f"   ✅ Current model (64) appears adequate")


if __name__ == "__main__":
    main()