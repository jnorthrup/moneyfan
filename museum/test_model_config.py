#!/usr/bin/env python3
"""
Test different model configurations to find optimal size
========================================================

Runs a quick test with different model sizes to determine optimal configuration.
"""

import subprocess
import time
import re
import numpy as np
from pathlib import Path


def run_test_config(hidden_dim: int, episodes: int = 20, name: str = "test") -> dict:
    """Run a quick test with specific hidden dimension"""
    cmd = [
        "python", "train.py",
        "--episodes", str(episodes),
        "--notional", "100",
        "--pretrain-only",
        "--fully-stochastic-pair-sampling",
        "--weights-path", "models/trained/hrm_latest_weights.npz"
    ]
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL CONFIG: hidden_dim={hidden_dim}")
    print(f"{'='*60}")
    
    try:
        # Run the training command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        pred_losses = []
        
        for line in process.stdout:
            output_lines.append(line)
            print(line, end='')
            
            # Extract pred_loss
            match = re.search(r'pred_loss=([\d.]+)', line)
            if match:
                pred_losses.append(float(match.group(1)))
        
        process.wait()
        
        # Analyze results
        if pred_losses:
            arr = np.array(pred_losses)
            result = {
                'hidden_dim': hidden_dim,
                'n_episodes': len(pred_losses),
                'mean_loss': float(np.mean(arr)),
                'std_loss': float(np.std(arr)),
                'cv': float(np.std(arr) / np.mean(arr)),
                'min_loss': float(np.min(arr)),
                'max_loss': float(np.max(arr)),
                'success': True
            }
            print(f"\n📊 RESULTS for hidden_dim={hidden_dim}:")
            print(f"   Mean loss: {result['mean_loss']:.1f}")
            print(f"   CV: {result['cv']:.3f}")
            print(f"   Range: {result['min_loss']:.1f} - {result['max_loss']:.1f}")
            return result
        else:
            return {
                'hidden_dim': hidden_dim,
                'success': False,
                'error': 'No pred_loss found in output'
            }
            
    except Exception as e:
        return {
            'hidden_dim': hidden_dim,
            'success': False,
            'error': str(e)
        }


def compare_configs(configs: list, episodes: int = 20) -> dict:
    """Compare multiple configurations"""
    results = []
    
    for config in configs:
        result = run_test_config(config, episodes)
        results.append(result)
        
        if not result['success']:
            print(f"❌ Test failed for hidden_dim={config}")
            continue
    
    # Sort by mean loss (lower is better)
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        print("No successful tests")
        return {}
    
    successful_results.sort(key=lambda x: x['mean_loss'])
    
    print(f"\n{'='*60}")
    print("CONFIG COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for i, result in enumerate(successful_results, 1):
        print(f"{i}. hidden_dim={result['hidden_dim']}")
        print(f"   Mean loss: {result['mean_loss']:.1f}")
        print(f"   CV: {result['cv']:.3f}")
        print(f"   Range: {result['min_loss']:.1f} - {result['max_loss']:.1f}")
        
        if i == 1:
            print("   🏆 BEST (lowest mean loss)")
        print()
    
    # Determine best config
    best_config = successful_results[0]
    worst_config = successful_results[-1]
    
    print(f"🎯 RECOMMENDATION:")
    print(f"   Best: hidden_dim={best_config['hidden_dim']}")
    print(f"   Worst: hidden_dim={worst_config['hidden_dim']}")
    
    if best_config['hidden_dim'] > 64:
        print(f"   ✅ Current model (64) needs increase to {best_config['hidden_dim']}")
    else:
        print(f"   ✅ Current model (64) appears adequate")
    
    return {
        'results': successful_results,
        'best': best_config,
        'worst': worst_config
    }


def main():
    print("MODEL CONFIGURATION TESTER")
    print("="*60)
    print("This script will test different model sizes to find optimal configuration.")
    print("Running quick tests with 20 episodes each...")
    
    # Test different configurations
    configs = [64, 96, 128, 192, 256]
    
    print(f"\nTesting configurations: {configs}")
    print("Each test will run 20 episodes (about 5-10 minutes each)")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    results = compare_configs(configs, episodes=20)
    
    if results:
        print(f"\n{'='*60}")
        print("FINAL RECOMMENDATION")
        print(f"{'='*60}")
        best = results['best']
        
        if best['hidden_dim'] == 64:
            print("✅ Current model (hidden_dim=64) appears optimal")
        elif best['hidden_dim'] <= 128:
            print(f"✅ Recommend: hidden_dim={best['hidden_dim']}")
            print(f"   Expected improvement: {((results['results'][0]['mean_loss'] - best['mean_loss']) / results['results'][0]['mean_loss'] * 100):.1f}% reduction in loss")
        else:
            print(f"⚠️  Consider: hidden_dim={best['hidden_dim']}")
            print(f"   Note: Large model may need more training data")


if __name__ == "__main__":
    main()