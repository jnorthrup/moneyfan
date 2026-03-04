#!/usr/bin/env python3
"""
Run Autograd-Based Model Sizing Optimizer
==========================================

Simpler script to run the autograd optimizer with minimal setup.

Usage:
    python run_autograd_optimizer.py --quick
    python run_autograd_optimizer.py --iterations 5
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Check if MLX is available
    try:
        import mlx.core as mx
        print("✅ MLX available - running autograd optimization")
        optimizer_module = "autograd_model_optimizer"
    except ImportError:
        print("⚠️  MLX not available - running gradient-free optimization")
        optimizer_module = "quick_optimizer"
    
    # Build command
    cmd = [sys.executable, f"{optimizer_module}.py"]
    
    # Add arguments
    if "--quick" in sys.argv:
        cmd.append("--quick")
    
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 < len(sys.argv):
            cmd.extend(["--iterations", sys.argv[idx + 1]])
    
    if "--episodes" in sys.argv:
        idx = sys.argv.index("--episodes")
        if idx + 1 < len(sys.argv):
            cmd.extend(["--episodes", sys.argv[idx + 1]])
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("="*70)
    
    # Run the optimizer
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running optimizer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())