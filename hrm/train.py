#!/usr/bin/env python3
"""
Start HRM Training
Quick start for training on stochastic Coinbase bags
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fiduciary_controller import create_fiduciary_hrm
from coinbase_pipeline import CoinbasePipeline
from continuous_trainer import ContinuousTrainer


def main():
    print("=" * 70)
    print("   FIDUCIARY CONTROLLER HRM - Training Stochastic Coinbase Bags")
    print("=" * 70)
    
    model, config = create_fiduciary_hrm(n_instruments=64, n_models=12)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Config: {config.n_instruments} instruments, {config.n_models} models")
    print(f"        hidden={config.hidden_dim}, layers={config.n_layers}")
    
    pipeline = CoinbasePipeline()
    
    if '--init' in sys.argv or not os.path.exists("hrm/data/coinbase.db"):
        print("\n[INIT] Pulling historical data from Coinbase...")
        pipeline.initialize(pull_days=int(sys.argv[sys.argv.index('--days')+1]) if '--days' in sys.argv else 365)
    
    trainer = ContinuousTrainer(model, config, pipeline)
    
    if '--resume' in sys.argv:
        trainer.load_checkpoint()
    
    print("\n" + "=" * 70)
    print("   STARTING CONTINUOUS TRAINING")
    print("   Ctrl+C to stop and save checkpoint")
    print("=" * 70 + "\n")
    
    try:
        trainer.train_continuous(
            epochs_per_round=10,
            checkpoint_every=100,
            ab_test_every=500
        )
    except KeyboardInterrupt:
        print("\nStopped by user")
        checkpoint_path = trainer.save_checkpoint()
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Display final ranking benchmark
        print("\n" + "=" * 70)
        print("   FINAL HRM RANKING BENCHMARK")
        print("=" * 70)
        import subprocess
        import sys
        try:
            subprocess.run([sys.executable, "hrm/scores.py", "--checkpoint", str(checkpoint_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running scores: {e}")


if __name__ == "__main__":
    main()
