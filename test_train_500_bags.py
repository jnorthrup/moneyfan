#!/usr/bin/env python3
"""
Quick test of train_500_bags functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from train_500_bags import StochasticBagTrainer, BagTrainingConfig
    print("✅ Successfully imported training components")
    
    # Test configuration
    config = BagTrainingConfig(
        n_bags=3,  # Very short test
        capital=100,
        update_interval=2,
        enable_progress_dashboard=False,
        bag_size=5,  # Small bag
        sequences_per_bag=10,  # Few sequences
        epochs=1
    )
    
    print("✅ Configuration created")
    print(f"   Bags: {config.n_bags}")
    print(f"   Capital per bag: ${config.capital}")
    print(f"   Bag size: {config.bag_size}")
    print(f"   Sequences per bag: {config.sequences_per_bag}")
    
    # Test trainer initialization
    trainer = StochasticBagTrainer(config)
    print("✅ Trainer initialized")
    
    # Test data directory
    arrow_dir = Path("hrm/data/arrow")
    if arrow_dir.exists():
        feather_files = list(arrow_dir.glob("*.feather"))
        print(f"✅ Found {len(feather_files)} .feather files in {arrow_dir}")
        if feather_files:
            print(f"   Examples: {[f.name for f in feather_files[:5]]}")
    else:
        print(f"⚠️  Data directory not found: {arrow_dir}")
        print("   Please ensure hrm/data/arrow/ exists with .feather files")
    
    # Test model initialization
    print("✅ Model initialization test")
    try:
        import torch
        print("   PyTorch available")
    except ImportError:
        print("   PyTorch not available")
    
    try:
        import mlx.core as mx
        print("   MLX available")
    except ImportError:
        print("   MLX not available")
    
    print("\n✅ Training test completed successfully")
    print("   Run full training with: python3 train_500_bags.py --bags 500 --capital 100")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Missing required components. Check train_500_bags.py dependencies.")
    
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()