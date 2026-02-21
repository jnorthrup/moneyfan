#!/usr/bin/env python3
"""
Quick test of bag trading marathon functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from bag_trading_marathon import MarathonTradingEngine, MarathonConfig
    print("✅ Successfully imported marathon components")
    
    # Test configuration
    config = MarathonConfig(
        hours=1,  # Short test
        initial_capital=100,
        update_interval=2,
        enable_progress_dashboard=False,
        n_codecs=4,  # Reduced for testing
        bag_size=5   # Small bag for testing
    )
    
    print("✅ Configuration created")
    print(f"   Hours: {config.hours}")
    print(f"   Capital: ${config.initial_capital}")
    print(f"   Codecs: {config.n_codecs}")
    print(f"   Bag size: {config.bag_size}")
    
    # Test engine initialization
    engine = MarathonTradingEngine(config)
    print("✅ Engine initialized")
    
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
    
    print("\n✅ Marathon test completed successfully")
    print("   Run full marathon with: python bag_trading_marathon.py --hours 24 --capital 1000")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Missing required components. Check paper_trading.py dependencies.")
    
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()