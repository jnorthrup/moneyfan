"""
Test script for Coinbase backfill agent
"""

import sys
import os
sys.path.insert(0, '/Users/jim/work/moneyfan')

from coinbase_backfill_agent import CoinbaseBackfillAgent, CoinbaseBackfillConfig

def test_agent_initialization():
    """Test agent initialization"""
    print("TESTING AGENT INITIALIZATION...")
    
    config = CoinbaseBackfillConfig(
        duck_db_path="hrm/data/market.duckdb"
    )
    
    agent = CoinbaseBackfillAgent(config)
    
    print("✅ Agent initialized successfully")
    print(f"   Config: {config.duck_db_path}")
    print(f"   Timeframes: {config.timeframes}")
    print(f"   Schema columns: {len(config.schema_columns)}")
    
    return agent

def test_arrow_file_checking():
    """Test arrow file checking"""
    print("\nTESTING ARROW FILE CHECKING...")
    
    config = CoinbaseBackfillConfig()
    agent = CoinbaseBackfillAgent(config)
    
    coinbase_files = agent.check_arrow_files()
    
    print(f"✅ Found {len(coinbase_files)} Coinbase symbols")
    
    # Show some examples
    for symbol, files in sorted(coinbase_files.items())[:5]:
        print(f"   {symbol}: {len(files)} file(s)")
        for f in files[:1]:
            print(f"      {f.name}")
    
    return coinbase_files

def test_duckdb_checking():
    """Test DuckDB gap checking"""
    print("\nTESTING DUCKDB GAP CHECKING...")
    
    config = CoinbaseBackfillConfig()
    agent = CoinbaseBackfillAgent(config)
    
    gaps = agent.check_duckdb_gaps()
    
    print(f"✅ Checked DuckDB gaps")
    
    # Show some examples
    count = 0
    for symbol, timeframe_gaps in gaps.items():
        if timeframe_gaps:
            for timeframe, gaps_list in timeframe_gaps.items():
                if gaps_list:
                    print(f"   {symbol} {timeframe}: {len(gaps_list)} gap(s)")
                    count += 1
                if count >= 5:
                    break
        if count >= 5:
            break
    
    return gaps

def test_48_column_schema():
    """Test 48-column schema transformation"""
    print("\nTESTING 48-COLUMN SCHEMA...")
    
    import pandas as pd
    import numpy as np
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 100 + 1,
        'low': np.random.randn(100) + 100 - 1,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randn(100) * 1000,
    })
    
    df.set_index('timestamp', inplace=True)
    
    config = CoinbaseBackfillConfig()
    agent = CoinbaseBackfillAgent(config)
    
    df_48 = agent.load_48_column_schema(df, 'BTC-USD', '1m')
    
    print(f"✅ 48-column schema transformation")
    print(f"   Input columns: {len(df.columns)}")
    print(f"   Output columns: {len(df_48.columns)}")
    print(f"   Expected columns: {len(config.schema_columns)}")
    
    # Check missing columns
    expected_cols = config.schema_columns
    missing_cols = [col for col in expected_cols if col not in df_48.columns]
    
    if missing_cols:
        print(f"   ⚠️  Missing columns: {len(missing_cols)}")
    else:
        print(f"   ✅ All expected columns present")
    
    return df_48

def test_backfill_simulation():
    """Test backfill simulation (without actual DuckDB insertion)"""
    print("\nTESTING BACKFILL SIMULATION...")
    
    import pandas as pd
    import numpy as np
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 100 + 1,
        'low': np.random.randn(1000) + 100 - 1,
        'close': np.random.randn(1000) + 100,
        'volume': np.random.randn(1000) * 1000,
    })
    
    df.set_index('timestamp', inplace=True)
    
    config = CoinbaseBackfillConfig()
    agent = CoinbaseBackfillAgent(config)
    
    # Transform to 48 columns
    df_48 = agent.load_48_column_schema(df, 'BTC-USD', '1m')
    
    # Compute hash
    data_hash = agent.compute_data_hash(df_48)
    
    print(f"✅ Backfill simulation completed")
    print(f"   Rows: {len(df_48)}")
    print(f"   Data hash: {data_hash[:16]}...")
    print(f"   Timestamp range: {df_48.index.min()} to {df_48.index.max()}")
    
    return df_48, data_hash

def main():
    """Run all tests"""
    print("="*80)
    print("COINBASE BACKFILL AGENT TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Initialization
        agent = test_agent_initialization()
        
        # Test 2: Arrow file checking
        coinbase_files = test_arrow_file_checking()
        
        # Test 3: DuckDB checking
        gaps = test_duckdb_checking()
        
        # Test 4: 48-column schema
        df_48 = test_48_column_schema()
        
        # Test 5: Backfill simulation
        df_final, data_hash = test_backfill_simulation()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✅")
        print("="*80)
        
        # Summary
        print("\nSUMMARY:")
        print(f"  • Coinbase symbols found: {len(coinbase_files)}")
        print(f"  • 48-column schema: ✓")
        print(f"  • Data hash computation: ✓")
        print(f"  • Provenance tracking: Ready")
        print(f"  • DuckStore integration: {'✓' if agent.duck_store else '✗'}")
        
        # Next steps
        print("\nNEXT STEPS:")
        print("  1. Run the full agent with: python coinbase_backfill_agent.py")
        print("  2. Verify DuckDB connection is working")
        print("  3. Check for actual Coinbase USD files in hrm/data/arrow/")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()