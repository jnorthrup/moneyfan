"""
Fast test for Coinbase backfill agent (only a few symbols)
"""

import sys
import os
sys.path.insert(0, '/Users/jim/work/moneyfan')

from coinbase_backfill_agent import CoinbaseBackfillAgent, CoinbaseBackfillConfig
import asyncio

async def main():
    """Run backfill for a few symbols only"""
    
    # Create config with limited pairs
    config = CoinbaseBackfillConfig(
        duck_db_path="hrm/data/market.duckdb",
        coinbase_pairs=["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]  # Only 4 pairs
    )
    
    agent = CoinbaseBackfillAgent(config)
    
    print(f"\n{'='*80}")
    print("FAST COINBASE BACKFILL TEST")
    print(f"{'='*80}\n")
    
    # Check arrow files for specific symbols
    print("Checking arrow files...")
    coinbase_files = agent.check_arrow_files()
    
    # Filter to only our target symbols
    filtered_files = {}
    for symbol in config.coinbase_pairs:
        if symbol in coinbase_files:
            filtered_files[symbol] = coinbase_files[symbol]
            print(f"  Found {symbol}: {len(coinbase_files[symbol])} file(s)")
    
    if not filtered_files:
        print("  No target symbols found!")
        return
    
    # Backfill the filtered files
    print(f"\nBackfilling {len(filtered_files)} symbols...")
    backfill_results = agent.backfill_from_arrow(filtered_files)
    
    # Update results
    agent.results.update(backfill_results)
    
    # Generate report
    report = agent.generate_report()
    print(f"\n{report}")
    
    # Show summary
    print(f"\n{'='*80}")
    print("FAST TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Total rows backfilled: {agent.results.get('total_rows', 0)}")
    print(f"Pairs backfilled: {agent.results.get('total_pairs', 0)}")
    print(f"Failed pairs: {len(agent.results.get('failed_pairs', 0))}")
    
    # Verify data in database
    if agent.duck_store:
        print(f"\nVerifying database contents...")
        conn = agent.duck_store.conn
        
        for symbol in config.coinbase_pairs:
            result = conn.execute(
                "SELECT COUNT(*) as count FROM coinbase_source WHERE symbol = ?",
                (symbol,)
            ).fetchone()
            count = result[0]
            
            if count > 0:
                result = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM coinbase_source WHERE symbol = ?",
                    (symbol,)
                ).fetchone()
                print(f"  {symbol}: {count} rows from {result[0]} to {result[1]}")
            else:
                print(f"  {symbol}: No data")

if __name__ == "__main__":
    asyncio.run(main())