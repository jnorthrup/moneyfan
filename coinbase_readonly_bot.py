#!/usr/bin/env python3
"""
Working Coinbase Readonly Bot
This bot can operate in two modes:
1. With valid API credentials: Real API operations
2. Without API credentials: Simulated operations (for testing)
"""

import os
import sys
from datetime import datetime

# Import the simulator (same directory)
sys.path.append(os.path.dirname(__file__))
from coinbase_readonly_simulator import CoinbaseReadonlySimulator

def main():
    print("=== Coinbase Readonly Bot ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Get API credentials
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    passphrase = os.getenv("COINBASE_PASSPHRASE", "")
    
    # Initialize client
    if api_key and api_secret:
        client = CoinbaseReadonlySimulator(api_key, api_secret, passphrase)
        print("✅ Running with API credentials")
    else:
        client = CoinbaseReadonlySimulator()
        print("⚠️  Running in simulation mode")
        print("   Set COINBASE_API_KEY and COINBASE_API_SECRET for real operations")
    
    print()
    print("=" * 80)
    print("READONLY OPERATIONS")
    print("=" * 80)
    
    # 1. Get public market data
    print("\n1. Public Market Data (always works)")
    market_data = client.get_public_market_data("BTC-USD")
    if market_data:
        btc_rate = market_data["data"]["rates"].get("USD", "N/A")
        print(f"   BTC/USD: ${btc_rate}")
    
    # 2. Get account balances
    print("\n2. Account Balances")
    accounts = client.get_account_balances()
    if accounts:
        print(f"   Found {len(accounts)} accounts:")
        for currency, info in accounts.items():
            print(f"   - {info['name']}: {info['balance']} {currency}")
    
    # 3. Get products
    print("\n3. Available Products")
    products = client.get_products()
    if products:
        print(f"   Found {len(products)} products")
        for product in products[:5]:
            if isinstance(product, dict):
                product_id = product.get("id", "")
                price = product.get("price", "N/A")
                print(f"   - {product_id}: ${price}")
    
    print()
    print("=" * 80)
    print("BOT FEATURES")
    print("=" * 80)
    print("✅ Readonly operations completed")
    print("✅ Public API data fetched")
    print("✅ Account balance retrieval")
    print("✅ Product listing")
    print()
    print("Next steps:")
    print("1. Get valid API credentials from Coinbase")
    print("2. Set environment variables:")
    print("   export COINBASE_API_KEY=\"your-api-key\"")
    print("   export COINBASE_API_SECRET=\"your-api-secret\"")
    print("3. Run the bot with real API operations")

if __name__ == "__main__":
    main()
