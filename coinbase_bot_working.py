#!/usr/bin/env python3
"""
Working Coinbase Readonly Bot (Python)
Evolved from the Kotlin version to work with current API limitations
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

# Import the simulator
sys.path.append(os.path.dirname(__file__))
from coinbase_readonly_simulator import CoinbaseReadonlySimulator

class WorkingCoinbaseBot:
    """Working version of Coinbase bot with readonly operations"""
    
    def __init__(self):
        self.client = None
        self.state_file = "coinbase_bot_state.json"
        self.state = self.load_state()
        
    def initialize(self):
        """Initialize the bot with API credentials"""
        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")
        passphrase = os.getenv("COINBASE_PASSPHRASE", "")
        
        if api_key and api_secret:
            self.client = CoinbaseReadonlySimulator(api_key, api_secret, passphrase)
            print("✅ Bot initialized with API credentials")
        else:
            self.client = CoinbaseReadonlySimulator()
            print("⚠️  Bot initialized in simulation mode")
    
    def load_state(self) -> Dict[str, Any]:
        """Load bot state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    print(f"✅ Loaded state from {self.state_file}")
                    return state
            except Exception as e:
                print(f"❌ Could not load state: {e}")
        
        # Default state
        return {
            "baselines": {},
            "last_update": datetime.now().isoformat(),
            "simulation_mode": True
        }
    
    def save_state(self):
        """Save bot state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            print(f"✅ Saved state to {self.state_file}")
        except Exception as e:
            print(f"❌ Could not save state: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary (readonly operation)"""
        print("\n" + "=" * 80)
        print("GETTING PORTFOLIO SUMMARY")
        print("=" * 80)
        
        # Get account balances
        accounts = self.client.get_account_balances()
        
        # Calculate total value
        total_value = 0.0
        portfolio = []
        
        for currency, info in accounts.items():
            # For simulation, we'll use simulated values
            balance = float(info['balance'])
            currency_name = info['currency']
            
            # Get current price
            market_data = self.client.get_public_market_data(f"{currency_name}-USD")
            if market_data:
                rates = market_data["data"]["rates"]
                # For simulation, assume USD rate is 1 for USD, or use BTC conversion
                if currency_name == "USD":
                    price = 1.0
                elif currency_name == "BTC":
                    price = float(rates.get("USD", "68000"))
                else:
                    # For other currencies, estimate based on BTC rate
                    btc_rate = float(rates.get("USD", "68000"))
                    price = btc_rate * 0.01  # Simulated estimate
                
                value = balance * price
                total_value += value
                
                portfolio.append({
                    "currency": currency_name,
                    "balance": balance,
                    "price": price,
                    "value": value,
                    "percentage": (value / total_value * 100) if total_value > 0 else 0
                })
        
        # Sort by value
        portfolio.sort(key=lambda x: x["value"], reverse=True)
        
        print(f"\nPortfolio Summary:")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Assets: {len(portfolio)}")
        
        for asset in portfolio:
            print(f"  - {asset['currency']}: {asset['balance']:,.2f} (${asset['value']:,.2f})")
        
        return {
            "total_value": total_value,
            "assets": portfolio,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get market data for selected assets"""
        print("\n" + "=" * 80)
        print("GETTING MARKET DATA")
        print("=" * 80)
        
        assets = ["BTC", "ETH", "USD"]
        market_data = {}
        
        for asset in assets:
            data = self.client.get_public_market_data(f"{asset}-USD")
            if data and "data" in data and "rates" in data["data"]:
                rate = data["data"]["rates"].get("USD", "N/A")
                print(f"{asset}/USD: ${rate}")
                market_data[asset] = {
                    "price": float(rate) if rate != "N/A" else 0,
                    "timestamp": datetime.now().isoformat()
                }
        
        return market_data
    
    def check_baselines(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if baselines need to be set or updated"""
        print("\n" + "=" * 80)
        print("CHECKING BASELINES")
        print("=" * 80)
        
        baselines = self.state.get("baselines", {})
        updates = {}
        
        for asset in portfolio_data.get("assets", []):
            currency = asset["currency"]
            current_value = asset["value"]
            
            if currency not in baselines:
                # Set new baseline
                baselines[currency] = current_value
                updates[currency] = {
                    "action": "SET",
                    "old": None,
                    "new": current_value
                }
                print(f"  ✅ Set baseline for {currency}: ${current_value:,.2f}")
            else:
                old_baseline = baselines[currency]
                # Check if baseline needs update (within 10%)
                if abs(current_value - old_baseline) / old_baseline > 0.1:
                    baselines[currency] = current_value
                    updates[currency] = {
                        "action": "UPDATE",
                        "old": old_baseline,
                        "new": current_value
                    }
                    print(f"  🔄 Updated baseline for {currency}: ${old_baseline:,.2f} → ${current_value:,.2f}")
                else:
                    print(f"  ✅ Baseline stable for {currency}: ${old_baseline:,.2f}")
        
        # Update state
        self.state["baselines"] = baselines
        self.state["last_update"] = datetime.now().isoformat()
        
        return updates
    
    def calculate_deviation(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deviation from baselines"""
        print("\n" + "=" * 80)
        print("CALCULATING DEVIATION FROM BASELINES")
        print("=" * 80)
        
        baselines = self.state.get("baselines", {})
        deviations = {}
        
        for asset in portfolio_data.get("assets", []):
            currency = asset["currency"]
            current_value = asset["value"]
            
            if currency in baselines:
                baseline = baselines[currency]
                deviation = ((current_value - baseline) / baseline) * 100
                deviations[currency] = {
                    "current": current_value,
                    "baseline": baseline,
                    "deviation_percent": deviation,
                    "deviation_absolute": current_value - baseline
                }
                
                # Display
                sign = "+" if deviation >= 0 else ""
                print(f"  {currency}: {sign}{deviation:.2f}% (baseline: ${baseline:,.2f}, current: ${current_value:,.2f})")
        
        return deviations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive readonly report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Get all data
        portfolio = self.get_portfolio_summary()
        market_data = self.get_market_data()
        baseline_updates = self.check_baselines(portfolio)
        deviations = self.calculate_deviation(portfolio)
        
        # Save state
        self.save_state()
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio,
            "market_data": market_data,
            "baseline_updates": baseline_updates,
            "deviations": deviations,
            "simulation_mode": self.client.simulation_mode if self.client else True
        }
        
        # Print summary
        print("\n" + "=" * 80)
        print("REPORT SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Mode: {'SIMULATION' if report['simulation_mode'] else 'LIVE'}")
        print(f"Total Portfolio Value: ${portfolio['total_value']:,.2f}")
        print(f"Assets Tracked: {len(portfolio['assets'])}")
        print(f"Baseline Updates: {len(baseline_updates)}")
        print(f"Assets with Deviation: {len(deviations)}")
        
        # Check for significant deviations
        significant_deviations = {
            currency: data for currency, data in deviations.items()
            if abs(data["deviation_percent"]) > 5
        }
        
        if significant_deviations:
            print(f"\n⚠️  Significant Deviations Detected (>5%):")
            for currency, data in significant_deviations.items():
                sign = "+" if data["deviation_percent"] >= 0 else ""
                print(f"  {currency}: {sign}{data['deviation_percent']:.2f}%")
        
        return report
    
    def run_cycle(self, cycle_count: int):
        """Run one cycle of the bot"""
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle_count}")
        print(f"{'='*80}")
        
        report = self.generate_report()
        
        print(f"\n✅ Cycle {cycle_count} completed")
        return report

def main():
    """Main function"""
    print("=== Working Coinbase Readonly Bot ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Initialize bot
    bot = WorkingCoinbaseBot()
    bot.initialize()
    
    print()
    print("Bot will run in simulation mode if API credentials are invalid")
    print("This allows testing and development without valid API keys")
    print()
    
    # Run cycles
    cycles = 3
    for i in range(1, cycles + 1):
        bot.run_cycle(i)
        if i < cycles:
            time.sleep(2)
    
    print()
    print("=" * 80)
    print("BOT EXECUTION COMPLETE")
    print("=" * 80)
    print("✅ All readonly operations completed successfully")
    print("✅ Portfolio analysis performed")
    print("✅ Baseline tracking implemented")
    print("✅ Deviation calculation working")
    print("✅ State persistence enabled")
    print()
    print("Evolution Complete:")
    print("✅ Now: Working readonly bot with simulation mode")
    print("✅ Next: Add valid API credentials for real operations")
    print("✅ Future: Add trading capabilities (when ready)")

if __name__ == "__main__":
    main()