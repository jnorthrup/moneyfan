#!/usr/bin/env python3
"""
Live Trading Demo - Non-interactive version
Demonstrates the complete live trading environment
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal, ROUND_HALF_UP

# Import our trading systems
sys.path.append(os.path.dirname(__file__))

class LiveTradingDemo:
    """Demo version of live trading environment"""
    
    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")
        self.passphrase = os.getenv("COINBASE_PASSPHRASE", "")
        
        self.mode = "SIMULATION" if not (self.api_key and self.api_secret) else "LIVE"
        self.setup_logging()
        
        # Configuration
        self.config = {
            "trading_pairs": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "initial_balance": Decimal("10000"),
            "max_position_percent": Decimal("0.3"),
            "fee_structure": {
                "maker_fee": Decimal("0.004"),  # 0.4%
                "taker_fee": Decimal("0.006"),  # 0.6%
                "spread": Decimal("0.0025")     # 0.25%
            }
        }
        
        # State
        self.balance = self.config["initial_balance"]
        self.portfolio = {}
        self.trade_history = []
        self.state_file = "demo_trading_state.json"
        
        self.load_state()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_demo.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_state(self):
        """Load trading state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f, parse_float=Decimal)
                    self.balance = Decimal(str(state.get("balance", self.balance)))
                    self.portfolio = {k: Decimal(str(v)) for k, v in state.get("portfolio", {}).items()}
                    self.trade_history = state.get("trade_history", [])
                self.logger.info(f"✅ Loaded state from {self.state_file}")
            except Exception as e:
                self.logger.error(f"❌ Could not load state: {e}")
    
    def save_state(self):
        """Save trading state"""
        try:
            state = {
                "balance": str(self.balance),
                "portfolio": {k: str(v) for k, v in self.portfolio.items()},
                "trade_history": self.trade_history,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"✅ Saved state to {self.state_file}")
        except Exception as e:
            self.logger.error(f"❌ Could not save state: {e}")
    
    def get_price(self, pair: str) -> Optional[Decimal]:
        """Get current market price"""
        try:
            import requests
            response = requests.get(f"https://api.coinbase.com/v2/prices/{pair}/spot")
            if response.ok:
                data = response.json()
                if "data" in data:
                    return Decimal(data["data"]["amount"])
        except Exception as e:
            self.logger.error(f"Error getting price for {pair}: {e}")
        return None
    
    def calculate_fees(self, amount: Decimal, is_maker: bool = True) -> Dict[str, Decimal]:
        """Calculate fees for a trade"""
        fee_rate = self.config["fee_structure"]["maker_fee"] if is_maker else self.config["fee_structure"]["taker_fee"]
        spread_rate = self.config["fee_structure"]["spread"]
        
        trading_fee = amount * fee_rate
        spread = amount * spread_rate
        total_fee = trading_fee + spread
        
        return {
            "trading_fee": trading_fee,
            "spread": spread,
            "total_fee": total_fee,
            "effective_rate": (total_fee / amount * 100) if amount > 0 else Decimal("0")
        }
    
    def execute_trade(self, pair: str, side: str, amount: Decimal) -> Dict[str, Any]:
        """Execute a trade (simulated or real)"""
        base_currency, quote_currency = pair.split("-")
        
        # Get price
        price = self.get_price(pair)
        if not price:
            return {"success": False, "error": "Could not fetch price"}
        
        print(f"\n{'='*60}")
        print(f"EXECUTING {side} {pair}")
        print(f"{'='*60}")
        print(f"Price: ${price}")
        
        if side == "BUY":
            # Check balance
            if amount > self.balance:
                return {"success": False, "error": f"Insufficient balance: ${self.balance}"}
            
            # Calculate base amount
            base_amount = amount / price
            base_amount = base_amount.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            
            # Calculate fees
            fees = self.calculate_fees(amount, is_maker=True)
            total_cost = amount + fees["total_fee"]
            
            # Execute
            self.balance -= total_cost
            if base_currency in self.portfolio:
                self.portfolio[base_currency] += base_amount
            else:
                self.portfolio[base_currency] = base_amount
            
            # Record
            trade = {
                "timestamp": datetime.now().isoformat(),
                "pair": pair,
                "side": side,
                "amount": str(base_amount),
                "value": str(amount),
                "price": str(price),
                "fees": {k: str(v) for k, v in fees.items()},
                "total_cost": str(total_cost)
            }
            self.trade_history.append(trade)
            
            print(f"✅ BUY Executed:")
            print(f"   Amount: {base_amount} {base_currency}")
            print(f"   Value: ${amount}")
            print(f"   Fees: ${fees['total_fee']}")
            print(f"   Total Cost: ${total_cost}")
            print(f"   Remaining: ${self.balance}")
            
            return {"success": True, "trade": trade}
        
        elif side == "SELL":
            # Check portfolio
            if base_currency not in self.portfolio:
                return {"success": False, "error": f"No {base_currency} in portfolio"}
            
            if amount > self.portfolio[base_currency]:
                return {"success": False, "error": f"Insufficient {base_currency}"}
            
            # Calculate quote amount
            quote_amount = amount * price
            quote_amount = quote_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            
            # Calculate fees
            fees = self.calculate_fees(quote_amount, is_maker=True)
            
            # Execute
            self.portfolio[base_currency] -= amount
            if self.portfolio[base_currency] == 0:
                del self.portfolio[base_currency]
            
            net_proceeds = quote_amount - fees["total_fee"]
            self.balance += net_proceeds
            
            # Record
            trade = {
                "timestamp": datetime.now().isoformat(),
                "pair": pair,
                "side": side,
                "amount": str(amount),
                "value": str(quote_amount),
                "price": str(price),
                "fees": {k: str(v) for k, v in fees.items()},
                "net_proceeds": str(net_proceeds)
            }
            self.trade_history.append(trade)
            
            print(f"✅ SELL Executed:")
            print(f"   Amount: {amount} {base_currency}")
            print(f"   Value: ${quote_amount}")
            print(f"   Fees: ${fees['total_fee']}")
            print(f"   Net Proceeds: ${net_proceeds}")
            print(f"   New Balance: ${self.balance}")
            
            return {"success": True, "trade": trade}
        
        return {"success": False, "error": "Invalid side"}
    
    def analyze_portfolio(self) -> Dict[str, Any]:
        """Analyze current portfolio"""
        print(f"\n{'='*60}")
        print("PORTFOLIO ANALYSIS")
        print(f"{'='*60}")
        
        total_value = self.balance
        positions = {}
        
        for currency, amount in self.portfolio.items():
            if amount > 0:
                price = self.get_price(f"{currency}-USD")
                if price:
                    value = amount * price
                    positions[currency] = {
                        "amount": amount,
                        "price": price,
                        "value": value
                    }
                    total_value += value
        
        print(f"Balance: ${self.balance}")
        print(f"Positions:")
        for currency, data in positions.items():
            percentage = (data["value"] / total_value * 100) if total_value > 0 else 0
            print(f"  {currency}: {data['amount']} @ ${data['price']} = ${data['value']} ({percentage:.1f}%)")
        print(f"Total Portfolio Value: ${total_value}")
        
        return {
            "balance": self.balance,
            "positions": positions,
            "total_value": total_value
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate trading report"""
        print(f"\n{'='*60}")
        print("TRADING REPORT")
        print(f"{'='*60}")
        
        portfolio = self.analyze_portfolio()
        total_trades = len(self.trade_history)
        total_fees = Decimal("0")
        
        for trade in self.trade_history:
            if "fees" in trade:
                total_fees += Decimal(trade["fees"]["total_fee"])
        
        # Calculate P&L
        initial_balance = self.config["initial_balance"]
        profit_loss = portfolio["total_value"] - initial_balance
        profit_loss_percent = (profit_loss / initial_balance * 100) if initial_balance > 0 else Decimal("0")
        
        print(f"\n📊 Performance:")
        print(f"  Initial Balance: ${initial_balance}")
        print(f"  Current Balance: ${portfolio['balance']}")
        print(f"  Portfolio Value: ${portfolio['total_value'] - portfolio['balance']}")
        print(f"  Total Account Value: ${portfolio['total_value']}")
        print(f"  Profit/Loss: ${profit_loss} ({profit_loss_percent:.2f}%)")
        
        print(f"\n📈 Trading Activity:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Fees Paid: ${total_fees}")
        
        if total_trades > 0:
            print(f"\n📋 Recent Trades:")
            for trade in self.trade_history[-5:]:
                print(f"  {trade['timestamp'][:19]}: {trade['side']} {trade['pair']}")
        
        return {
            "account_summary": {
                "balance": str(self.balance),
                "total_value": str(portfolio["total_value"]),
                "profit_loss": str(profit_loss),
                "profit_loss_percent": str(profit_loss_percent),
                "total_trades": total_trades,
                "total_fees": str(total_fees)
            },
            "portfolio": {k: str(v) for k, v in self.portfolio.items()},
            "trade_history": self.trade_history
        }
    
    def run_demo(self):
        """Run the demo"""
        print(f"\n{'='*80}")
        print("LIVE TRADING DEMO")
        print(f"{'='*80}")
        
        print(f"Mode: {self.mode}")
        print(f"Initial Balance: ${self.config['initial_balance']}")
        print(f"Fee Structure:")
        for fee, rate in self.config["fee_structure"].items():
            print(f"  {fee}: {rate*100}%")
        print()
        
        # Execute sample trades
        print("Executing sample trades...")
        
        # Trade 1: Buy BTC
        buy_result = self.execute_trade("BTC-USD", "BUY", Decimal("1000"))
        if buy_result["success"]:
            self.save_state()
            time.sleep(2)
        
        # Trade 2: Buy ETH
        buy_result2 = self.execute_trade("ETH-USD", "BUY", Decimal("500"))
        if buy_result2["success"]:
            self.save_state()
            time.sleep(2)
        
        # Trade 3: Sell BTC
        if "BTC" in self.portfolio:
            btc_amount = self.portfolio["BTC"]
            sell_amount = btc_amount * Decimal("0.5")
            sell_result = self.execute_trade("BTC-USD", "SELL", sell_amount)
            if sell_result["success"]:
                self.save_state()
                time.sleep(2)
        
        # Generate final report
        report = self.generate_report()
        
        print(f"\n{'='*80}")
        print("DEMO COMPLETE")
        print(f"{'='*80}")
        print("✅ Fee structure implemented (Maker 0.4%, Taker 0.6%, Spread 0.25%)")
        print("✅ Live price fetching from Coinbase API")
        print("✅ Complete portfolio tracking")
        print("✅ Trade execution with fee calculations")
        print("✅ State persistence enabled")
        print("✅ Trading report generated")
        print()
        print("FILES CREATED:")
        print("  - demo_trading_state.json (trading state)")
        print("  - trading_demo.log (log file)")
        print()
        print("TO ENABLE LIVE TRADING:")
        print("1. Get API credentials from Coinbase")
        print("2. Set environment variables:")
        print("   export COINBASE_API_KEY='your-key'")
        print("   export COINBASE_API_SECRET='your-secret'")
        print("3. Update the execute_trade method to use real API calls")
        print("4. Run with actual trading permissions")

def main():
    """Main function"""
    print("=== Coinbase Live Trading Demo ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Run demo
    demo = LiveTradingDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()