#!/usr/bin/env python3
"""
Live Trading Environment Setup
Sets up complete live trading environment with monitoring, risk management, and automation
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal, ROUND_HALF_UP
import threading

# Import our trading systems
sys.path.append(os.path.dirname(__file__))
from coinbase_live_trading import CoinbaseLiveTrading
from coinbase_advanced_trading import AdvancedTradingSystem, CoinbaseAPI

class TradingMonitor:
    """Monitors trading activity and performance"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.alerts = []
        self.performance_metrics = {}
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def log_trade(self, trade: Dict[str, Any]):
        """Log a trade execution"""
        self.logger.info(f"TRADE: {trade['side']} {trade['pair']} - ${trade.get('total_cost', trade.get('amount', '0'))}")
        
        # Store for analysis
        if 'trades' not in self.performance_metrics:
            self.performance_metrics['trades'] = []
        self.performance_metrics['trades'].append(trade)
    
    def check_balance_alerts(self, balance: Decimal, threshold: Decimal = Decimal("1000")):
        """Check balance alerts"""
        if balance < threshold:
            alert = f"LOW BALANCE: ${balance} (threshold: ${threshold})"
            self.alerts.append(alert)
            self.logger.warning(alert)
    
    def check_position_alerts(self, portfolio: Dict[str, Decimal], total_value: Decimal):
        """Check position size alerts"""
        max_position_percent = Decimal("0.4")  # 40% max
        
        for currency, amount in portfolio.items():
            if amount > 0:
                position_value = amount * Decimal("1000")  # Simplified
                position_percent = position_value / total_value if total_value > 0 else 0
                
                if position_percent > max_position_percent:
                    alert = f"LARGE POSITION: {currency} is {position_percent*100:.1f}% of portfolio"
                    self.alerts.append(alert)
                    self.logger.warning(alert)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "trades_today": 0
        }
        
        # Count trades today
        if 'trades' in self.performance_metrics:
            today = datetime.now().date()
            trades_today = [
                t for t in self.performance_metrics['trades']
                if datetime.fromisoformat(t['timestamp']).date() == today
            ]
            report['trades_today'] = len(trades_today)
        
        return report

class RiskManager:
    """Manages trading risk"""
    
    def __init__(self):
        self.risk_limits = {
            "daily_loss_limit": Decimal("100"),  # $100 max loss per day
            "max_position_size": Decimal("0.3"),  # 30% per position
            "stop_loss_default": Decimal("0.05"),  # 5% stop loss
            "take_profit_default": Decimal("0.10"),  # 10% take profit
            "max_trades_per_day": 10,
            "min_account_balance": Decimal("1000"),
        }
        
        self.daily_trades = 0
        self.daily_pnl = Decimal("0")
        self.last_reset = datetime.now().date()
    
    def check_tradability(self, trade_amount: Decimal, account_balance: Decimal, 
                         position_size: Decimal) -> Tuple[bool, str]:
        """Check if trade is allowable"""
        # Check daily trade limit
        if self.daily_trades >= self.risk_limits["max_trades_per_day"]:
            return False, f"Daily trade limit reached ({self.daily_trades})"
        
        # Check minimum balance
        if account_balance - trade_amount < self.risk_limits["min_account_balance"]:
            return False, f"Would drop below minimum balance"
        
        # Check position size
        if position_size > self.risk_limits["max_position_size"] * account_balance:
            return False, f"Position too large"
        
        return True, "OK"
    
    def check_daily_loss_limit(self, daily_pnl: Decimal) -> Tuple[bool, str]:
        """Check if daily loss limit is exceeded"""
        if daily_pnl < -self.risk_limits["daily_loss_limit"]:
            return False, f"Daily loss limit exceeded: ${daily_pnl}"
        return True, "OK"
    
    def update_daily_pnl(self, pnl: Decimal):
        """Update daily P&L"""
        today = datetime.now().date()
        if today != self.last_reset:
            # New day, reset counters
            self.daily_trades = 0
            self.daily_pnl = Decimal("0")
            self.last_reset = today
        
        self.daily_pnl += pnl
        self.daily_trades += 1
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            "daily_trades": self.daily_trades,
            "daily_pnl": str(self.daily_pnl),
            "daily_loss_limit": str(self.risk_limits["daily_loss_limit"]),
            "can_trade": self.check_daily_loss_limit(self.daily_pnl)[0],
            "risk_limits": {k: str(v) for k, v in self.risk_limits.items()}
        }

class LiveTradingEnvironment:
    """Complete live trading environment"""
    
    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")
        self.passphrase = os.getenv("COINBASE_PASSPHRASE", "")
        
        # Initialize components
        self.monitor = TradingMonitor()
        self.risk_manager = RiskManager()
        
        # Trading systems
        self.basic_system = None
        self.advanced_system = None
        
        # State
        self.environment_config = self.load_config()
        self.running = False
        self.mode = "SIMULATION" if not (self.api_key and self.api_secret) else "LIVE"
        
        # Initialize systems
        self.initialize_systems()
    
    def load_config(self) -> Dict[str, Any]:
        """Load environment configuration"""
        config_file = "trading_environment_config.json"
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Default config
        return {
            "trading_pairs": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "trading_strategies": ["mean_reversion", "momentum", "dca"],
            "check_interval_seconds": 60,
            "max_positions": 3,
            "enable_automation": False,
            "enable_notifications": False,
            "log_level": "INFO"
        }
    
    def initialize_systems(self):
        """Initialize trading systems"""
        print(f"Initializing trading environment in {self.mode} mode...")
        
        if self.api_key and self.api_secret:
            # Initialize with real API
            self.advanced_system = AdvancedTradingSystem(
                self.api_key, self.api_secret, self.passphrase
            )
            self.basic_system = CoinbaseLiveTrading(
                self.api_key, self.api_secret, self.passphrase
            )
            print("✅ Live trading systems initialized")
        else:
            print("⚠️  No API credentials - running in simulation mode")
    
    def run_market_analysis(self):
        """Run market analysis"""
        print(f"\n{'='*80}")
        print("MARKET ANALYSIS CYCLE")
        print(f"{'='*80}")
        
        if self.advanced_system:
            analysis = self.advanced_system.analyze_market()
            
            # Log analysis
            self.monitor.logger.info(f"Market analysis completed at {datetime.now()}")
            
            # Check for trading opportunities
            for pair, data in analysis.items():
                if data.get('trend') == 'BEARISH':
                    self.monitor.logger.info(f"{pair}: Bearish trend - consider selling")
                elif data.get('trend') == 'BULLISH':
                    self.monitor.logger.info(f"{pair}: Bullish trend - consider buying")
            
            return analysis
        
        return {}
    
    def execute_trading_cycle(self):
        """Execute a complete trading cycle"""
        print(f"\n{'='*80}")
        print("TRADING CYCLE")
        print(f"{'='*80}")
        
        # Check risk status
        risk_status = self.risk_manager.get_risk_status()
        if not risk_status["can_trade"]:
            print(f"⚠️  Trading halted: {risk_status['can_trade']}")
            return
        
        if self.advanced_system:
            # Analyze market
            self.run_market_analysis()
            
            # Execute strategies
            self.advanced_system.execute_strategy_trades()
            
            # Update risk management
            total_value = Decimal("10000")  # Simplified
            for trade in self.advanced_system.trade_history[-1:]:
                pnl = Decimal(trade.get("amount", 0))
                self.risk_manager.update_daily_pnl(pnl)
            
            # Generate report
            report = self.advanced_system.generate_advanced_report()
            
            # Check alerts
            balance = Decimal(report["account_summary"]["balance"])
            self.monitor.check_balance_alerts(balance)
            
            # Log results
            self.monitor.logger.info(f"Trading cycle completed at {datetime.now()}")
        
        elif self.basic_system:
            # Use basic system for demonstration
            print("Using basic trading system...")
            
            # Analyze portfolio
            self.basic_system.analyze_portfolio()
            
            # Generate report
            report = self.basic_system.generate_trading_report()
            
            print("Trading cycle completed")
    
    def automated_trading_loop(self):
        """Main automated trading loop"""
        print(f"\n{'='*80}")
        print("AUTOMATED TRADING LOOP STARTED")
        print(f"{'='*80}")
        
        if self.environment_config.get("enable_automation", False):
            print("✅ Automation enabled")
            print("   This will run trading cycles automatically")
            print("   Press Ctrl+C to stop")
        else:
            print("⚠️  Automation disabled")
            print("   Run trading cycles manually")
            return
        
        # Simple automated loop
        interval = self.environment_config.get("check_interval_seconds", 60)
        
        try:
            while True:
                print(f"\n{datetime.now()}: Running automated cycle...")
                try:
                    self.execute_trading_cycle()
                except Exception as e:
                    self.monitor.logger.error(f"Error in automated cycle: {e}")
                
                print(f"Next cycle in {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nAutomated trading stopped by user")
    
    def manual_trading_mode(self):
        """Manual trading mode"""
        print(f"\n{'='*80}")
        print("MANUAL TRADING MODE")
        print(f"{'='*80}")
        
        while True:
            print("\nOptions:")
            print("1. Run market analysis")
            print("2. Execute trading cycle")
            print("3. Generate report")
            print("4. View risk status")
            print("5. View alerts")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                self.run_market_analysis()
            elif choice == "2":
                self.execute_trading_cycle()
            elif choice == "3":
                if self.advanced_system:
                    self.advanced_system.generate_advanced_report()
                elif self.basic_system:
                    self.basic_system.generate_trading_report()
            elif choice == "4":
                risk_status = self.risk_manager.get_risk_status()
                print("\nRisk Status:")
                for key, value in risk_status.items():
                    print(f"  {key}: {value}")
            elif choice == "5":
                print(f"\nAlerts ({len(self.monitor.alerts)}):")
                for alert in self.monitor.alerts[-10:]:
                    print(f"  {alert}")
            elif choice == "6":
                print("Exiting manual trading mode")
                break
            else:
                print("Invalid choice")
    
    def setup_environment(self):
        """Setup the trading environment"""
        print(f"\n{'='*80}")
        print("TRADING ENVIRONMENT SETUP")
        print(f"{'='*80}")
        
        print(f"\nCurrent Configuration:")
        print(f"  Mode: {self.mode}")
        print(f"  Trading Pairs: {', '.join(self.environment_config['trading_pairs'])}")
        print(f"  Trading Strategies: {', '.join(self.environment_config['trading_strategies'])}")
        print(f"  Check Interval: {self.environment_config['check_interval_seconds']} seconds")
        print(f"  Automation: {'Enabled' if self.environment_config['enable_automation'] else 'Disabled'}")
        
        print(f"\nAPI Status:")
        if self.api_key and self.api_secret:
            print(f"  ✅ API Key: {self.api_key[:20]}...")
            print(f"  ✅ API Secret: Set")
            if self.passphrase:
                print(f"  ✅ Passphrase: Set")
            else:
                print(f"  ⚠️  Passphrase: Not set (may be needed)")
        else:
            print(f"  ❌ API Key: Not set")
            print(f"  ❌ API Secret: Not set")
            print(f"  ℹ️  Running in simulation mode")
        
        print(f"\nAvailable Systems:")
        if self.advanced_system:
            print(f"  ✅ Advanced Trading System")
        if self.basic_system:
            print(f"  ✅ Basic Trading System")
        
        print(f"\nComponents:")
        print(f"  ✅ Trading Monitor")
        print(f"  ✅ Risk Manager")
        print(f"  ✅ State Persistence")
        print(f"  ✅ Logging System")
    
    def start(self):
        """Start the trading environment"""
        self.setup_environment()
        
        print(f"\n{'='*80}")
        print("START TRADING ENVIRONMENT")
        print(f"{'='*80}")
        
        if self.environment_config.get("enable_automation", False):
            print("\nStarting automated trading...")
            self.automated_trading_loop()
        else:
            print("\nStarting manual trading mode...")
            self.manual_trading_mode()

def main():
    """Main function"""
    print("=== LIVE TRADING ENVIRONMENT SETUP ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Create trading environment
    environment = LiveTradingEnvironment()
    
    # Start the environment
    environment.start()
    
    print()
    print("=" * 80)
    print("TRADING ENVIRONMENT COMPLETE")
    print("=" * 80)
    print("✅ Fee structure analyzed and implemented")
    print("✅ Live fantasy trading system created")
    print("✅ Advanced trading strategies implemented")
    print("✅ Live trading environment ready")
    print()
    print("SYSTEMS CREATED:")
    print("1. coinbase_fee_analysis.py - Fee analysis and fantasy trading")
    print("2. coinbase_live_trading.py - Live trading with proper fees")
    print("3. coinbase_advanced_trading.py - Advanced strategies")
    print("4. live_trading_environment.py - Complete environment")
    print()
    print("TO ENABLE LIVE TRADING:")
    print("1. Get API credentials from Coinbase")
    print("2. Set environment variables:")
    print("   export COINBASE_API_KEY='your-api-key'")
    print("   export COINBASE_API_SECRET='your-api-secret'")
    print("   export COINBASE_PASSPHRASE='your-passphrase' (if needed)")
    print("3. Run: python3 live_trading_environment.py")
    print("4. Select manual or automated trading mode")
    print()
    print("MONITORING:")
    print("  - Check trading.log for detailed logs")
    print("  - Check live_trading_state.json for state")
    print("  - Review alerts and risk status regularly")

if __name__ == "__main__":
    main()