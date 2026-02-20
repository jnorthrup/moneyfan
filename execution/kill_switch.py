"""
Kill Switch & Metrics Review System
====================================

Monitors live paper trading and automatically stops if limits are breached.

Features:
- Real-time metric monitoring
- Auto-kill switch on limits
- 4-hour metric reports
- Manual kill switch command
- Health checks
"""

import asyncio
import time
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class KillSwitchConfig:
    """Configuration for kill switch"""
    # Hard limits from GOALS.md
    max_single_trade_loss: float = 0.02  # 2%
    max_daily_drawdown: float = 0.05  # 5%
    max_consecutive_losses: int = 3
    pause_duration_minutes: int = 60  # 1hr pause
    
    # Performance thresholds
    min_profit_factor: float = 1.5
    min_sharpe: float = 1.0
    
    # Monitoring intervals
    check_interval_seconds: int = 60  # Check every minute
    report_interval_hours: int = 4  # Report every 4 hours
    
    # Alert thresholds
    warning_drawdown: float = 0.03  # 3% warning
    warning_consecutive_losses: int = 2

class KillSwitch:
    """
    Monitor live trading and stop if limits are breached
    """
    
    def __init__(self, config: KillSwitchConfig, paper_results_dir: str = "paper_results"):
        self.config = config
        self.paper_results_dir = Path(paper_results_dir)
        self.paper_results_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.last_report_time = 0
        self.start_time = time.time()
        self.is_paused = False
        self.pause_until = 0
        
        # Metrics
        self.metrics: Dict[str, Any] = {}
        
        # Signal handling
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigterm)
        
        print(f"[KillSwitch] Initialized")
        print(f"[KillSwitch] Hard limits:")
        print(f"  - Max single trade loss: {config.max_single_trade_loss:.1%}")
        print(f"  - Max daily drawdown: {config.max_daily_drawdown:.1%}")
        print(f"  - Max consecutive losses: {config.max_consecutive_losses}")
        print(f"  - Pause duration: {config.pause_duration_minutes} min")
    
    def handle_sigint(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n" + "="*60)
        print("🎯 KILL SWITCH ACTIVATED (Ctrl+C)")
        print("="*60)
        self.emergency_stop("Manual interrupt")
    
    def handle_sigterm(self, signum, frame):
        """Handle termination signal"""
        print("\n" + "="*60)
        print("🎯 KILL SWITCH ACTIVATED (SIGTERM)")
        print("="*60)
        self.emergency_stop("Termination signal")
    
    def emergency_stop(self, reason: str = "Unknown"):
        """Emergency stop trading"""
        print(f"[KillSwitch] EMERGENCY STOP - {reason}")
        print(f"[KillSwitch] Writing final metrics...")
        
        # Save final state
        self.save_metrics(reason=reason)
        
        print(f"[KillSwitch] System stopped.")
        sys.exit(1)
    
    def check_metrics(self) -> Optional[str]:
        """Check current metrics against limits"""
        if not self.metrics:
            return None
        
        # Check for pause
        if self.is_paused:
            if time.time() > self.pause_until:
                print(f"[KillSwitch] Pause ended, resuming trading")
                self.is_paused = False
                self.consecutive_losses = 0
            else:
                remaining = int(self.pause_until - time.time())
                return f"PAUSED - {remaining} seconds remaining"
        
        # Get current metrics
        equity = self.metrics.get("final_equity", 1000.0)
        peak = self.metrics.get("peak_equity", 1000.0)
        trough = self.metrics.get("trough_equity", 1000.0)
        
        # Calculate drawdown
        if peak > 0:
            self.current_drawdown = (peak - equity) / peak
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check max daily drawdown (5% limit)
        if self.current_drawdown >= self.config.max_daily_drawdown:
            self.emergency_stop(f"Daily drawdown exceeded {self.config.max_daily_drawdown:.1%} "
                              f"(Current: {self.current_drawdown:.1%})")
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.emergency_stop(f"Consecutive losses exceeded {self.config.max_consecutive_losses} "
                              f"(Current: {self.consecutive_losses})")
        
        # Check profit factor
        profit_factor = self.metrics.get("profit_factor", 0)
        if profit_factor > 0 and profit_factor < self.config.min_profit_factor:
            return f"Profit factor below threshold: {profit_factor:.2f} < {self.config.min_profit_factor}"
        
        # Check Sharpe ratio
        sharpe = self.metrics.get("sharpe", 0)
        if sharpe < self.config.min_sharpe:
            return f"Sharpe below threshold: {sharpe:.2f} < {self.config.min_sharpe}"
        
        # Check warning levels
        if self.current_drawdown >= self.config.warning_drawdown:
            return f"⚠️  WARNING: Drawdown approaching limit: {self.current_drawdown:.1%}"
        
        if self.consecutive_losses >= self.config.warning_consecutive_losses:
            return f"⚠️  WARNING: Consecutive losses approaching limit: {self.consecutive_losses}"
        
        return None
    
    def update_from_trade(self, trade_result: Dict[str, Any]):
        """Update metrics from trade execution"""
        if "error" in trade_result:
            # Trade failed
            self.consecutive_losses += 1
            print(f"[KillSwitch] Trade failed, consecutive losses: {self.consecutive_losses}")
            return
        
        # Check for loss
        pnl = trade_result.get("pnl", 0.0)
        if pnl < 0:
            loss_pct = abs(pnl) / trade_result.get("size_usd", 1.0)
            if loss_pct >= self.config.max_single_trade_loss:
                self.emergency_stop(f"Single trade loss exceeded {self.config.max_single_trade_loss:.1%} "
                                  f"(Loss: {loss_pct:.1%})")
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on win
    
    def update_from_metrics_file(self, metrics_path: Optional[Path] = None):
        """Update metrics from paper_results/metrics.json"""
        if metrics_path is None:
            metrics_path = self.paper_results_dir / "metrics.json"
        
        if not metrics_path.exists():
            return
        
        try:
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
        except Exception as e:
            print(f"[KillSwitch] Error reading metrics: {e}")
    
    def generate_4h_report(self) -> str:
        """Generate 4-hour report"""
        uptime = time.time() - self.start_time
        hours = uptime / 3600
        
        report = []
        report.append("="*60)
        report.append(f"MONEYFAN PAPER TRADING - 4-HOUR REPORT")
        report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Uptime: {hours:.1f} hours")
        report.append("="*60)
        report.append("")
        
        # Performance metrics
        if self.metrics:
            report.append("PERFORMANCE METRICS:")
            report.append(f"  Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
            report.append(f"  Sharpe Ratio: {self.metrics.get('sharpe', 0):.2f}")
            report.append(f"  Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}")
            report.append(f"  Win Rate: {self.metrics.get('win_rate', 0):.1f}%")
            report.append(f"  Total Trades: {self.metrics.get('total_trades', 0)}")
            report.append(f"  Equity: ${self.metrics.get('final_equity', 0):.2f}")
            report.append("")
        
        # Status
        status = self.check_metrics()
        if status:
            report.append("STATUS: " + status)
            report.append("")
        
        # Hard limits
        report.append("HARD LIMITS:")
        report.append(f"  Max Single Trade Loss: {self.config.max_single_trade_loss:.1%}")
        report.append(f"  Max Daily Drawdown: {self.config.max_daily_drawdown:.1%}")
        report.append(f"  Max Consecutive Losses: {self.config.max_consecutive_losses}")
        report.append(f"  Current Drawdown: {self.current_drawdown:.1%}")
        report.append(f"  Consecutive Losses: {self.consecutive_losses}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if self.current_drawdown > 0.03:
            report.append("  ⚠️  Consider reducing position size")
        if self.consecutive_losses >= 2:
            report.append("  ⚠️  Consider pausing trading")
        if self.metrics.get("profit_factor", 0) < 1.5:
            report.append("  ⚠️  Profit factor below target, consider adjusting strategy")
        report.append("")
        
        # Kill switch status
        report.append("KILL SWITCH STATUS: 🟢 ACTIVE")
        report.append("Press Ctrl+C to trigger manual kill switch")
        
        return "\n".join(report)
    
    def save_metrics(self, reason: str = "Normal"):
        """Save current metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = self.paper_results_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save report
        report_file = self.paper_results_dir / f"report_{timestamp}.txt"
        report = self.generate_4h_report()
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"[KillSwitch] Metrics saved to {metrics_file}")
        print(f"[KillSwitch] Report saved to {report_file}")
    
    async def monitor_loop(self, check_interval: int = None):
        """Main monitoring loop"""
        if check_interval is None:
            check_interval = self.config.check_interval_seconds
        
        print(f"[KillSwitch] Starting monitoring loop (check every {check_interval}s)")
        
        while True:
            try:
                # Update metrics from file
                self.update_from_metrics_file()
                
                # Check metrics
                status = self.check_metrics()
                
                # Generate 4-hour report
                current_time = time.time()
                if current_time - self.last_report_time >= self.config.report_interval_hours * 3600:
                    self.last_report_time = current_time
                    report = self.generate_4h_report()
                    print("\n" + report)
                    
                    # Save report
                    self.save_metrics()
                
                # Print status if problematic
                if status:
                    print(f"[KillSwitch] {status}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"[KillSwitch] Monitoring error: {e}")
                await asyncio.sleep(check_interval)

# Example usage
async def main():
    config = KillSwitchConfig()
    kill_switch = KillSwitch(config)
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(kill_switch.monitor_loop())
    
    # Simulate trading loop
    print("\nSimulating trading session...")
    print("Press Ctrl+C to stop (or wait for hard limits)")
    print("")
    
    # Simulate trades
    for i in range(20):
        trade_result = {
            "trade_id": i,
            "pnl": 10.0 if i % 2 == 0 else -5.0,  # Mix of wins and losses
            "size_usd": 100.0
        }
        kill_switch.update_from_trade(trade_result)
        await asyncio.sleep(2)  # Simulate 2-second intervals
    
    # Wait for monitoring
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())