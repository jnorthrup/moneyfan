"""
Live Executor - Pure Python execution on Coinbase Advanced Trade
===============================================================

Direct execution layer using coinbase-advanced-py SDK.
100% Python, no JS/Kotlin bridges needed.

Features:
- Direct paper trading on Coinbase Advanced Trade
- Real-time execution via official SDK
- Logging for HRM reward calculation
- Vector cache updates from execution results
"""

from pathlib import Path
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from coinbase_advanced_trading import CoinbaseAdvancedTradingClient
    HAS_COINBASE_SDK = True
except ImportError:
    HAS_COINBASE_SDK = False
    print("[LiveExecutor] Coinbase SDK not available, using mock execution")

@dataclass
class LiveExecutorConfig:
    """Configuration for live executor"""
    paper_mode: bool = True
    api_key: str = ""
    api_secret: str = ""
    log_dir: str = "paper_results"
    max_position_size: float = 1000.0  # $1000 max position
    risk_per_trade: float = 0.02  # 2% risk per trade

class LiveExecutor:
    """
    Pure Python execution layer for HRM signals
    
    Direct integration with Coinbase Advanced Trade SDK.
    """
    
    def __init__(self, config: LiveExecutorConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize Coinbase client
        if HAS_COINBASE_SDK:
            try:
                self.client = CoinbaseAdvancedTradingClient(
                    paper=config.paper_mode,
                    api_key=config.api_key,
                    api_secret=config.api_secret
                )
                print(f"[LiveExecutor] Coinbase client initialized (paper_mode={config.paper_mode})")
            except Exception as e:
                print(f"[LiveExecutor] Failed to init Coinbase client: {e}")
                self.client = None
        else:
            self.client = None
            print("[LiveExecutor] Using mock execution (Coinbase SDK not available)")
        
        # Execution history
        self.execution_history = []
        self.last_execution_time = 0
    
    def execute_hrm_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HRM signal on Coinbase Advanced Trade
        
        Args:
            signal: HRM decision signal
                {
                    "timestamp": int,
                    "product_id": str,
                    "direction": "long" | "short" | "flat",
                    "size_usd": float,
                    "stop_loss_pct": float,
                    "take_profit_pct": float,
                    "hrm_weight": float,
                    "convergence_score": float
                }
        
        Returns:
            Execution result
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_execution_time < 1.0:  # 1 second cooldown
            return {"error": "Rate limited", "cooldown": 1.0 - (current_time - self.last_execution_time)}
        self.last_execution_time = current_time
        
        # Validate signal
        if not self._validate_signal(signal):
            return {"error": "Invalid signal"}
        
        # Prepare execution
        product = signal.get("product_id", "BTC-USD")
        direction = signal.get("direction", "flat")
        size_usd = min(signal.get("size_usd", 0), self.config.max_position_size)
        
        if size_usd <= 0:
            return {"error": "Size must be positive"}
        
        # Calculate stop loss and take profit
        stop_loss_pct = signal.get("stop_loss_pct", 0.75)
        take_profit_pct = signal.get("take_profit_pct", 1.50)
        
        # Execute based on direction
        result = {"timestamp": int(time.time()), "signal": signal}
        
        try:
            if direction == "flat":
                # Close all positions
                if HAS_COINBASE_SDK and self.client:
                    close_result = self.client.close_all_positions(product)
                    result["action"] = "CLOSE_ALL"
                    result["result"] = close_result
                else:
                    result["action"] = "CLOSE_ALL_MOCK"
                    result["result"] = {"status": "mock_closed", "product": product}
                
            elif direction == "long":
                # Place market buy
                if HAS_COINBASE_SDK and self.client:
                    order_result = self.client.place_market_buy(
                        product_id=product,
                        usd_size=size_usd,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct
                    )
                    result["action"] = "BUY"
                    result["result"] = order_result
                else:
                    result["action"] = "BUY_MOCK"
                    result["result"] = {
                        "status": "mock_placed",
                        "product": product,
                        "size_usd": size_usd,
                        "stop_loss": stop_loss_pct,
                        "take_profit": take_profit_pct
                    }
            
            elif direction == "short":
                # Place market sell
                if HAS_COINBASE_SDK and self.client:
                    order_result = self.client.place_market_sell(
                        product_id=product,
                        usd_size=size_usd,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct
                    )
                    result["action"] = "SELL"
                    result["result"] = order_result
                else:
                    result["action"] = "SELL_MOCK"
                    result["result"] = {
                        "status": "mock_placed",
                        "product": product,
                        "size_usd": size_usd,
                        "stop_loss": stop_loss_pct,
                        "take_profit": take_profit_pct
                    }
            
            else:
                result["error"] = f"Unknown direction: {direction}"
                return result
            
            # Log execution
            self._log_execution(result)
            
            # Update execution history
            self.execution_history.append(result)
            
            # Print summary
            print(f"[LiveExecutor] {result['action']:12} {size_usd:6.0f} {product} | "
                  f"SL: {stop_loss_pct:.2f} TP: {take_profit_pct:.2f}")
            
            return result
            
        except Exception as e:
            error_result = {
                "timestamp": int(time.time()),
                "signal": signal,
                "error": str(e),
                "action": "ERROR"
            }
            self._log_execution(error_result)
            return error_result
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal before execution"""
        required_fields = ["timestamp", "product_id", "direction", "size_usd"]
        for field in required_fields:
            if field not in signal:
                return False
        
        direction = signal["direction"]
        if direction not in ["long", "short", "flat"]:
            return False
        
        size_usd = signal["size_usd"]
        if not isinstance(size_usd, (int, float)) or size_usd <= 0:
            return False
        
        return True
    
    def _log_execution(self, result: Dict[str, Any]):
        """Log execution to file"""
        log_file = self.log_dir / "live_execution_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"count": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if "error" not in r)
        buys = sum(1 for r in self.execution_history if r.get("action") in ["BUY", "BUY_MOCK"])
        sells = sum(1 for r in self.execution_history if r.get("action") in ["SELL", "SELL_MOCK"])
        closes = sum(1 for r in self.execution_history if r.get("action") in ["CLOSE_ALL", "CLOSE_ALL_MOCK"])
        
        return {
            "total": total,
            "successful": successful,
            "error_rate": (total - successful) / total if total > 0 else 0,
            "buys": buys,
            "sells": sells,
            "closes": closes,
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }

# Example usage
if __name__ == "__main__":
    config = LiveExecutorConfig(paper_mode=True)
    executor = LiveExecutor(config)
    
    # Test execution
    test_signal = {
        "timestamp": int(time.time()),
        "product_id": "BTC-USD",
        "direction": "long",
        "size_usd": 100.0,
        "stop_loss_pct": 0.75,
        "take_profit_pct": 1.50,
        "hrm_weight": 0.8,
        "convergence_score": 0.6
    }
    
    result = executor.execute_hrm_signal(test_signal)
    print(f"\nExecution result: {result}")
    
    stats = executor.get_execution_stats()
    print(f"\nExecution stats: {stats}")