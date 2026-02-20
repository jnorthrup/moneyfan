"""
Integrated Pipeline: Full Stack from WS to Execution

Complete stack:
Coinbase WS → PANDAS candles → 30 instruments → {24 tradebots/codecs} → HRM IO → Fiduciary → Execution

Architecture:
1. CoinbaseRealtime: WebSocket feed → PANDAS candles
2. CoinbaseInstruments: Registry of 30+ instruments
3. TradebotRegistry: 24 SOTA strategies from DeFlorio Thesis
4. HRM IO: Signal orchestration and HRM inference
5. FiduciaryOverlay: Risk management and position sizing
6. ExecutionEngine: Order placement and execution
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
import pandas as pd
import numpy as np

# Import all components
try:
    from hrm.coinbase_pipeline import CoinbasePipeline, CoinbaseRealtime, CoinbaseInstruments
    from hrm.tradebots import create_bot_registry, TradeBotRegistry
    from hrm.orchestrator_bridge import OrchestratorBridge
    from hrm.fiduciary_overlay import FiduciaryOverlay, RiskLevel
    from hrm.execution_engine import ExecutionEngine, Order, OrderType
except ImportError:
    from coinbase_pipeline import CoinbasePipeline, CoinbaseRealtime, CoinbaseInstruments
    from tradebots import create_bot_registry, TradeBotRegistry
    from orchestrator_bridge import OrchestratorBridge
    from fiduciary_overlay import FiduciaryOverlay, RiskLevel
    from execution_engine import ExecutionEngine, Order, OrderType


@dataclass
class PipelineState:
    """Current state of the integrated pipeline"""
    timestamp: str
    instruments_loaded: int
    bots_loaded: int
    current_signals: int
    portfolio_value: float
    active_positions: int
    last_trade: Optional[str]
    risk_level: str
    execution_status: str


class IntegratedPipeline:
    """
    Full integration: WS → PANDAS → instruments → 24 tradebots → HRM → Fiduciary → Execution
    """
    
    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        max_instruments: int = 30,
        execution_enabled: bool = False,
    ):
        self.risk_level = risk_level
        self.max_instruments = max_instruments
        self.execution_enabled = execution_enabled
        
        # Component initialization
        print("Initializing Integrated Pipeline...")
        
        # 1. Coinbase Pipeline (WS + History)
        print("  [1/6] Coinbase Pipeline...")
        self.coinbase_pipeline = CoinbasePipeline()
        
        # 2. Instruments (30+ pairs)
        print("  [2/6] Instruments...")
        self.instruments = self.coinbase_pipeline.instruments
        
        # 3. Tradebot Registry (24 strategies)
        print("  [3/6] Tradebot Registry (24 strategies)...")
        self.bot_registry = create_bot_registry()
        print(f"      Loaded {len(self.bot_registry.get_all_states())} tradebots")
        
        # 4. HRM IO / Orchestrator Bridge
        print("  [4/6] HRM IO / Orchestrator Bridge...")
        self.bridge = OrchestratorBridge()
        
        # 5. Fiduciary Overlay
        print("  [5/6] Fiduciary Overlay...")
        self.fiduciary = FiduciaryOverlay(risk_level=risk_level)
        
        # 6. Execution Engine
        print("  [6/6] Execution Engine...")
        self.executor = ExecutionEngine()
        
        # Runtime state
        self.running = False
        self.state_history: List[PipelineState] = []
        self.last_update = None
        
        # Holdings (for gravity filtering)
        self.holdings: Dict[str, float] = {
            'BTC-USD': 50000,
            'ETH-USD': 25000,
            'SOL-USD': 10000,
        }
        self.coinbase_pipeline.update_holdings(self.holdings)
        
        print(f"Pipeline initialized with {self.max_instruments} instruments")
    
    def start(self, interval: int = 60):
        """Start the integrated pipeline"""
        print(f"\nStarting pipeline with {interval}s update interval...")
        self.running = True
        
        # Start in background thread
        thread = threading.Thread(
            target=self._run_loop,
            args=(interval,),
            daemon=True,
        )
        thread.start()
        
        return thread
    
    def _run_loop(self, interval: int):
        """Main pipeline loop"""
        update_count = 0
        
        while self.running:
            try:
                update_count += 1
                print(f"\n{'='*70}")
                print(f"PIPELINE UPDATE #{update_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                print(f"{'='*70}")
                
                # Run full pipeline
                self.run_pipeline()
                
                # Sleep until next update
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Pipeline error: {e}")
                time.sleep(5)  # Back off on error
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute one complete pipeline cycle"""
        
        # 1. Get active instruments (filtered by holdings)
        symbols = self.coinbase_pipeline.get_active_instruments(self.max_instruments)
        print(f"\n[1] Active Instruments ({len(symbols)}): {symbols[:5]}...")
        
        # 2. Get recent data (simulated or from WS)
        data_map = self._get_recent_data(symbols)
        if not data_map:
            print("  No data available, skipping...")
            return {"status": "no_data"}
        
        # 3. Run tradebots on each instrument
        bot_signals = {}
        for symbol in symbols[:10]:  # Limit to 10 for speed
            if symbol in data_map:
                df = data_map[symbol]
                signals = self._run_tradebots(symbol, df)
                if signals:
                    bot_signals[symbol] = signals
        
        print(f"[2] Tradebot Signals: {len(bot_signals)} symbols")
        
        # 4. Compute HRM tensor and inference
        hrm_outputs = {}
        for symbol, signals in list(bot_signals.items())[:5]:  # Limit for speed
            hrm_output = self._compute_hrm(symbol, signals)
            if hrm_output:
                hrm_outputs[symbol] = hrm_output
        
        print(f"[3] HRM Outputs: {len(hrm_outputs)} symbols")
        
        # 5. Apply Fiduciary Overlay
        if not hrm_outputs:
            print("  No HRM outputs, skipping fiduciary...")
            return {"status": "no_hrm_output"}
        
        current_positions = self._get_current_positions()
        portfolio_value = self._get_portfolio_value()
        prices = {s: data_map[s]['close'].iloc[-1] for s in data_map if s in hrm_outputs}
        
        fiduciary_orders = self.fiduciary.apply(
            hrms=hrm_outputs,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            prices=prices,
        )
        
        print(f"[4] Fiduciary Orders: {len(fiduciary_orders)}")
        
        # 6. Execute orders (if enabled)
        execution_results = {}
        if self.execution_enabled:
            execution_results = self._execute_orders(fiduciary_orders)
            print(f"[5] Execution Results: {len(execution_results)}")
        else:
            print("[5] Execution: Disabled (simulation mode)")
        
        # 7. Update state
        self._update_state(bot_signals, hrm_outputs, fiduciary_orders, execution_results)
        
        return {
            "status": "success",
            "instruments": len(symbols),
            "bots": len(bot_signals),
            "hrm": len(hrm_outputs),
            "orders": len(fiduciary_orders),
            "executed": len(execution_results) if self.execution_enabled else 0,
        }
    
    def _get_recent_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get recent candle data for symbols"""
        data_map = {}
        
        for symbol in symbols:
            # Try to get from history
            df = self.coinbase_pipeline.history.load_range(
                symbol,
                datetime.utcnow() - timedelta(hours=24),
                datetime.utcnow(),
            )
            
            if len(df) > 100:
                data_map[symbol] = df
        
        return data_map
    
    def _run_tradebots(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all tradebots on a symbol"""
        signals = {}
        
        # Update instrument data
        from instruments import InstrumentRegistry, LazyInstrument
        registry = InstrumentRegistry()
        registry.register(f"{symbol}_data", LazyInstrument(f"{symbol}_data", lambda: df))
        
        # Run each bot
        for bot_id, bot in self.bot_registry.bots.items():
            if symbol in bot.instruments:
                try:
                    # Compute signal
                    result = bot.compute(registry)
                    if result is not None and not result.empty:
                        # Get latest signal
                        latest = result.iloc[-1]
                        signals[bot_id] = {
                            'signal': latest.get('signal', 0),
                            'confidence': latest.get('confidence', 0.5),
                            'energy': latest.get('energy', 0),
                        }
                except Exception as e:
                    print(f"  Bot {bot_id} error: {e}")
        
        return signals
    
    def _compute_hrm(self, symbol: str, signals: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Compute HRM inference from tradebot signals"""
        try:
            # Convert signals to HRM format
            signal_tensor = self._signals_to_tensor(signals)
            
            if signal_tensor is None:
                return None
            
            # Get HRM inference (simplified)
            # In production, would use actual HRM model
            weights = {}
            total_weight = 0
            
            for bot_id, sig in signals.items():
                # Weight by signal strength and confidence
                weight = sig['signal'] * sig['confidence'] * sig['energy']
                weights[bot_id] = weight
                total_weight += abs(weight)
            
            # Normalize weights
            if total_weight > 0:
                for bot_id in weights:
                    weights[bot_id] /= total_weight
            
            return weights
            
        except Exception as e:
            print(f"  HRM compute error: {e}")
            return None
    
    def _signals_to_tensor(self, signals: Dict[str, Any]):
        """Convert tradebot signals to HRM tensor format"""
        # In production, this would create proper [1, seq_len, 32] tensor
        # For now, return a simplified representation
        return np.zeros((1, 32))
    
    def _get_current_positions(self) -> Dict[str, Dict]:
        """Get current portfolio positions"""
        # In production, would query actual positions
        # For now, return empty
        return {}
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # In production, would calculate from positions + prices
        # For now, return default
        return 100000.0
    
    def _execute_orders(self, fiduciary_orders: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute orders via ExecutionEngine"""
        results = {}
        
        for symbol, order_data in fiduciary_orders.items():
            if symbol.startswith('__'):
                continue  # Skip special commands
            
            # Create Order object
            order = Order(
                symbol=symbol,
                action=order_data['action'],
                size=order_data['size'],
                price=order_data['price'],
                order_type=OrderType.LIMIT,
                stop_loss=order_data.get('stop_loss'),
                take_profit=order_data.get('take_profit'),
                trailing_stop=order_data.get('trailing_stop', False),
                trail_distance=order_data.get('trail_distance'),
            )
            
            # Place order
            order_id = self.executor.place_order(order)
            results[symbol] = {
                'order_id': order_id,
                'status': order.status.value,
                'size': order.size,
            }
        
        return results
    
    def _update_state(self, bot_signals, hrm_outputs, fiduciary_orders, execution_results):
        """Update pipeline state"""
        state = PipelineState(
            timestamp=datetime.utcnow().isoformat(),
            instruments_loaded=len(self.coinbase_pipeline.get_active_instruments(self.max_instruments)),
            bots_loaded=len(self.bot_registry.bots),
            current_signals=len(bot_signals),
            portfolio_value=self._get_portfolio_value(),
            active_positions=len(self._get_current_positions()),
            last_trade=list(execution_results.keys())[0] if execution_results else None,
            risk_level=self.risk_level.value,
            execution_status="enabled" if self.execution_enabled else "disabled",
        )
        
        self.state_history.append(state)
        self.last_update = datetime.utcnow()
        
        # Print state
        print(f"\n[6] Pipeline State:")
        print(f"  Portfolio Value: ${state.portfolio_value:,.2f}")
        print(f"  Active Positions: {state.active_positions}")
        print(f"  Last Trade: {state.last_trade or 'None'}")
        print(f"  Risk Level: {state.risk_level}")
        
        # Print execution stats
        if self.execution_enabled:
            stats = self.executor.get_execution_stats()
            print(f"  Execution Stats: {stats}")
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get dashboard data"""
        if not self.state_history:
            return {"status": "no_data"}
        
        latest = self.state_history[-1]
        
        # Get risk report
        risk_report = self.fiduciary.get_risk_report()
        
        return {
            "timestamp": latest.timestamp,
            "portfolio_value": latest.portfolio_value,
            "active_positions": latest.active_positions,
            "bots_loaded": latest.bots_loaded,
            "instruments": latest.instruments_loaded,
            "risk_level": latest.risk_level,
            "execution_status": latest.execution_status,
            "risk_report": risk_report,
            "last_trade": latest.last_trade,
        }
    
    def stop(self):
        """Stop the pipeline"""
        print("\nStopping pipeline...")
        self.running = False
        self.executor.cancel_all_orders()


if __name__ == "__main__":
    print("=" * 70)
    print("  INTEGRATED PIPELINE")
    print("  Coinbase WS → PANDAS → 30 instruments → 24 tradebots")
    print("  → HRM IO → Fiduciary → Execution")
    print("=" * 70)
    
    # Create pipeline
    pipeline = IntegratedPipeline(
        risk_level=RiskLevel.MODERATE,
        max_instruments=30,
        execution_enabled=False,  # Start in simulation mode
    )
    
    # Run one cycle
    print("\nRunning single pipeline cycle...")
    result = pipeline.run_pipeline()
    
    print(f"\nPipeline Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Get dashboard
    dashboard = pipeline.get_dashboard()
    print(f"\nDashboard:")
    for key, value in dashboard.items():
        if key != 'risk_report':
            print(f"  {key}: {value}")
    
    if 'risk_report' in dashboard:
        print(f"\nRisk Report:")
        for key, value in dashboard['risk_report'].items():
            print(f"  {key}: {value}")
