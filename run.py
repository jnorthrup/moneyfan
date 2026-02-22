#!/usr/bin/env python3
"""
Run Live/Paper Trading
======================

Execute HRM model in live or paper trading mode.

Usage:
    python run.py --mode paper --capital 500
    python run.py --mode live --capital 1000 --broker coinbase
"""

import sys
import os
import argparse
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        HierarchicalCodecConfig
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from train import CandlePipeline, CandleCache
from hrm.order_intent import NormalizedTradeIntent, RiskTier
from execution.order_intent_adapter import (
    intent_to_coinbase_order_preview,
    intent_to_legacy_signal,
)


@dataclass
class TradingConfig:
    mode: str = "paper"
    capital: float = 100.0
    broker: str = "coinbase"
    symbols: List[str] = None
    risk_per_trade: float = 0.01
    max_positions: int = 10
    stop_loss: float = 0.05
    take_profit: float = 0.10
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
                'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD'
            ]


class TradingEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.cache = CandleCache()
        self.pipeline = CandlePipeline(self.cache)
        
        self.model = None
        self.model_config = None
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        self.trades = []
        self.running = False
        
        self._load_model()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_model(self):
        if not HAS_MLX:
            print("❌ MLX not available")
            return
        
        model_path = Path('training_results.json')
        if not model_path.exists():
            print("⚠️  No trained model found. Run training first: python train.py")
            return
        
        self.model_config = HierarchicalCodecConfig(
            n_codec_outputs=24,
            hidden_dim=64,
            ob_depth_frames=20,
            ob_lookback_horizon=200
        )
        
        self.model = MLXHierarchicalCodec(self.model_config)
        print("✅ Model loaded")
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Shutting down...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    def generate_signals(self, symbol: str) -> Dict:
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=7)
        
        df = self.pipeline.load_candles(
            [symbol],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty or len(df) < 64:
            return {
                'symbol': symbol,
                'signal': 0,
                'confidence': 0.0,
                'error': 'Insufficient data'
            }
        
        signals = self.pipeline.compute_signals(df, self.model_config.n_signals)
        
        seq_len = min(256, len(signals))
        batch = signals[-seq_len:].reshape(1, seq_len, -1)
        
        try:
            batch_mx = mx.array(batch)
            output, _ = self.model(batch_mx, mode='pretrain')
            
            pred = float(output[0, 0])
            confidence = float(np.abs(pred))
            signal = np.sign(pred) if confidence > 0.3 else 0
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'prediction': pred
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'signal': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def execute_trade(self, signal: Dict):
        symbol = signal['symbol']
        direction = signal['signal']
        confidence = signal['confidence']
        
        if direction == 0 or confidence < 0.3:
            return None
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            if np.sign(pos['direction']) != direction:
                self._close_position(symbol)
            else:
                return None
        
        if len(self.positions) >= self.config.max_positions:
            return None

        stop_loss = float(signal.get('stop_loss_pct', self.config.stop_loss))
        take_profit = float(signal.get('take_profit_pct', self.config.take_profit))
        position_fraction = float(signal.get('position_fraction', 1.0))
        position_fraction = min(1.0, max(0.0, position_fraction))

        base_position_size = self.config.capital * self.config.risk_per_trade / max(self.config.stop_loss, 1e-6)
        position_size = base_position_size * position_fraction
        
        position = {
            'symbol': symbol,
            'direction': direction,
            'size': position_size,
            'entry_price': self._get_current_price(symbol),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.positions[symbol] = position
        
        if self.config.mode == 'paper':
            print(f"📝 PAPER: {direction:+.0f} {symbol} @ ${position['entry_price']:.2f} "
                  f"(size: ${position_size:.2f}, conf: {confidence:.2f})")
        else:
            print(f"🔴 LIVE: {direction:+.0f} {symbol} @ ${position['entry_price']:.2f}")
        
        return position

    def _signal_to_intent(self, signal: Dict) -> Optional[NormalizedTradeIntent]:
        if 'error' in signal:
            return None

        direction = float(signal.get('signal', 0.0))
        confidence = float(signal.get('confidence', 0.0))
        pred = float(signal.get('prediction', 0.0))

        return NormalizedTradeIntent(
            symbol=signal['symbol'],
            direction=direction,
            pred_fwd_return=pred,
            confidence=confidence,
            position_fraction=min(1.0, max(0.0, confidence)),
            stop_loss_pct=-abs(self.config.stop_loss),
            take_profit_pct=abs(self.config.take_profit),
            risk_tier=RiskTier.NORMAL,
        )

    def execute_trade_intent(self, intent: NormalizedTradeIntent):
        if intent.vetoed:
            return None

        # Record a broker-agnostic preview for observability / future adapters.
        self.orders.append(intent_to_coinbase_order_preview(intent))
        legacy_signal = intent_to_legacy_signal(intent)
        return self.execute_trade(legacy_signal)
    
    def _get_current_price(self, symbol: str) -> float:
        return np.random.uniform(100, 1000)
    
    def _close_position(self, symbol: str):
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        exit_price = self._get_current_price(symbol)
        
        pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['size']
        if pos['direction'] < 0:
            pnl = -pnl
        
        trade = {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        self.pnl += pnl
        del self.positions[symbol]
        
        print(f"💰 Closed {symbol}: PnL ${pnl:.2f} (Total: ${self.pnl:.2f})")
    
    def update_positions(self):
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self._get_current_price(symbol)
            
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            if pos['direction'] < 0:
                pnl_pct = -pnl_pct
            
            if pnl_pct <= -pos['stop_loss']:
                print(f"🛑 Stop loss hit: {symbol}")
                self._close_position(symbol)
            elif pnl_pct >= pos['take_profit']:
                print(f"🎯 Take profit hit: {symbol}")
                self._close_position(symbol)
    
    def run(self):
        if not self.model:
            print("❌ Cannot start: model not loaded")
            return
        
        print(f"\n🚀 Starting {self.config.mode.upper()} trading")
        print(f"💰 Capital: ${self.config.capital:.2f}")
        print(f"📊 Symbols: {', '.join(self.config.symbols[:5])}...")
        print(f"⚡ Max positions: {self.config.max_positions}")
        print("\nPress Ctrl+C to stop\n")
        
        self.running = True
        iteration = 0
        
        while self.running:
            iteration += 1
            print(f"\n--- Iteration {iteration} ({datetime.now().strftime('%H:%M:%S')}) ---")
            
            for symbol in self.config.symbols:
                if not self.running:
                    break
                
                signal = self.generate_signals(symbol)
                intent = self._signal_to_intent(signal)
                if intent is None:
                    continue
                self.execute_trade_intent(intent)
            
            self.update_positions()
            self._save_state()
            
            print(f"\n📊 Status: {len(self.positions)} positions, PnL: ${self.pnl:.2f}")
            
            time.sleep(60)
    
    def _save_state(self):
        state = {
            'mode': self.config.mode,
            'capital': self.config.capital,
            'pnl': self.pnl,
            'positions': self.positions,
            'trades': self.trades[-100:],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('trading_state.json', 'w') as f:
            json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run HRM Trading')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help='Trading mode (paper or live)')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Starting capital')
    parser.add_argument('--broker', default='coinbase',
                        help='Broker/exchange')
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade (fraction of capital)')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Maximum concurrent positions')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        print("⚠️  WARNING: Live trading mode enabled!")
        print("   Ensure you have proper risk controls in place.")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    config = TradingConfig(
        mode=args.mode,
        capital=args.capital,
        broker=args.broker,
        risk_per_trade=args.risk,
        max_positions=args.max_positions
    )
    
    engine = TradingEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
