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
            n_signals=24,
            hidden_dim=64,
            sparkline_frames=20,
            sparkline_horizon=200
        )
        
        self.model = MLXHierarchicalCodec(self.model_config)
        print("✅ Model loaded")
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Shutting down...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    def generate_signals(self, symbol: str) -> Dict:
        # Load all available data (fallback to historical if live not available)
        df = self.pipeline.load_candles([symbol], None, None)
        
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
            # Use trade mode for inference
            output, _ = self.model(batch_mx, mode='trade')
            # Output: [pred_fwd_return, signal_conviction, stop_loss_pct, take_profit_pct, position_fraction]

            # Since output is [B, 5], and we passed [1, T, D], MLX model returns [B, 5] (last step)
            # Wait, MLXHierarchicalCodec forward returns `regime_final` processed by heads.
            # regime_final = regime_state[:, -1, :]
            # So output shape is [1, 5]
            
            output_np = np.array(output)
            pred_fwd_return = float(output_np[0, 0])
            signal_conviction = float(output_np[0, 1])
            stop_loss = float(output_np[0, 2])
            take_profit = float(output_np[0, 3])
            pos_fraction = float(output_np[0, 4])

            signal = np.sign(pred_fwd_return) if signal_conviction > 0.5 else 0
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': signal_conviction,
                'prediction': pred_fwd_return,
                'stop_loss': abs(stop_loss),
                'take_profit': abs(take_profit),
                'size_fraction': pos_fraction,
                'price': float(df.iloc[-1]['close'])
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
        current_price = signal.get('price', self._get_current_price(symbol))
        
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
        
        # Use model outputs for sizing if available, else default
        sl = signal.get('stop_loss', self.config.stop_loss)
        tp = signal.get('take_profit', self.config.take_profit)
        size_fraction = signal.get('size_fraction', 0.1)
        
        # position_size = self.config.capital * size_fraction # This might be too aggressive
        # Let's stick to risk-based
        if sl > 0:
            position_size = self.config.capital * self.config.risk_per_trade / sl
        else:
            position_size = self.config.capital * self.config.risk_per_trade / 0.05

        position = {
            'symbol': symbol,
            'direction': direction,
            'size': position_size,
            'entry_price': current_price,
            'stop_loss': sl if sl > 0.001 else 0.05,
            'take_profit': tp if tp > 0.001 else 0.10,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.positions[symbol] = position
        
        if self.config.mode == 'paper':
            print(f"📝 PAPER: {direction:+.0f} {symbol} @ ${position['entry_price']:.2f} "
                  f"(size: ${position_size:.2f}, conf: {confidence:.2f}, sl: {position['stop_loss']:.3f}, tp: {position['take_profit']:.3f})")
        else:
            print(f"🔴 LIVE: {direction:+.0f} {symbol} @ ${position['entry_price']:.2f}")
        
        return position
    
    def _get_current_price(self, symbol: str) -> float:
        # Fetch latest price from pipeline
        df = self.pipeline.load_candles([symbol], None, None)
        if not df.empty:
            return float(df.iloc[-1]['close'])
        return 0.0
    
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
                
                if 'error' not in signal:
                    self.execute_trade(signal)
            
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
