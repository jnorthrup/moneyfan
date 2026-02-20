"""
Signal Writer Module
====================

Writes MLX signals to stdout for Kotlin to read via stdin/stdout bridge.
Format: JSON lines (one JSON object per line)
"""

import json
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class SignalWriter:
    """
    Write trading signals to stdout for Kotlin consumption
    """
    
    @staticmethod
    def write_signal(signal: Dict[str, Any]) -> None:
        """
        Write single signal to stdout as JSON line
        
        Args:
            signal: Dictionary with signal data:
                - 'timestamp': ISO 8601 timestamp
                - 'symbol': Trading pair (e.g., 'BTC-USD')
                - 'signal_strength': float in [-1, 1] (negative = sell, positive = buy)
                - 'confidence': float in [0, 1] (signal confidence)
                - 'position_size': float (position size in base currency)
                - 'stop_loss': float (stop loss price)
                - 'take_profit': float (take profit price)
                - 'regime': str (bullish/bearish/sideways)
                - 'codec_id': int (codec identifier)
                - 'weight': float (Dirichlet weight for portfolio allocation)
        """
        # Ensure timestamp
        if 'timestamp' not in signal:
            signal['timestamp'] = datetime.now().isoformat()
        
        # Ensure required fields
        required_fields = ['symbol', 'signal_strength', 'confidence']
        for field in required_fields:
            if field not in signal:
                signal[field] = 0.0
        
        # Convert numpy types to Python types
        for key, value in signal.items():
            if isinstance(value, (np.float32, np.float64)):
                signal[key] = float(value)
            elif isinstance(value, np.int32, np.int64):
                signal[key] = int(value)
        
        # Write JSON line
        json_line = json.dumps(signal, ensure_ascii=False)
        print(json_line, flush=True)
        sys.stdout.flush()
    
    @staticmethod
    def write_signals_batch(signals: List[Dict[str, Any]]) -> None:
        """
        Write batch of signals to stdout
        
        Args:
            signals: List of signal dictionaries
        """
        for signal in signals:
            SignalWriter.write_signal(signal)
    
    @staticmethod
    def write_heartbeat() -> None:
        """
        Write heartbeat signal to stdout (for liveness monitoring)
        """
        heartbeat = {
            'timestamp': datetime.now().isoformat(),
            'type': 'HEARTBEAT',
            'message': 'Python MLX system running'
        }
        print(json.dumps(heartbeat), flush=True)
        sys.stdout.flush()
    
    @staticmethod
    def write_error(error_msg: str, context: Dict[str, Any] = None) -> None:
        """
        Write error message to stderr
        
        Args:
            error_msg: Error message
            context: Additional context
        """
        error_dict = {
            'timestamp': datetime.now().isoformat(),
            'type': 'ERROR',
            'message': error_msg
        }
        
        if context:
            error_dict['context'] = context
        
        # Write to stderr (doesn't interfere with stdout pipe)
        print(json.dumps(error_dict), file=sys.stderr, flush=True)
        sys.stderr.flush()
    
    @staticmethod
    def write_log(level: str, message: str, data: Dict[str, Any] = None) -> None:
        """
        Write log message to stderr
        
        Args:
            level: Log level (INFO, WARN, ERROR)
            message: Log message
            data: Additional data
        """
        log_dict = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        if data:
            log_dict['data'] = data
        
        print(json.dumps(log_dict), file=sys.stderr, flush=True)
        sys.stderr.flush()





class SignalBatchWriter:
    """
    Batch signal writer with batching and throttling
    """
    
    def __init__(self, batch_size: int = 10, throttle_ms: float = 100):
        """
        Initialize batch writer
        
        Args:
            batch_size: Number of signals to batch
            throttle_ms: Minimum milliseconds between batches
        """
        self.batch_size = batch_size
        self.throttle_ms = throttle_ms
        self.buffer = []
        self.last_flush = 0
        
    def add_signal(self, signal: Dict[str, Any]) -> None:
        """
        Add signal to buffer
        
        Args:
            signal: Signal dictionary
        """
        self.buffer.append(signal)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffer to stdout"""
        if not self.buffer:
            return
        
        # Throttle if needed
        current_time = time.time() * 1000
        if current_time - self.last_flush < self.throttle_ms:
            time.sleep((self.throttle_ms - (current_time - self.last_flush)) / 1000)
        
        # Write all signals
        SignalWriter.write_signals_batch(self.buffer)
        
        # Clear buffer and update timestamp
        self.buffer.clear()
        self.last_flush = time.time() * 1000
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


# Utility functions
def create_signal(symbol: str, 
                 signal_strength: float, 
                 confidence: float,
                 position_size: float = 0.0,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 regime: str = "neutral",
                 codec_id: int = 0,
                 weight: float = 1.0) -> Dict[str, Any]:
    """
    Helper function to create a signal dictionary
    
    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        signal_strength: Signal strength in [-1, 1]
        confidence: Confidence in [0, 1]
        position_size: Position size in base currency
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        regime: Market regime
        codec_id: Codec identifier
        weight: Portfolio weight
        
    Returns:
        Signal dictionary
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'signal_strength': signal_strength,
        'confidence': confidence,
        'position_size': position_size,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'regime': regime,
        'codec_id': codec_id,
        'weight': weight,
    }


def create_error_signal(error_msg: str, symbol: str = "UNKNOWN") -> Dict[str, Any]:
    """
    Create error signal for Kotlin
    
    Args:
        error_msg: Error message
        symbol: Symbol that caused error
        
    Returns:
        Error signal dictionary
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'signal_strength': 0.0,
        'confidence': 0.0,
        'position_size': 0.0,
        'regime': 'ERROR',
        'error': error_msg,
    }