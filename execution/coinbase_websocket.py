"""
Coinbase WebSocket Feed - Real-time market data
===============================================

Replaces simulated ticks with real Coinbase WebSocket data.
100% Python, async implementation for real-time processing.

Features:
- Real-time orderbook updates
- Trade data streaming
- Candle aggregation (configurable intervals)
- Error recovery and reconnection
"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

@dataclass
class WebSocketConfig:
    """Configuration for Coinbase WebSocket"""
    products: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    channels: List[str] = field(default_factory=lambda: ["level2", "ticker", "matches"])
    candle_interval: int = 60  # seconds for candle aggregation
    max_queue_size: int = 1000
    reconnect_delay: int = 5  # seconds between reconnects
    
    # API endpoints
    ws_endpoint: str = "wss://ws-feed.exchange.coinbase.com"
    rest_endpoint: str = "https://api.exchange.coinbase.com"

@dataclass
class TickData:
    """Real-time tick data from Coinbase"""
    timestamp: int
    price: float
    volume: float
    orderbook_imbalance: Optional[float] = None
    product_id: str = "BTC-USD"
    side: str = "buy"  # buy/sell
    size: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None

class CoinbaseWebSocket:
    """
    Real-time Coinbase WebSocket feed
    
    Connects to Coinbase Advanced Trade WebSocket API
    and streams real-time market data.
    """
    
    def __init__(self, config: WebSocketConfig, on_tick: Callable[[TickData], None]):
        self.config = config
        self.on_tick = on_tick  # Callback for each tick
        self.ws = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_message_time = 0
        
        # Market state
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.last_price: float = 0.0
        self.last_volume: float = 0.0
        
        # Stats
        self.total_messages = 0
        self.total_ticks = 0
        self.start_time = None
        
        # Queue for processing
        self.tick_queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        print(f"[CoinbaseWebSocket] Initialized for products: {config.products}")
    
    async def connect(self):
        """Connect to Coinbase WebSocket"""
        print(f"[CoinbaseWebSocket] Connecting to {self.config.ws_endpoint}...")
        
        while True:
            try:
                async with websockets.connect(self.config.ws_endpoint) as ws:
                    self.ws = ws
                    self.is_connected = True
                    self.start_time = time.time()
                    self.reconnect_count = 0
                    
                    print(f"[CoinbaseWebSocket] Connected! Subscribing to channels...")
                    
                    # Subscribe to products and channels
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": self.config.products,
                        "channels": self.config.channels
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # Listen for messages
                    await self._listen()
                    
            except Exception as e:
                print(f"[CoinbaseWebSocket] Connection error: {e}")
                self.is_connected = False
                self.reconnect_count += 1
                
                if self.reconnect_count > 10:
                    print(f"[CoinbaseWebSocket] Too many reconnects, giving up")
                    break
                
                print(f"[CoinbaseWebSocket] Reconnecting in {self.config.reconnect_delay}s...")
                await asyncio.sleep(self.config.reconnect_delay)
    
    async def _listen(self):
        """Listen for WebSocket messages"""
        async for message in self.ws:
            self.total_messages += 1
            self.last_message_time = time.time()
            
            try:
                data = json.loads(message)
                await self._process_message(data)
            except Exception as e:
                print(f"[CoinbaseWebSocket] Message processing error: {e}")
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        msg_type = data.get("type", "")
        
        if msg_type == "error":
            print(f"[CoinbaseWebSocket] Error: {data.get('message', 'Unknown error')}")
        
        elif msg_type == "snapshot":
            # Initial orderbook snapshot
            product_id = data.get("product_id", "BTC-USD")
            self.bids = {float(p): float(s) for p, s in data.get("bids", [])}
            self.asks = {float(p): float(s) for p, s in data.get("asks", [])}
            print(f"[CoinbaseWebSocket] Snapshot received for {product_id}")
        
        elif msg_type == "l2update":
            # Orderbook update
            product_id = data.get("product_id", "BTC-USD")
            for side, price, size in data.get("changes", []):
                price_f = float(price)
                size_f = float(size)
                if side == "buy":
                    if size_f == 0:
                        self.bids.pop(price_f, None)
                    else:
                        self.bids[price_f] = size_f
                else:  # sell
                    if size_f == 0:
                        self.asks.pop(price_f, None)
                    else:
                        self.asks[price_f] = size_f
        
        elif msg_type == "ticker":
            # Ticker update (price, volume, etc.)
            product_id = data.get("product_id", "BTC-USD")
            timestamp = int(float(data.get("time", time.time())) * 1000)
            price = float(data.get("price", 0.0))
            volume = float(data.get("last_size", 0.0))
            side = data.get("side", "buy")
            
            # Calculate orderbook imbalance
            orderbook_imbalance = self._calculate_orderbook_imbalance()
            
            tick = TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                orderbook_imbalance=orderbook_imbalance,
                product_id=product_id,
                side=side,
                size=volume
            )
            
            # Update last price/volume
            self.last_price = price
            self.last_volume = volume
            
            # Queue tick for processing
            try:
                self.tick_queue.put_nowait(tick)
                self.total_ticks += 1
            except asyncio.QueueFull:
                print(f"[CoinbaseWebSocket] Tick queue full, dropping tick")
        
        elif msg_type == "match":
            # Trade match
            product_id = data.get("product_id", "BTC-USD")
            timestamp = int(float(data.get("time", time.time())) * 1000)
            price = float(data.get("price", 0.0))
            size = float(data.get("size", 0.0))
            side = data.get("side", "buy")
            
            tick = TickData(
                timestamp=timestamp,
                price=price,
                volume=size,
                orderbook_imbalance=None,  # Calculate from bids/asks
                product_id=product_id,
                side=side,
                size=size
            )
            
            # Queue tick
            try:
                self.tick_queue.put_nowait(tick)
                self.total_ticks += 1
            except asyncio.QueueFull:
                print(f"[CoinbaseWebSocket] Tick queue full, dropping tick")
    
    def _calculate_orderbook_imbalance(self) -> Optional[float]:
        """Calculate orderbook imbalance (bid/ask ratio)"""
        if not self.bids or not self.asks:
            return None
        
        # Get top of book
        best_bid = max(self.bids.keys()) if self.bids else 0.0
        best_ask = min(self.asks.keys()) if self.asks else 0.0
        
        if best_bid == 0 or best_ask == 0:
            return None
        
        # Calculate imbalance
        bid_volume = self.bids.get(best_bid, 0.0)
        ask_volume = self.asks.get(best_ask, 0.0)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return None
        
        # Return imbalance (0 = balanced, 1 = all bids, 0 = all asks)
        return bid_volume / total_volume
    
    async def get_ticks(self, max_ticks: int = 10) -> List[TickData]:
        """Get ticks from queue"""
        ticks = []
        for _ in range(max_ticks):
            try:
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=0.1)
                ticks.append(tick)
                self.tick_queue.task_done()
            except asyncio.TimeoutError:
                break
        return ticks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics"""
        if self.start_time is None:
            return {"connected": False}
        
        uptime = time.time() - self.start_time
        return {
            "connected": self.is_connected,
            "uptime_seconds": uptime,
            "total_messages": self.total_messages,
            "total_ticks": self.total_ticks,
            "messages_per_second": self.total_messages / uptime if uptime > 0 else 0,
            "ticks_per_second": self.total_ticks / uptime if uptime > 0 else 0,
            "reconnect_count": self.reconnect_count,
            "last_price": self.last_price,
            "last_volume": self.last_volume,
            "queue_size": self.tick_queue.qsize()
        }
    
    async def stop(self):
        """Stop WebSocket connection"""
        print("[CoinbaseWebSocket] Stopping...")
        self.is_connected = False
        if self.ws:
            await self.ws.close()
        print("[CoinbaseWebSocket] Stopped")

# Example usage
async def example_usage():
    """Example of using CoinbaseWebSocket"""
    
    def on_tick(tick: TickData):
        print(f"Tick: {tick.timestamp} {tick.product_id} ${tick.price:.2f} x{tick.volume:.4f}")
    
    config = WebSocketConfig(
        products=["BTC-USD"],
        channels=["ticker", "matches"]
    )
    
    ws = CoinbaseWebSocket(config, on_tick)
    
    # Start listening in background
    listen_task = asyncio.create_task(ws.connect())
    
    # Process ticks
    try:
        while True:
            ticks = await ws.get_ticks(max_ticks=5)
            if ticks:
                for tick in ticks:
                    # Process tick here (add to buffer, generate vectors, etc.)
                    pass
            
            # Print stats every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = ws.get_stats()
                print(f"[Stats] TPS: {stats['ticks_per_second']:.1f}, "
                      f"Queue: {stats['queue_size']}, "
                      f"Price: ${stats['last_price']:.2f}")
            
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        await ws.stop()
        listen_task.cancel()

if __name__ == "__main__":
    print("Coinbase WebSocket Feed - Test Mode")
    print("="*50)
    asyncio.run(example_usage())