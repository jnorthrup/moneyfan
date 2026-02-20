"""
Bounded Coinbase API
Realtime WebSocket + Historical Puller
Feeds stochastic frame bags for continuous HRM training
"""

import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import pandas as pd

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from hrm.duck_store import DuckStore
except ImportError:
    try:
        from duck_store import DuckStore
    except:
        from hrm.arrow_store import ArrowStore as DuckStore
    
try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests


COINBASE_EXCHANGE_REST_API = "https://api.exchange.coinbase.com"
COINBASE_ADVANCED_WS_MARKET = "wss://advanced-trade-ws.coinbase.com"
COINBASE_ADVANCED_WS_USER = "wss://advanced-trade-ws-user.coinbase.com"


@dataclass
class Instrument:
    symbol: str
    base: str
    quote: str
    min_size: float
    max_size: float
    step_size: float
    active: bool = True
    

@dataclass 
class Tick:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0
    ask: float = 0
    last: float = 0


@dataclass
class MarketEvent:
    channel: str
    symbol: str
    event_type: str
    sequence_num: Optional[int]
    timestamp: int
    payload: Dict[str, Any]
    source: str = "coinbase-advanced-trade-ws"


class CoinbaseInstruments:
    """Registry of tradeable instruments with filtering by holdings gravity"""
    
    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}
        self.prices: Dict[str, float] = {}
        self._loaded = False
        
    def load(self) -> bool:
        """Load instruments from Coinbase"""
        try:
            resp = requests.get(f"{COINBASE_EXCHANGE_REST_API}/products", timeout=10)
            resp.raise_for_status()
            products = resp.json()
            
            for p in products:
                if p.get('trading_disabled') or p.get('cancel_only'):
                    continue
                    
                self.instruments[p['id']] = Instrument(
                    symbol=p['id'],
                    base=p['base_currency'],
                    quote=p['quote_currency'],
                    min_size=float(p.get('base_min_size', 0.001)),
                    max_size=float(p.get('base_max_size', 100000)),
                    step_size=float(p.get('base_increment', 0.00000001)),
                    active=True
                )
                
            self._loaded = True
            print(f"Loaded {len(self.instruments)} instruments from Coinbase")
            return True
            
        except Exception as e:
            print(f"Failed to load instruments: {e}")
            return False
    
    def filter_by_holdings(self, holdings: Dict[str, float], 
                          top_n: int = 64,
                          min_confidence: float = 0.1) -> List[str]:
        """
        Filter instruments based on holdings gravity.
        Assets with larger holdings get more bandwidth.
        """
        if not self._loaded:
            self.load()
            
        scored = []
        total_value = sum(holdings.values()) if holdings else 1
        
        for symbol, inst in self.instruments.items():
            if not inst.active:
                continue
                
            position_value = holdings.get(symbol, 0) * self.prices.get(symbol, 1)
            gravity = position_value / (total_value + 1e-8)
            
            scored.append((symbol, gravity))
            
        scored.sort(key=lambda x: -x[1])
        return [s[0] for s in scored[:top_n]]
    
    def get_price(self, symbol: str) -> float:
        if symbol not in self.prices:
            try:
                resp = requests.get(f"{COINBASE_EXCHANGE_REST_API}/products/{symbol}/ticker", timeout=5)
                if resp.ok:
                    self.prices[symbol] = float(resp.json().get('price', 0))
            except:
                pass
        return self.prices.get(symbol, 0)


class CoinbaseHistory:
    """Pull historical candles for arbitrary timespan"""
    
    def __init__(self, db_path: str = "hrm/data/coinbase.duckdb", arrow_dir: str = "hrm/data/arrow"):
        self.store = DuckStore(db_path, arrow_dir)
        
    def pull_range(self, symbol: str, start: datetime, end: datetime,
                   granularity: int = 60) -> pd.DataFrame:
        """Pull historical candles for a time range"""
        
        candles = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(seconds=granularity * 300), end)
            
            try:
                url = f"{COINBASE_EXCHANGE_REST_API}/products/{symbol}/candles"
                params = {
                    'start': current.isoformat(),
                    'end': chunk_end.isoformat(),
                    'granularity': granularity
                }
                
                # Retry loop for 429s and connection errors
                backoff = 1.0
                max_retries = 10
                
                for attempt in range(max_retries):
                    try:
                        resp = requests.get(url, params=params, timeout=10)
                        
                        if resp.status_code == 429:
                            if attempt < max_retries - 1:
                                sleep_time = backoff * (1.5 ** attempt)
                                # print(f"   Rate limited {symbol}, sleeping {sleep_time:.1f}s...")
                                time.sleep(sleep_time)
                                continue
                            else:
                                resp.raise_for_status()
                                
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            raise e
                
                for c in reversed(data):
                    candles.append({
                        'timestamp': int(c[0]),
                        'low': float(c[1]),
                        'high': float(c[2]),
                        'open': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    })
                    
                # Incremental Save
                if len(candles) >= 2000:
                    df_chunk = pd.DataFrame(candles)
                    df_chunk['time'] = pd.to_datetime(df_chunk['timestamp'], unit='s')
                    df_chunk.set_index('time', inplace=True)
                    self._save_candles(symbol, df_chunk)
                    candles = []
                    
                current = chunk_end + timedelta(seconds=granularity)
                time.sleep(0.2)  # Reduced pause for faster pulls (use with caution)
                
            except Exception as e:
                print(f"Error pulling {symbol} at {current}: {e}")
                current = chunk_end + timedelta(seconds=granularity)
                time.sleep(1)
                
        if candles:
            df_chunk = pd.DataFrame(candles)
            df_chunk['time'] = pd.to_datetime(df_chunk['timestamp'], unit='s')
            df_chunk.set_index('time', inplace=True)
            self._save_candles(symbol, df_chunk)
            
        return self.load_range(symbol, start, end)
    
    def _save_candles(self, symbol: str, df: pd.DataFrame):
        self.store.upsert(symbol, df)
        
    def load_range(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load candles from database"""
        return self.store.load(symbol, start, end)
    
    def sample_bag(self, symbols: List[str], n_samples: int = 1000,
                   min_len: int = 128) -> List[pd.DataFrame]:
        """Sample random bags of historical data"""
        min_ts, max_ts = None, None
        
        for symbol in symbols:
            bounds = self.store.get_bounds(symbol)
            if bounds[0]:
                if min_ts is None or bounds[0] < min_ts:
                    min_ts = bounds[0]
                if max_ts is None or bounds[1] > max_ts:
                    max_ts = bounds[1]
        
        if min_ts is None:
            return []
            
        bags = []
        
        for _ in range(n_samples):
            frames = []
            lookback_seconds = 24 * 3600
            search_start = max(int(min_ts.timestamp()), int(max_ts.timestamp()) - lookback_seconds)
            
            start_ts = random.randint(search_start, int(max_ts.timestamp()) - 50 * 60)
            end_ts = start_ts + 50 * 60
            
            for symbol in symbols:
                df = self.store.load(symbol)
                if len(df) >= min_len:
                    df = df.copy()
                    df['symbol'] = symbol
                    frames.append(df.tail(50))

            if frames:
                bags.append(pd.concat(frames))
        
        print(f"SampleBag produced {len(bags)} bags of size {[len(b) for b in bags]}")
        return bags


class CoinbaseRealtime:
    """
    Production-style Advanced Trade market websocket adapter.

    Features:
      - one-channel-per-subscribe semantics
      - reconnect with exponential backoff
      - optional JWT rotation per subscribe message
      - normalized MarketEvent fanout for HRM/oversight
      - legacy ticker fallback support
    """

    def __init__(
        self,
        instruments: CoinbaseInstruments,
        market_ws_url: str = COINBASE_ADVANCED_WS_MARKET,
        channels: Optional[List[str]] = None,
        jwt_provider: Optional[Callable[[], str]] = None,
    ):
        self.instruments = instruments
        self.market_ws_url = market_ws_url
        self.channels = channels or ["heartbeats", "ticker", "market_trades", "level2"]
        self.jwt_provider = jwt_provider

        self.ticks: Dict[str, deque] = {}
        self.callbacks: List[Callable[[Tick], None]] = []
        self.event_callbacks: List[Callable[[MarketEvent], None]] = []
        self.events: deque = deque(maxlen=100000)
        self.running = False
        self._ws = None

        self._sequence_by_channel: Dict[str, int] = {}
        self.sequence_gaps: deque = deque(maxlen=1000)
        self._resubscribe_interval_s = 90
        self._max_backoff_s = 30

    def subscribe(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Tick], None]] = None,
        event_callback: Optional[Callable[[MarketEvent], None]] = None,
    ):
        """Subscribe to symbols and register callbacks."""
        for symbol in symbols:
            if symbol not in self.ticks:
                self.ticks[symbol] = deque(maxlen=10000)

        if callback:
            self.callbacks.append(callback)
        if event_callback:
            self.event_callbacks.append(event_callback)

    def on_event(self, callback: Callable[[MarketEvent], None]):
        self.event_callbacks.append(callback)

    def _next_jwt(self) -> Optional[str]:
        if self.jwt_provider:
            try:
                token = self.jwt_provider()
                return token or None
            except Exception as e:
                print(f"JWT provider error: {e}")
                return None
        token = os.getenv("COINBASE_WS_JWT", "").strip()
        return token or None

    async def _send_subscribe(self, ws, channel: str, symbols: List[str]):
        msg = {
            "type": "subscribe",
            "channel": channel,
            "product_ids": symbols,
        }
        jwt_token = self._next_jwt()
        if jwt_token:
            msg["jwt"] = jwt_token
        await ws.send(json.dumps(msg))

    async def _send_initial_subscriptions(self, ws, symbols: List[str]):
        for channel in self.channels:
            await self._send_subscribe(ws, channel=channel, symbols=symbols)

    async def _refresh_subscriptions(self, ws, symbols: List[str]):
        for channel in self.channels:
            try:
                await self._send_subscribe(ws, channel=channel, symbols=symbols)
            except Exception as e:
                print(f"Refresh subscribe failed ({channel}): {e}")

    def _track_sequence(self, channel: str, sequence_num: Optional[int]):
        if sequence_num is None:
            return
        prev = self._sequence_by_channel.get(channel)
        if prev is not None and sequence_num > prev + 1:
            self.sequence_gaps.append({
                "channel": channel,
                "prev_sequence": prev,
                "next_sequence": sequence_num,
                "timestamp": int(time.time()),
            })
        self._sequence_by_channel[channel] = sequence_num

    async def _connect(self, symbols: List[str]):
        backoff = 1
        while self.running:
            try:
                async with websockets.connect(
                    self.market_ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10000,
                    open_timeout=15,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    await self._send_initial_subscriptions(ws, symbols)
                    next_refresh_at = time.time() + self._resubscribe_interval_s
                    backoff = 1

                    while self.running:
                        timeout = max(1.0, next_refresh_at - time.time())
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                            data = json.loads(raw)
                            self._process_message(data)
                        except asyncio.TimeoutError:
                            await ws.ping()
                        except json.JSONDecodeError:
                            continue

                        if time.time() >= next_refresh_at:
                            await self._refresh_subscriptions(ws, symbols)
                            next_refresh_at = time.time() + self._resubscribe_interval_s
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                if not self.running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

    def _parse_ts(self, raw_ts: Any) -> int:
        if raw_ts is None:
            return int(time.time())
        if isinstance(raw_ts, (int, float)):
            if raw_ts > 1e12:
                return int(raw_ts / 1000)
            return int(raw_ts)
        try:
            ts = pd.to_datetime(raw_ts, utc=True)
            return int(ts.timestamp())
        except Exception:
            return int(time.time())

    def _emit_event(self, event: MarketEvent):
        self.events.append(event)
        for cb in self.event_callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"Event callback error: {e}")

    def _emit_tick(self, tick: Tick, channel: str, sequence_num: Optional[int], payload: Dict[str, Any], event_type: str):
        if tick.symbol not in self.ticks:
            self.ticks[tick.symbol] = deque(maxlen=10000)

        self.ticks[tick.symbol].append(tick)
        self.instruments.prices[tick.symbol] = tick.close

        for cb in self.callbacks:
            try:
                cb(tick)
            except Exception as e:
                print(f"Tick callback error: {e}")

        self._emit_event(MarketEvent(
            channel=channel,
            symbol=tick.symbol,
            event_type=event_type,
            sequence_num=sequence_num,
            timestamp=tick.timestamp,
            payload=payload,
        ))

    def _process_ticker_record(self, channel: str, sequence_num: Optional[int], event_type: str, ticker: Dict[str, Any]):
        symbol = ticker.get("product_id")
        if not symbol:
            return

        price = float(ticker.get("price", ticker.get("close", 0.0)) or 0.0)
        bid = float(ticker.get("best_bid", ticker.get("bid", 0.0)) or 0.0)
        ask = float(ticker.get("best_ask", ticker.get("ask", 0.0)) or 0.0)
        volume = float(
            ticker.get("volume_24_h", ticker.get("volume_24h", ticker.get("volume", 0.0))) or 0.0
        )
        ts = self._parse_ts(ticker.get("time"))

        tick = Tick(
            symbol=symbol,
            timestamp=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            bid=bid,
            ask=ask,
            last=price,
        )
        self._emit_tick(tick, channel=channel, sequence_num=sequence_num, payload=ticker, event_type=event_type)

    def _process_trade_record(self, channel: str, sequence_num: Optional[int], event_type: str, trade: Dict[str, Any]):
        symbol = trade.get("product_id")
        if not symbol:
            return
        price = float(trade.get("price", 0.0) or 0.0)
        size = float(trade.get("size", trade.get("volume", 0.0)) or 0.0)
        ts = self._parse_ts(trade.get("time"))

        tick = Tick(
            symbol=symbol,
            timestamp=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            last=price,
        )
        self._emit_tick(tick, channel=channel, sequence_num=sequence_num, payload=trade, event_type=event_type)

    def _process_candle_record(self, channel: str, sequence_num: Optional[int], event_type: str, candle: Dict[str, Any]):
        symbol = candle.get("product_id")
        if not symbol:
            return
        ts = self._parse_ts(candle.get("start"))
        o = float(candle.get("open", 0.0) or 0.0)
        h = float(candle.get("high", 0.0) or 0.0)
        l = float(candle.get("low", 0.0) or 0.0)
        c = float(candle.get("close", 0.0) or 0.0)
        v = float(candle.get("volume", 0.0) or 0.0)

        tick = Tick(
            symbol=symbol,
            timestamp=ts,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            last=c,
        )
        self._emit_tick(tick, channel=channel, sequence_num=sequence_num, payload=candle, event_type=event_type)

    def _process_legacy_ticker(self, data: Dict[str, Any]):
        symbol = data.get("product_id")
        if not symbol:
            return
        tick = Tick(
            symbol=symbol,
            timestamp=int(time.time()),
            open=float(data.get("open_24h", 0)),
            high=float(data.get("high_24h", 0)),
            low=float(data.get("low_24h", 0)),
            close=float(data.get("price", 0)),
            volume=float(data.get("volume_24h", 0)),
            bid=float(data.get("bid", 0)),
            ask=float(data.get("ask", 0)),
            last=float(data.get("price", 0)),
        )
        self._emit_tick(tick, channel="ticker", sequence_num=None, payload=data, event_type="legacy_ticker")

    def _process_message(self, data: Dict[str, Any]):
        channel = data.get("channel")
        sequence_num = data.get("sequence_num")

        if channel:
            self._track_sequence(channel, sequence_num)
            events = data.get("events") or []
            for event in events:
                event_type = event.get("type", "update")
                if channel in ("ticker", "ticker_batch"):
                    for ticker in event.get("tickers", []):
                        self._process_ticker_record(channel, sequence_num, event_type, ticker)
                elif channel == "market_trades":
                    for trade in event.get("trades", []):
                        self._process_trade_record(channel, sequence_num, event_type, trade)
                elif channel == "candles":
                    for candle in event.get("candles", []):
                        self._process_candle_record(channel, sequence_num, event_type, candle)
                else:
                    symbol = event.get("product_id", "")
                    self._emit_event(MarketEvent(
                        channel=channel,
                        symbol=symbol,
                        event_type=event_type,
                        sequence_num=sequence_num,
                        timestamp=self._parse_ts(event.get("time")),
                        payload=event,
                    ))
            return

        if data.get("type") == "ticker":
            self._process_legacy_ticker(data)

    def start(self, symbols: List[str]):
        if not HAS_WEBSOCKETS:
            print("websockets not installed, using REST polling")
            self._start_polling(symbols)
            return

        self.running = True
        thread = threading.Thread(
            target=lambda: asyncio.run(self._connect(symbols)),
            daemon=True,
        )
        thread.start()

    def _start_polling(self, symbols: List[str]):
        self.running = True
        thread = threading.Thread(
            target=self._poll_loop,
            args=(symbols,),
            daemon=True,
        )
        thread.start()

    def _poll_loop(self, symbols: List[str]):
        while self.running:
            for symbol in symbols:
                try:
                    resp = requests.get(
                        f"{COINBASE_EXCHANGE_REST_API}/products/{symbol}/ticker",
                        timeout=5,
                    )
                    if resp.ok:
                        data = resp.json()
                        self._process_legacy_ticker({
                            "product_id": symbol,
                            "price": data.get("price", 0),
                            "open_24h": data.get("open_24h", 0),
                            "high_24h": data.get("high_24h", 0),
                            "low_24h": data.get("low_24h", 0),
                            "volume_24h": data.get("volume_24h", 0),
                            "bid": data.get("bid", 0),
                            "ask": data.get("ask", 0),
                        })
                except Exception:
                    pass
            time.sleep(1)

    def stop(self):
        self.running = False

    def get_recent(self, symbol: str, n: int = 100) -> List[Tick]:
        return list(self.ticks.get(symbol, []))[-n:]

    def get_events(self, symbol: Optional[str] = None, channel: Optional[str] = None, n: int = 100) -> List[MarketEvent]:
        events = list(self.events)
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        if channel:
            events = [e for e in events if e.channel == channel]
        return events[-n:]


class CoinbasePipeline:
    """Complete pipeline: history + realtime → stochastic bags"""
    
    def __init__(self, db_path: str = "hrm/data/coinbase.duckdb"):
        self.instruments = CoinbaseInstruments()
        self.history = CoinbaseHistory(db_path)
        self.realtime = CoinbaseRealtime(self.instruments)
        self.holdings: Dict[str, float] = {}
        
    def initialize(self, pull_days: int = 365, granularity: int = 60) -> bool:
        """Initialize with historical data"""
        if not self.instruments.load():
            return False
            
        symbols = list(self.instruments.instruments.keys())[:64]
        end = datetime.utcnow()
        start = end - timedelta(days=pull_days)
        
        print(f"Pulling {pull_days} days of history for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] {symbol}")
            self.history.pull_range(symbol, start, end, granularity=granularity)
            time.sleep(0.2)
            
        return True
        
    def update_holdings(self, holdings: Dict[str, float]):
        """Update current holdings for gravity filtering"""
        self.holdings = holdings
        
    def get_active_instruments(self, top_n: int = 64) -> List[str]:
        """Get instruments filtered by holdings gravity"""
        return self.instruments.filter_by_holdings(self.holdings, top_n)
        
    def sample_training_bag(self, n_samples: int = 1000) -> List[pd.DataFrame]:
        """Sample stochastic bag for training"""
        symbols = self.get_active_instruments()
        return self.history.sample_bag(symbols, n_samples)

    def sample_stratified_bag(self, n_samples: int = 1, lookback_days: int = 30) -> List[pd.DataFrame]:
        """
        Sample exclusive Coinbase bag:
        - 10 Winners (Top performers)
        - 10 Losers (Bottom performers)
        - 10 Countercoins (Stable/Hedge pairs: USDT, USDC, BTC, ETH, etc.)
        """
        if not self.instruments._loaded:
            self.instruments.load()
            
        symbols = list(self.instruments.instruments.keys())
        # Filter for USD/USDT/USDC pairs only to ensure exclusivity
        cb_symbols = [s for s in symbols if s.endswith(('-USD', '-USDT', '-USDC'))]
        
        # ONLY include symbols we have arrow data for
        available_symbols = []
        arrow_dir = "hrm/data/arrow"
        for s in cb_symbols:
            slug = s.replace("-", "_").replace("/", "_")
            if os.path.exists(os.path.join(arrow_dir, f"{slug}.feather")):
                available_symbols.append(s)
        
        if not available_symbols:
            print("No instruments with arrow data found.")
            return []
            
        print(f"Sampling bag from {len(available_symbols)} symbols with arrow data...")
        
        # Calculate performance over lookback
        returns = []
        for symbol in available_symbols:
            df = self.history.load_range(symbol, datetime.utcnow() - timedelta(days=lookback_days), datetime.utcnow())
            if len(df) > 100:
                ret = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1.0
                returns.append((symbol, ret))
        
        returns.sort(key=lambda x: x[1], reverse=True)
        
        winners = [x[0] for x in returns[:10]]
        losers = [x[0] for x in returns[-10:]]
        
        # Countercoins: Major pairs and stables
        counter_candidates = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'DAI-USD', 'WBTC-USD', 'SOL-USD', 'BNB-USD', 'LINK-USD', 'ADA-USD']
        countercoins = [c for c in counter_candidates if c in self.instruments.instruments][:10]
        
        target_symbols = list(set(winners + losers + countercoins))
        print(f"Stratified Bag: {len(winners)} Winners, {len(losers)} Losers, {len(countercoins)} Counter")
        
        return self.history.sample_bag(target_symbols, n_samples)
        
    def start_realtime(self, callback: Callable = None):
        """Start real-time feed"""
        symbols = self.get_active_instruments()
        if callback:
            self.realtime.subscribe(symbols, callback)
        self.realtime.start(symbols)
        
    def stop(self):
        self.realtime.stop()


def main():
    print("=" * 60)
    print("  COINBASE PIPELINE - Stochastic Bag Generator")
    print("=" * 60)
    
    pipeline = CoinbasePipeline()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--init':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
        pipeline.initialize(pull_days=days)
    elif len(sys.argv) > 1 and sys.argv[1] == '--stratified':
        print("Generating 10/10/10 Stratified Bag (Exclusive to Coinbase)...")
        bags = pipeline.sample_stratified_bag(n_samples=1)
        if bags:
            print(f"Sample bag shape: {bags[0].shape}")
            print(f"Symbols in bag: {bags[0]['symbol'].unique()}")
    else:
        print("Sampling from existing data...")
        bags = pipeline.sample_training_bag(n_samples=100)
        print(f"Generated {len(bags)} stochastic bags")
        
        if bags:
            print(f"Sample bag shape: {bags[0].shape}")
            print(f"Symbols in bag: {bags[0]['symbol'].unique()[:5]}")


if __name__ == "__main__":
    main()
