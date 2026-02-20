"""
Coinbase Advanced Trade Emulator (backtest)
============================================

A high-fidelity backtesting engine that simulates Coinbase Advanced Trading execution.
Focuses on realistic costs: fees, slippage, and latency.

Fee Tier (default):
  Maker: 0.40%
  Taker: 0.60%
"""

import uuid
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TRIGGER_BRACKET = "TRIGGER_BRACKET"

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TimeInForce(Enum):
    GTC = "GTC"   # Good Till Cancelled (default)
    IOC = "IOC"   # Immediate Or Cancel — fill what's available, cancel rest
    GTD = "GTD"   # Good Till Date — GTC with expiry timestamp
    FOK = "FOK"   # Fill Or Kill — all-or-nothing

class StopDirection(Enum):
    STOP_UP = "STOP_DIRECTION_STOP_UP"     # Trigger when price >= stop
    STOP_DOWN = "STOP_DIRECTION_STOP_DOWN" # Trigger when price <= stop

class OrderStatus(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"   # GTD order reached end_time
    PENDING = "PENDING"   # Stop/bracket awaiting trigger

@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None           # Limit price (None for Market)
    stop_price: Optional[float] = None      # Stop trigger price
    post_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    stop_direction: Optional[StopDirection] = None  # Explicit trigger direction
    end_time: Optional[float] = None        # GTD expiry timestamp
    
    # Trigger Bracket fields
    take_profit_price: Optional[float] = None   # TP limit for bracket
    stop_loss_price: Optional[float] = None     # SL stop for bracket
    
    # Self-trade prevention
    self_trade_prevention_id: Optional[str] = None
    
    # Linked bracket children (internal bookkeeping)
    bracket_parent_id: Optional[str] = None     # Points to parent bracket order
    bracket_child_ids: List[str] = field(default_factory=list)  # OCO children
    
    # State
    status: OrderStatus = OrderStatus.OPEN
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    timestamp: float
    is_maker: bool

class CoinbaseSimulator:
    def __init__(self, 
                 initial_balance: Dict[str, float] = None,
                 maker_fee: float = 0.004,
                 taker_fee: float = 0.006,
                 latency_ms: float = 50.0):
        
        self.balances = initial_balance or {"USD": 10000.0}
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency_s = latency_ms / 1000.0
        
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.current_time = 0.0
        self.market_data: Dict[str, Dict] = {} # {symbol: {bid: ..., ask: ..., last: ...}}

    def update_market(self, timestamp: float, prices: Dict[str, Dict]):
        """
        Update simulation state with new market data.
        prices: {symbol: {'bid': float, 'ask': float, 'last': float, 'high': float, 'low': float}}
        """
        self.current_time = timestamp
        self.market_data.update(prices)
        self._process_orders(prices)

    def place_order(self, 
                    symbol: str, 
                    side: Side, 
                    quantity: float, 
                    order_type: OrderType = OrderType.MARKET, 
                    price: float = None,
                    stop_price: float = None,
                    stop_direction: StopDirection = None,
                    post_only: bool = False,
                    time_in_force: TimeInForce = TimeInForce.GTC,
                    end_time: float = None,
                    take_profit_price: float = None,
                    stop_loss_price: float = None,
                    self_trade_prevention_id: str = None) -> Order:
        """
        Place a spot order. Mirrors Coinbase Advanced Trade create_order.
        
        Args:
            symbol: Product ID e.g. "BTC-USD"
            side: BUY or SELL
            quantity: Base asset size
            order_type: MARKET, LIMIT, STOP_LIMIT, STOP_MARKET, TRIGGER_BRACKET
            price: Limit price (required for LIMIT, STOP_LIMIT, TRIGGER_BRACKET)
            stop_price: Stop trigger price (required for STOP_LIMIT, STOP_MARKET)
            stop_direction: STOP_UP or STOP_DOWN (auto-inferred from side if None)
            post_only: If True, reject if would take liquidity
            time_in_force: GTC, IOC, GTD, FOK
            end_time: Expiry timestamp for GTD orders
            take_profit_price: Take-profit limit for TRIGGER_BRACKET
            stop_loss_price: Stop-loss trigger for TRIGGER_BRACKET
            self_trade_prevention_id: Prevents self-crossing
        """
        # Validation
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if time_in_force == TimeInForce.GTD and end_time is None:
            raise ValueError("GTD orders require end_time")
        
        if order_type == OrderType.TRIGGER_BRACKET:
            if take_profit_price is None or stop_loss_price is None:
                raise ValueError("TRIGGER_BRACKET requires take_profit_price and stop_loss_price")
            if price is None:
                raise ValueError("TRIGGER_BRACKET requires entry limit price")
        
        # Auto-infer stop_direction from side if not provided
        if stop_price is not None and stop_direction is None:
            if side == Side.BUY:
                stop_direction = StopDirection.STOP_UP    # Buy breakout
            else:
                stop_direction = StopDirection.STOP_DOWN  # Sell stop-loss
        
        # Check balance (simplified pre-check)
        base, quote = symbol.split('-') # e.g. BTC-USD
        if side == Side.BUY:
            est_cost = quantity * (price if price else self._get_estimated_price(symbol, Side.BUY))
            if self.balances.get(quote, 0) < est_cost:
                pass  # Allow in simulation, reject at execution time
                
        elif side == Side.SELL:
            if self.balances.get(base, 0) < quantity:
                raise ValueError(f"Insufficient {base} balance")

        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            stop_direction=stop_direction,
            post_only=post_only,
            time_in_force=time_in_force,
            end_time=end_time,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            self_trade_prevention_id=self_trade_prevention_id,
            created_at=self.current_time,
            updated_at=self.current_time
        )
        
        self.orders[order.id] = order
        
        # Market orders will be matched on next _process_orders call
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single open order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                order.status = OrderStatus.CANCELLED
                order.updated_at = self.current_time
                # Cancel bracket children (OCO cleanup)
                self._cancel_bracket_children(order)
                return True
        return False

    def cancel_orders(self, order_ids: List[str]) -> List[Tuple[str, bool]]:
        """Batch cancel. Returns list of (order_id, success)."""
        return [(oid, self.cancel_order(oid)) for oid in order_ids]

    def edit_order(self, order_id: str, 
                   price: float = None, 
                   quantity: float = None) -> bool:
        """
        Edit an open limit order's price and/or size.
        Only LIMIT orders in OPEN status can be edited.
        """
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status != OrderStatus.OPEN:
            return False
        if order.order_type != OrderType.LIMIT:
            return False
        if price is not None:
            order.price = price
        if quantity is not None:
            if quantity <= 0:
                return False
            order.quantity = quantity
        order.updated_at = self.current_time
        return True

    def _get_estimated_price(self, symbol: str, side: Side) -> float:
        """Get naive price for balance check"""
        data = self.market_data.get(symbol)
        if not data:
            return 0.0
        return data['ask'] if side == Side.BUY else data['bid']

    def _cancel_bracket_children(self, parent: Order):
        """Cancel all OCO children of a bracket order."""
        for child_id in parent.bracket_child_ids:
            if child_id in self.orders:
                child = self.orders[child_id]
                if child.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                    child.status = OrderStatus.CANCELLED
                    child.updated_at = self.current_time

    def _cancel_oco_sibling(self, filled_child: Order):
        """When one bracket child fills, cancel the other (OCO)."""
        if not filled_child.bracket_parent_id:
            return
        parent = self.orders.get(filled_child.bracket_parent_id)
        if not parent:
            return
        for child_id in parent.bracket_child_ids:
            if child_id != filled_child.id and child_id in self.orders:
                sibling = self.orders[child_id]
                if sibling.status in (OrderStatus.OPEN, OrderStatus.PENDING):
                    sibling.status = OrderStatus.CANCELLED
                    sibling.updated_at = self.current_time

    def _spawn_bracket_children(self, parent: Order):
        """After bracket entry fills, spawn TP limit + SL stop as OCO pair."""
        # Take-Profit: opposite side limit
        tp_side = Side.SELL if parent.side == Side.BUY else Side.BUY
        tp_order = Order(
            id=str(uuid.uuid4()),
            symbol=parent.symbol,
            side=tp_side,
            order_type=OrderType.LIMIT,
            quantity=parent.filled_quantity,
            price=parent.take_profit_price,
            time_in_force=parent.time_in_force,
            end_time=parent.end_time,
            bracket_parent_id=parent.id,
            created_at=self.current_time,
            updated_at=self.current_time
        )
        
        # Stop-Loss: opposite side stop-market
        sl_direction = StopDirection.STOP_DOWN if parent.side == Side.BUY else StopDirection.STOP_UP
        sl_order = Order(
            id=str(uuid.uuid4()),
            symbol=parent.symbol,
            side=tp_side,
            order_type=OrderType.STOP_MARKET,
            quantity=parent.filled_quantity,
            stop_price=parent.stop_loss_price,
            stop_direction=sl_direction,
            time_in_force=parent.time_in_force,
            end_time=parent.end_time,
            bracket_parent_id=parent.id,
            created_at=self.current_time,
            updated_at=self.current_time
        )
        
        parent.bracket_child_ids = [tp_order.id, sl_order.id]
        self.orders[tp_order.id] = tp_order
        self.orders[sl_order.id] = sl_order

    def _check_self_trade(self, order: Order) -> bool:
        """Returns True if order would self-trade (and should be prevented).
        
        Only the newer (aggressor) order is cancelled. The resting (older) 
        order stays open. This matches Coinbase's STP behavior.
        """
        if not order.self_trade_prevention_id:
            return False
        for other_id, other in self.orders.items():
            if other.id == order.id:
                continue
            if other.status != OrderStatus.OPEN:
                continue
            if other.symbol != order.symbol:
                continue
            if other.self_trade_prevention_id != order.self_trade_prevention_id:
                continue
            # Same STP ID, same symbol, opposite sides
            if other.side != order.side:
                # Only cancel the newer order (the aggressor)
                if order.created_at >= other.created_at:
                    return True
        return False

    def _process_orders(self, prices: Dict[str, Dict]):
        """Match open orders against new candle/tick data."""
        for order_id, order in list(self.orders.items()):
            if order.status not in (OrderStatus.OPEN, OrderStatus.PENDING):
                continue
            
            symbol = order.symbol
            if symbol not in prices:
                continue
            
            candle = prices[symbol]
            current_price = candle.get('close', candle.get('last'))
            
            # 0. GTD EXPIRY CHECK
            if order.time_in_force == TimeInForce.GTD and order.end_time is not None:
                if self.current_time >= order.end_time:
                    order.status = OrderStatus.EXPIRED
                    order.updated_at = self.current_time
                    self._cancel_bracket_children(order)
                    continue
            
            # 1. LATENCY CHECK
            if self.current_time - order.created_at < self.latency_s:
                continue

            # 2. SELF-TRADE PREVENTION
            if self._check_self_trade(order):
                order.status = OrderStatus.CANCELLED
                order.updated_at = self.current_time
                continue

            # 3. TRIGGER STOP ORDERS (uses stop_direction if set)
            if order.order_type in (OrderType.STOP_LIMIT, OrderType.STOP_MARKET):
                if order.stop_price is None:
                    continue
                
                triggered = False
                if order.stop_direction == StopDirection.STOP_UP:
                    triggered = current_price >= order.stop_price
                elif order.stop_direction == StopDirection.STOP_DOWN:
                    triggered = current_price <= order.stop_price
                else:
                    # Legacy fallback: infer from side
                    if order.side == Side.BUY:
                        triggered = current_price >= order.stop_price
                    else:
                        triggered = current_price <= order.stop_price
                
                if triggered:
                    if order.order_type == OrderType.STOP_LIMIT:
                        order.order_type = OrderType.LIMIT
                    else:
                        order.order_type = OrderType.MARKET
                    order.status = OrderStatus.OPEN
                    continue  # Process as new type on next tick

            # 4. TRIGGER BRACKET ENTRY
            if order.order_type == OrderType.TRIGGER_BRACKET:
                # Bracket entry acts like a limit order
                limit_price = order.price
                would_fill = False
                if order.side == Side.BUY:
                    if candle.get('low', float('inf')) <= limit_price:
                        would_fill = True
                elif order.side == Side.SELL:
                    if candle.get('high', -1.0) >= limit_price:
                        would_fill = True
                
                if would_fill:
                    self._execute_trade(order, limit_price, is_maker=True)
                    if order.status == OrderStatus.FILLED:
                        self._spawn_bracket_children(order)
                continue  # Bracket entry handled, skip normal matching

            # 5. MATCHING LOGIC
            executed_price = None
            is_maker = False
            
            if order.order_type == OrderType.MARKET:
                base_price = candle.get('close', candle.get('last'))
                slippage = self._calculate_slippage(order.quantity, candle.get('volume', 10000))
                if order.side == Side.BUY:
                    executed_price = base_price * (1 + slippage)
                else:
                    executed_price = base_price * (1 - slippage)
                is_maker = False

            elif order.order_type == OrderType.LIMIT:
                limit_price = order.price
                would_execute = False
                
                if order.side == Side.BUY:
                    if candle.get('low', float('inf')) <= limit_price:
                        would_execute = True
                elif order.side == Side.SELL:
                    if candle.get('high', -1.0) >= limit_price:
                        would_execute = True
                
                # POST ONLY CHECK
                if order.post_only:
                    ref_price = candle.get('open', candle.get('last'))
                    is_taker_price = (order.side == Side.BUY and limit_price >= ref_price) or \
                                     (order.side == Side.SELL and limit_price <= ref_price)
                                     
                    if is_taker_price and (self.current_time - order.created_at < 1.0): 
                        order.status = OrderStatus.REJECTED
                        continue

                if would_execute:
                    executed_price = limit_price
                    is_maker = True

            # 6. TIF ENFORCEMENT (IOC / FOK)
            if order.time_in_force == TimeInForce.IOC:
                # IOC: fill what's available on this tick, cancel rest
                if executed_price is not None:
                    self._execute_trade(order, executed_price, is_maker)
                    if order.bracket_parent_id:
                        self._cancel_oco_sibling(order)
                else:
                    # Nothing fillable on this tick — cancel
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = self.current_time
                continue  # IOC is always resolved on first matchable tick

            if order.time_in_force == TimeInForce.FOK:
                # FOK: all-or-nothing — in backtest candle context, 
                # we check if the limit would have filled the full quantity.
                # Since we don't have depth, treat "would_execute" as full fill possible.
                if executed_price is not None:
                    self._execute_trade(order, executed_price, is_maker)
                    if order.bracket_parent_id:
                        self._cancel_oco_sibling(order)
                else:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = self.current_time
                continue  # FOK is always resolved on first matchable tick

            # 7. EXECUTE TRADES (GTC / GTD — standard fill)
            if executed_price:
                self._execute_trade(order, executed_price, is_maker)
                if order.status == OrderStatus.FILLED and order.bracket_parent_id:
                    self._cancel_oco_sibling(order)

    def _execute_trade(self, order: Order, price: float, is_maker: bool):
        # Calculate Fees
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        
        # Quote Value
        value = order.quantity * price
        fee = value * fee_rate
        
        # Update Balances
        base, quote = order.symbol.split('-')
        
        if order.side == Side.BUY:
            # Spend Quote, Receive Base
            cost = value + fee
            if self.balances.get(quote, 0) >= cost:
                self.balances[quote] -= cost
                self.balances[base] = self.balances.get(base, 0) + order.quantity
            else:
                # Partial / Reject (Simulate full fill or bust)
                order.status = OrderStatus.REJECTED
                return
        else:
            # Sell Base, Receive Quote
            if self.balances.get(base, 0) >= order.quantity:
                self.balances[base] -= order.quantity
                proceeds = value - fee
                self.balances[quote] = self.balances.get(quote, 0) + proceeds
            else:
                order.status = OrderStatus.REJECTED
                return

        # Update Order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = price
        order.fees_paid = fee
        order.updated_at = self.current_time
        
        # Record Trade
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            fee=fee,
            timestamp=self.current_time,
            is_maker=is_maker
        )
        self.trades.append(trade)

    def _calculate_slippage(self, quantity: float, volume: float) -> float:
        """
        Estimate slippage based on order size vs volume.
        Square root law or similar.
        """
        if volume <= 0:
            return 0.01 # 1% default on no volume
        
        participation = quantity / volume
        # Simple linear model: 10% participation -> 1% slippage
        slippage = min(participation * 0.1, 0.05) # Cap at 5%
        return slippage

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        total = self.balances.get("USD", 0.0)
        for sym, qty in self.balances.items():
            if sym == "USD": continue
            # Assume price is keyed by symbol like "BTC-USD"
            # We need to find the price for this asset
            # Naive lookup
            pair = f"{sym}-USD" 
            if pair in prices:
                total += qty * prices[pair]
        return total

# Example Usage
if __name__ == "__main__":
    sim = CoinbaseSimulator()
    print("Balance:", sim.balances)
    
    # 1. Market Order Buy
    sim.update_market(1000, {"BTC-USD": {"bid": 50000, "ask": 50100, "last": 50050, "volume": 100}})
    order = sim.place_order("BTC-USD", Side.BUY, 0.1, OrderType.MARKET)
    sim.update_market(1001, {"BTC-USD": {"bid": 50100, "ask": 50200, "last": 50150, "close": 50150, "volume": 100}})
    
    print(f"Order: {order.status} @ {order.avg_fill_price}")
    print("Balance:", sim.balances)
    print("Trades:", len(sim.trades))
