import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm.backtest import (CoinbaseSimulator, OrderType, Side, OrderStatus, 
                           TimeInForce, StopDirection)

class TestCoinbaseSpot(unittest.TestCase):
    def setUp(self):
        self.sim = CoinbaseSimulator(initial_balance={"USD": 10000.0, "BTC": 1.0})
        # Base market
        self.sim.update_market(1000, {
            "BTC-USD": {"bid": 50000, "ask": 50100, "last": 50050, "close": 50050, "volume": 100}
        })

    def test_stop_limit_buy(self):
        """Test Stop Limit Buy triggers when price rises"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.STOP_LIMIT, 
            price=51100, stop_price=51000.0
        )
        
        self.assertEqual(order.status, OrderStatus.OPEN)
        self.assertEqual(order.order_type, OrderType.STOP_LIMIT)

        # 1. Price below stop - No Trigger
        self.sim.update_market(1001, {
            "BTC-USD": {"last": 50500, "close": 50500}
        })
        self.assertEqual(order.order_type, OrderType.STOP_LIMIT)

        # 2. Price crosses stop - Trigger!
        self.sim.update_market(1002, {
            "BTC-USD": {"last": 51050, "close": 51050}
        })
        self.assertEqual(order.order_type, OrderType.LIMIT) # Should have converted
        self.assertEqual(order.status, OrderStatus.OPEN)    # Not filled yet (limit 51100 vs close 51050)

        # 3. Price crosses limit - Fill
        self.sim.update_market(1003, {
            "BTC-USD": {"last": 51150, "close": 51150, "low": 50900} # Low 50900 <= Limit 51100 -> Fill
        })
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_stop_market_sell(self):
        """Test Stop Market Sell triggers when price drops"""
        order = self.sim.place_order("BTC-USD", Side.SELL, 0.1, OrderType.STOP_MARKET,
                                     stop_price=49000.0)

        # 1. Price above stop
        self.sim.update_market(1001, {"BTC-USD": {"last": 49500}})
        self.assertEqual(order.order_type, OrderType.STOP_MARKET)

        # 2. Price drops - Trigger
        self.sim.update_market(1002, {"BTC-USD": {"last": 48900}})
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.status, OrderStatus.OPEN)

        # 3. Next tick matches Market
        self.sim.update_market(1003, {"BTC-USD": {"last": 48800, "close": 48800}})
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_post_only_reject(self):
        """Test Post Only rejects taker orders"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=50200, post_only=True
        )
        
        self.sim.update_market(1000.5, {
            "BTC-USD": {"last": 50050, "ask": 50100, "open": 50050}
        })
        
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_post_only_accept(self):
        """Test Post Only accepts maker orders"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=49900, post_only=True
        )
        
        self.sim.update_market(1000.5, {
             "BTC-USD": {"last": 50050, "ask": 50100, "open": 50050}
        })
        self.assertEqual(order.status, OrderStatus.OPEN)

    # ---------------------------------------------------------------
    # NEW: Time-in-Force tests
    # ---------------------------------------------------------------
    def test_limit_ioc_partial_fill(self):
        """IOC limit that crosses -> fills immediately"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, 
            price=50500, time_in_force=TimeInForce.IOC
        )
        # Next tick: low <= limit price -> fillable
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 50000, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_limit_ioc_no_fill(self):
        """IOC limit that doesn't cross -> cancelled"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, 
            price=49000, time_in_force=TimeInForce.IOC
        )
        # Next tick: low 49500 > limit 49000 -> unfillable
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 49500, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.CANCELLED)

    def test_limit_fok_fill(self):
        """FOK limit that crosses -> fills"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, 
            price=50500, time_in_force=TimeInForce.FOK
        )
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 50000, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_limit_fok_reject(self):
        """FOK limit that can't fill fully -> cancelled"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, 
            price=49000, time_in_force=TimeInForce.FOK
        )
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 49500, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.CANCELLED)

    def test_limit_gtd_expiry(self):
        """GTD limit expires when current_time >= end_time"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, 
            price=49000, time_in_force=TimeInForce.GTD, end_time=1005.0
        )
        # Orders are open before end_time
        self.sim.update_market(1003, {
            "BTC-USD": {"close": 50200, "low": 49500, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.OPEN)
        
        # Reaches end_time -> expired
        self.sim.update_market(1005, {
            "BTC-USD": {"close": 50200, "low": 49500, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.EXPIRED)

    # ---------------------------------------------------------------
    # NEW: Trigger Bracket tests
    # ---------------------------------------------------------------
    def test_trigger_bracket_gtc(self):
        """Bracket order: entry fill -> spawns TP + SL children"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.TRIGGER_BRACKET,
            price=50100,               # Entry limit
            take_profit_price=52000,    # TP sell limit
            stop_loss_price=49000       # SL sell stop
        )
        self.assertEqual(order.status, OrderStatus.OPEN)
        
        # Entry fills when low <= entry price
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 50000, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(len(order.bracket_child_ids), 2)
        
        # Children should exist
        tp_order = self.sim.orders[order.bracket_child_ids[0]]
        sl_order = self.sim.orders[order.bracket_child_ids[1]]
        self.assertEqual(tp_order.order_type, OrderType.LIMIT)
        self.assertEqual(tp_order.price, 52000)
        self.assertEqual(sl_order.order_type, OrderType.STOP_MARKET)
        self.assertEqual(sl_order.stop_price, 49000)
        
        # TP fills -> SL cancelled (OCO)
        self.sim.update_market(1002, {
            "BTC-USD": {"close": 52100, "low": 51900, "high": 52200, "volume": 100}
        })
        self.assertEqual(tp_order.status, OrderStatus.FILLED)
        self.assertEqual(sl_order.status, OrderStatus.CANCELLED)

    def test_trigger_bracket_gtd_expiry(self):
        """Bracket children expire with GTD end_time"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.TRIGGER_BRACKET,
            price=50100,
            take_profit_price=52000,
            stop_loss_price=49000,
            time_in_force=TimeInForce.GTD,
            end_time=1005.0
        )
        # Entry fills
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50200, "low": 50000, "high": 50600, "volume": 100}
        })
        self.assertEqual(order.status, OrderStatus.FILLED)
        
        tp_order = self.sim.orders[order.bracket_child_ids[0]]
        sl_order = self.sim.orders[order.bracket_child_ids[1]]
        
        # Neither child fills before expiry
        self.sim.update_market(1003, {
            "BTC-USD": {"close": 50500, "low": 50200, "high": 50800, "volume": 100}
        })
        self.assertEqual(tp_order.status, OrderStatus.OPEN)
        
        # Both children expire at end_time
        self.sim.update_market(1005, {
            "BTC-USD": {"close": 50500, "low": 50200, "high": 50800, "volume": 100}
        })
        self.assertEqual(tp_order.status, OrderStatus.EXPIRED)
        self.assertEqual(sl_order.status, OrderStatus.EXPIRED)

    # ---------------------------------------------------------------
    # NEW: Edit Order test
    # ---------------------------------------------------------------
    def test_edit_order(self):
        """Edit an open limit order's price and size"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=49000
        )
        self.assertEqual(order.price, 49000)
        
        success = self.sim.edit_order(order.id, price=49500, quantity=0.2)
        self.assertTrue(success)
        self.assertEqual(order.price, 49500)
        self.assertEqual(order.quantity, 0.2)
        
        # Cannot edit a market order
        mkt_order = self.sim.place_order("BTC-USD", Side.BUY, 0.1, OrderType.MARKET)
        self.assertFalse(self.sim.edit_order(mkt_order.id, price=50000))

    # ---------------------------------------------------------------
    # NEW: Batch Cancel test
    # ---------------------------------------------------------------
    def test_batch_cancel(self):
        """Cancel multiple orders at once"""
        o1 = self.sim.place_order("BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=48000)
        o2 = self.sim.place_order("BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=47000)
        o3 = self.sim.place_order("BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=46000)
        
        results = self.sim.cancel_orders([o1.id, o2.id, "nonexistent-id"])
        self.assertEqual(results[0], (o1.id, True))
        self.assertEqual(results[1], (o2.id, True))
        self.assertEqual(results[2], ("nonexistent-id", False))
        self.assertEqual(o3.status, OrderStatus.OPEN)  # Untouched

    # ---------------------------------------------------------------
    # NEW: Stop Direction tests
    # ---------------------------------------------------------------
    def test_stop_direction_up(self):
        """Explicit STOP_UP triggers on price rise"""
        order = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.STOP_MARKET,
            stop_price=51000, stop_direction=StopDirection.STOP_UP
        )
        # Below stop
        self.sim.update_market(1001, {"BTC-USD": {"close": 50500}})
        self.assertEqual(order.order_type, OrderType.STOP_MARKET)
        
        # Crosses stop upward
        self.sim.update_market(1002, {"BTC-USD": {"close": 51100}})
        self.assertEqual(order.order_type, OrderType.MARKET)

    def test_stop_direction_down(self):
        """Explicit STOP_DOWN triggers on price drop"""
        order = self.sim.place_order(
            "BTC-USD", Side.SELL, 0.1, OrderType.STOP_MARKET,
            stop_price=49000, stop_direction=StopDirection.STOP_DOWN
        )
        # Above stop
        self.sim.update_market(1001, {"BTC-USD": {"close": 49500}})
        self.assertEqual(order.order_type, OrderType.STOP_MARKET)
        
        # Crosses stop downward
        self.sim.update_market(1002, {"BTC-USD": {"close": 48900}})
        self.assertEqual(order.order_type, OrderType.MARKET)

    # ---------------------------------------------------------------
    # NEW: Self-Trade Prevention test
    # ---------------------------------------------------------------
    def test_self_trade_prevention(self):
        """Orders with same STP ID on opposite sides get cancelled"""
        # Resting sell limit — deep enough it won't fill this tick
        sell = self.sim.place_order(
            "BTC-USD", Side.SELL, 0.1, OrderType.LIMIT, price=55000,
            self_trade_prevention_id="account-1"
        )
        # Advance time so the buy is strictly newer
        self.sim.current_time = 1000.5
        # Buy that would cross the resting sell in the same account
        buy = self.sim.place_order(
            "BTC-USD", Side.BUY, 0.1, OrderType.LIMIT, price=56000,
            self_trade_prevention_id="account-1"
        )
        
        # On next tick, the buy (newer) should be cancelled by STP
        self.sim.update_market(1001, {
            "BTC-USD": {"close": 50250, "low": 50100, "high": 50400, "open": 50200, "volume": 100}
        })
        # The buy has a resting opposite-side order with same STP ID -> cancelled
        self.assertEqual(buy.status, OrderStatus.CANCELLED)
        # Original sell is untouched (not crossed, and is the older resting order)
        self.assertEqual(sell.status, OrderStatus.OPEN)

if __name__ == '__main__':
    unittest.main()
