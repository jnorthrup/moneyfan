"""Tests for SignalCache and TickFrame infrastructure."""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm.signal_cache import SignalCache
from hrm.tick_frame import TickFrame


class MockOrchestrator:
    """Minimal orchestrator stub that counts calls."""

    def __init__(self):
        self.services = {'grid': type('S', (), {'__name__': 'GridService'})(),
                         'momentum': type('S', (), {'__name__': 'MomentumService'})()}
        self.compositions = {'composite_alpha': None}
        self.call_count = 0

    def run(self, df):
        self.call_count += 1
        n = len(df)
        return {
            'signals': {
                'grid': pd.Series(np.random.randn(n), index=df.index),
                'momentum': pd.Series(np.random.randn(n), index=df.index),
            },
            'compositions': {
                'composite_alpha': pd.Series(np.random.randn(n), index=df.index),
            },
        }


def _make_candles(n=200, symbol='BTC-USD'):
    """Create a synthetic candle DataFrame."""
    times = pd.date_range('2021-01-01', periods=n, freq='5min')
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        'open': close - 10,
        'high': close + 20,
        'low': close - 20,
        'close': close,
        'volume': np.random.rand(n) * 1000,
    }, index=times)


class TestSignalCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = SignalCache(self.tmpdir)
        self.orch = MockOrchestrator()
        self.df = _make_candles(200)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_idempotency(self):
        """Two calls with same data return same result without recomputing."""
        r1 = self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.assertEqual(self.orch.call_count, 1)

        r2 = self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.assertEqual(self.orch.call_count, 1)  # No recompute

        pd.testing.assert_frame_equal(r1, r2, check_freq=False)

    def test_invalidate(self):
        """Invalidating forces recompute."""
        self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.assertEqual(self.orch.call_count, 1)

        self.cache.invalidate('BTC-USD')
        self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.assertEqual(self.orch.call_count, 2)

    def test_stats(self):
        """Stats returns correct symbol count."""
        self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.cache.get_or_compute('ETH-USD', self.df, self.orch)
        stats = self.cache.stats()
        self.assertEqual(stats['symbols'], 2)
        self.assertGreater(stats['bytes'], 0)

    def test_columns_present(self):
        """Cached DataFrame contains signal columns."""
        result = self.cache.get_or_compute('BTC-USD', self.df, self.orch)
        self.assertIn('grid', result.columns)
        self.assertIn('momentum', result.columns)
        self.assertIn('composite_alpha', result.columns)


class TestTickFrame(unittest.TestCase):

    def setUp(self):
        self.btc = _make_candles(100, 'BTC-USD')
        self.eth = _make_candles(100, 'ETH-USD')
        # Matching signal DataFrames
        self.sig_btc = pd.DataFrame({
            'alpha': np.random.randn(100),
        }, index=self.btc.index)
        self.sig_eth = pd.DataFrame({
            'alpha': np.random.randn(100),
        }, index=self.eth.index)

        self.tf = TickFrame(
            candles={'BTC-USD': self.btc, 'ETH-USD': self.eth},
            signals={'BTC-USD': self.sig_btc, 'ETH-USD': self.sig_eth},
        )

    def test_step_returns_data(self):
        """First step returns tick data for both symbols."""
        tick = self.tf.step()
        self.assertIsNotNone(tick)
        self.assertIn('BTC-USD', tick)
        self.assertIn('close', tick['BTC-USD'].index)

    def test_step_exhaustion(self):
        """Stepping past the end returns None."""
        for _ in range(self.tf.total()):
            self.tf.step()
        self.assertIsNone(self.tf.step())

    def test_seek(self):
        """Seek jumps to correct position."""
        t = self.tf.time_index[50]
        self.tf.seek(t)
        self.assertEqual(self.tf.current_pos, 50)

    def test_window(self):
        """Window returns correct lookback size."""
        # Advance to position 60
        for _ in range(60):
            self.tf.step()
        win = self.tf.window('BTC-USD', 32)
        self.assertIsNotNone(win)
        self.assertEqual(len(win), 32)
        # Should include signal columns
        self.assertIn('alpha', win.columns)
        self.assertIn('close', win.columns)

    def test_remaining(self):
        self.assertEqual(self.tf.remaining(), self.tf.total())
        self.tf.step()
        self.assertEqual(self.tf.remaining(), self.tf.total() - 1)

    def test_reset(self):
        for _ in range(10):
            self.tf.step()
        self.tf.reset()
        self.assertEqual(self.tf.current_pos, 0)


if __name__ == '__main__':
    unittest.main()
