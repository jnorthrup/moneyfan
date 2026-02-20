"""
HRM Bot Runner
==============

Main entry point for the HRM Tradebot.
Connects the data (Orchestrator) -> Brain (HRM) -> Execution (Stub).
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

# Ensure imports work
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hrm.signal_hrm import SignalHRM, SignalHRMConfig, HAS_MLX
from hrm.orchestrator_bridge import OrchestratorBridge
from hrm.fiduciary_controller import FiduciaryControllerHRM # For future use
from hrm.currency_graph import build_coinbase_graph_depth, DepthBasedRouter, Currency
from hrm.backtest import CoinbaseSimulator, OrderType, Side

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    pass

def load_hrm_model(config: SignalHRMConfig, dry_run: bool = False):
    """Load the HRM model (random weights for now as we don't have a checkpoint)"""
    if dry_run or not HAS_MLX:
        print("MLX not found (or dry_run=True). Running in simulation mode without neural net inference.")
        return None
    
    model = SignalHRM(config)
    mx.eval(model.parameters())
    return model

def run_bot(symbols: List[str], interval: int = 60, lookback: int = 500, paper_mode: bool = False):
    """
    Main trading loop.
    
    Args:
        symbols: List of trading pairs (e.g., ['BTC', 'ETH'])
        interval: Loop interval in seconds
        lookback: Number of candles to load
        paper_mode: If True, simulate execution with backtest
    """
    print(f"Initializing HRM Bot for {symbols}...")
    
    # 1. Setup Bridge
    bridge = OrchestratorBridge()
    
    # 2. Setup HRM
    cfg = SignalHRMConfig()
    model = load_hrm_model(cfg, dry_run=not HAS_MLX)
    
    # 3. Setup Execution (Graph + Simulator)
    graph = build_coinbase_graph_depth()
    router = DepthBasedRouter(graph)
    
    sim = None
    if paper_mode:
        print("Starting in PAPER TRADING mode.")
        sim = CoinbaseSimulator(initial_balance={"USD": 10000.0, "BTC": 0.0})
    else:
        print("Starting in DRY RUN mode (no execution).")

    print(f"Graph initialized with {len(graph.currencies)} currencies. Routing enabled.")
    print("System initialized. Starting loop...")
    
    try:
        while True:
            current_time = time.time()
            print(f"\n--- {time.ctime()} ---")
            
            for symbol in symbols: # symbol e.g., "BTC" (asset)
                # A. Get Market State (Tensor)
                try:
                    # Request tensor + context
                    tensor_np = bridge.compute_tensor(symbol, lookback=lookback, seq_len=cfg.seq_len, include_context=True)
                except Exception as e:
                    print(f"[{symbol}] Error computing signals: {e}")
                    continue
                
                if tensor_np is None:
                    print(f"[{symbol}] Insufficient data.")
                    continue
                
                # B. Run Brain (HRM)
                alpha_val = 0.0
                conv_val = 0.0
                
                if HAS_MLX and model:
                    tensor_mx = mx.array(tensor_np)
                    # HRM forward pass
                    weights, alpha, convergence, _mem = model(tensor_mx)
                    
                    # Convert to python types
                    alpha_val = float(alpha[0].item())
                    conv_val = float(convergence[0].item())
                    weights_np = np.array(weights[0])
                    
                    # Log Decision
                    direction = "LONG" if alpha_val > 0.05 else ("SHORT" if alpha_val < -0.05 else "NEUTRAL")
                    print(f"[{symbol}] HRM Decision: {direction:<7} | Alpha: {alpha_val:+.4f} | Converg: {conv_val:.4f}")
                    
                    # Log Weights (top 3 signals)
                    from hrm.signal_hrm import SIGNAL_16
                    top_indices = weights_np.argsort()[-3:][::-1]
                    top_signals = [(SIGNAL_16[i], weights_np[i]) for i in top_indices]
                    top_str = ", ".join([f"{n}: {w:.2f}" for n, w in top_signals])
                    # print(f"       Top Factors: {top_str}")

                # C. Execution (Paper Mode)
                if paper_mode and sim:
                    # Mock price update
                    df = bridge.loader.load_symbol(symbol, lookback=1)
                    if not df.empty:
                        last_price = df['close'].iloc[-1]
                        pair_symbol = f"{symbol}-USD"
                        sim.update_market(current_time, {
                            pair_symbol: {"close": last_price, "last": last_price, "bid": last_price, "ask": last_price}
                        })
                        
                        # Decision Logic
                        target_currency = Currency(symbol)
                        usd_currency = Currency("USD")
                        
                        if alpha_val > 0.5:
                            # BUY ROUTE
                            route = router.find_route(usd_currency, target_currency, prefer_depth=True)
                            if route:
                                print(f"[{symbol}] Executing BUY via route: {route}")
                                qty = (sim.balances.get("USD", 0) * 0.1) / last_price
                                if qty > 0.0001:
                                    sim.place_order(pair_symbol, Side.BUY, qty, OrderType.MARKET)
                        
                        elif alpha_val < -0.5:
                            # SELL ROUTE
                            route = router.find_route(target_currency, usd_currency, prefer_depth=True)
                            if route:
                                print(f"[{symbol}] Executing SELL via route: {route}")
                                qty = sim.balances.get(symbol, 0)
                                if qty > 0.0001:
                                    sim.place_order(pair_symbol, Side.SELL, qty, OrderType.MARKET)
                        
            if paper_mode and sim:
                pv = sim.get_portfolio_value({f"{s}-USD": 50000.0 for s in symbols}) # Mock price for PV
                print(f"Portfolio Value: ${pv:.2f} | Trades: {len(sim.trades)}")
                
            # Sleep
            print(f"Sleeping {interval}s...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping bot...")

class BacktestEngine:
    """5-year seekable backtest with 25 agents and signal caching.
    
    Flow:
    1. Load candle data from ArrowStore (mmap, zero-copy)
    2. Pre-compute + cache all signals via SignalCache  
    3. Build TickFrame (unified time index)
    4. Step through ticks:
       a. Update CoinbaseSimulator market state
       b. All 25 RankedTraders observe cached signals (no recompute)
       c. Each trader places orders
       d. Simulator matches orders
    5. Emit top/bottom 3 rankings at conclusion during test-time training, plot convergence or identify paralsysis 
    """
    
    PAIRS = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD',
        'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD', 'LTC-USD',
        'SHIB-USD', 'UNI-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD',
        'NEAR-USD', 'ATOM-USD', 'FIL-USD', 'HBAR-USD', 'APT-USD',
        'LDO-USD', 'VET-USD', 'QNT-USD', 'MKR-USD', 'AAVE-USD',
        'FTM-USD', 'SAND-USD', 'MANA-USD', 'XTZ-USD', 'EOS-USD',
        'ETH-BTC', 'SOL-ETH', 'SOL-BTC', 'AVAX-BTC', 'MATIC-BTC'
    ]
    
    def __init__(self, pairs: List[str] = None, n_traders: int = 25, 
                 days: int = 365 * 5, report_every: int = 5000,
                 cache_dir: str = 'hrm/data/signal_cache'):
        self.pairs = pairs or self.PAIRS
        self.n_traders = n_traders
        self.days = days
        self.report_every = report_every
        self.cache_dir = cache_dir
        
        # Ranking snapshots for convergence plotting
        self.snapshots: List[Dict] = []
    
    def run(self, start: pd.Timestamp = None, end: pd.Timestamp = None):
        from hrm.coinbase_pipeline import CoinbasePipeline
        from hrm.signal_cache import SignalCache
        from hrm.tick_frame import TickFrame
        from hrm.ranking_harness import RankedTrader
        from hrm.signal_hrm import SignalHRMConfig, HAS_MLX
        from signal_orchestrator import (
            Orchestrator, GridService, MomentumService, RSIService,
            TrendService, VolatilityService, VolumeService, CompositionModel
        )
        from datetime import datetime, timedelta
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        years = self.days // 365
        print(f"{'=' * 72}")
        print(f"  HRM BACKTEST  {years}Y @ 5min  |  {self.n_traders} agents  |  {len(self.pairs)} pairs")
        print(f"{'=' * 72}")
        
        # ── 1. DATA LOAD ─────────────────────────────────────────────
        pipeline = CoinbasePipeline()
        
        if end is None:
            end = pd.Timestamp(datetime.utcnow())
        if start is None:
            start = end - pd.Timedelta(days=self.days)
        
        candles = {}
        def fetch(pair):
            df = pipeline.history.load_range(pair, start.to_pydatetime(), end.to_pydatetime())
            if not df.empty:
                df = df[~df.index.duplicated(keep='first')]
                return (pair, df)
            return None
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch, p): p for p in self.pairs}
            for f in as_completed(futures):
                res = f.result()
                if res and len(res[1]) >= 100:
                    candles[res[0]] = res[1]
        
        print(f"  Loaded {len(candles)} pairs | ", end="")
        total_rows = sum(len(v) for v in candles.values())
        print(f"{total_rows:,} candle rows")
        
        if not candles:
            print("  No data. Exiting.")
            return
        
        # ── 2. SIGNAL CACHE ───────────────────────────────────────────
        orchestrator = Orchestrator(max_workers=8)
        orchestrator.register_service(GridService())
        orchestrator.register_service(MomentumService())
        orchestrator.register_service(RSIService())
        orchestrator.register_service(TrendService())
        orchestrator.register_service(VolatilityService())
        orchestrator.register_service(VolumeService())
        orchestrator.register_composition(
            CompositionModel('composite_alpha')
                .add_signal('momentum', 1.0)
                .add_signal('rsi', 1.0, op='multiply')
                .add_signal('trend', 1.0, op='multiply')
                .add_signal('volatility', 1.0, op='multiply')
        )
        
        sig_cache = SignalCache(self.cache_dir)
        signals = {}
        
        for sym, df in candles.items():
            signals[sym] = sig_cache.get_or_compute(sym, df, orchestrator)
        
        cache_stats = sig_cache.stats()
        print(f"  Signal cache: {cache_stats['symbols']} symbols, "
              f"{cache_stats['bytes'] / 1e6:.1f} MB")
        
        # ── 3. TICK FRAME ─────────────────────────────────────────────
        tick_frame = TickFrame(candles, signals)
        print(f"  TickFrame: {tick_frame.total():,} ticks across {len(tick_frame.symbols)} symbols")
        
        # ── 4. AGENTS ─────────────────────────────────────────────────
        cfg = SignalHRMConfig(seq_len=32, hidden_dim=64)
        traders = []
        for i in range(self.n_traders):
            # Alternate A/B: even=B(MLX), odd=A(CPU)
            mode = "B" if (HAS_MLX and i % 2 == 0) else "A"
            traders.append(RankedTrader(i, cfg, mode=mode))
        
        # ── 5. MAIN LOOP ─────────────────────────────────────────────
        tick_count = 0
        
        while True:
            tick = tick_frame.step()
            if tick is None:
                break
            
            t = tick_frame.current_time
            ts = float(t.timestamp()) if hasattr(t, 'timestamp') else 0.0
            
            # Build market state for simulator (all symbols this tick)
            for sym, row in tick.items():
                close = float(row.get('close', 0))
                if close <= 0:
                    continue
                
                # Feed each trader
                for trader in traders:
                    trader.process_candle(
                        low=float(row.get('low', close)),
                        high=float(row.get('high', close)),
                        close=close
                    )
                    
                    # Build tensor from signal window
                    win = tick_frame.window(sym, cfg.seq_len)
                    if win is not None and len(win) >= cfg.seq_len:
                        # Extract signal columns for HRM input
                        sig_cols = [c for c in win.select_dtypes(include=[np.number]).columns 
                                    if c not in ('open', 'high', 'low', 'close', 'volume', 'timestamp')]
                        if sig_cols:
                            from hrm.signal_hrm import N_SIGNALS
                            raw = win[sig_cols].values[-cfg.seq_len:]  # [seq_len, n_sig]
                            
                            # Local MinMax Scaling ([-1, 1]) per signal
                            raw_min = raw.min(axis=0)
                            raw_max = raw.max(axis=0)
                            raw_range = raw_max - raw_min
                            raw_range[raw_range == 0] = 1.0
                            raw = 2.0 * (raw - raw_min) / raw_range - 1.0
                            
                            n_sig = raw.shape[1]
                            
                            # Interleave signal + confidence (1.0) to match input_dim
                            # Layout: [s0, c0, s1, c1, ...] + context padding
                            interleaved = np.zeros((cfg.seq_len, N_SIGNALS * 2), dtype=np.float32)
                            for j in range(min(n_sig, N_SIGNALS)):
                                interleaved[:, j * 2] = raw[:, j]       # signal
                                interleaved[:, j * 2 + 1] = 1.0        # confidence
                            
                            # Pad to full input_dim (N_SIGNALS*2 + context_dim)
                            full_width = cfg.input_dim
                            if interleaved.shape[1] < full_width:
                                pad = np.zeros((cfg.seq_len, full_width - interleaved.shape[1]))
                                sig_tensor = np.concatenate([interleaved, pad], axis=1)
                            else:
                                sig_tensor = interleaved[:, :full_width]
                            
                            sig_tensor = sig_tensor.reshape(1, cfg.seq_len, full_width).astype(np.float32)
                            
                            # Forward return for TTL
                            ret = np.array([0.0], dtype=np.float32)
                            if 'close' in win.columns and len(win) > 1:
                                prices = win['close'].values
                                ret = np.array([(prices[-1] / prices[-2]) - 1.0], dtype=np.float32)
                            
                            if HAS_MLX:
                                import mlx.core as mx
                                sig_tensor = mx.array(sig_tensor)
                                ret = mx.array(ret)
                            
                            trader.update(sig_tensor, ret, close)
            
            tick_count += 1
            
            # ── SNAPSHOT (silent) ─────────────────────────────────
            if tick_count % self.report_every == 0:
                self._collect_snapshot(traders, tick_count, tick_frame)
        
        # ── CONCLUSION ────────────────────────────────────────────
        self._collect_snapshot(traders, tick_count, tick_frame)
        
        print(f"\n{'=' * 72}")
        print(f"  FINAL RANKING  ({tick_count:,} ticks)")
        print(f"{'=' * 72}")
        self._print_ranking(traders)
        self._print_convergence()
        
        return self.snapshots
    
    def _collect_snapshot(self, traders, tick_count: int, tick_frame):
        """Silently collect ranking snapshot for convergence plotting."""
        rows = []
        for t in traders:
            wealth = t.balance + t.position * 100  # approx
            pnl = wealth - 10000.0
            rows.append({
                'id': t.trader_id,
                'mode': t.mode,
                'reward': t.cumulative_reward,
                'pnl': pnl,
                'pnl_pct': pnl / 10000.0,
                'wealth': wealth,
            })
        
        df = pd.DataFrame(rows).sort_values('reward', ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        
        self.snapshots.append({
            'tick': tick_count,
            'progress': tick_frame.progress(),
            'top3': df.head(3)[['id', 'mode', 'reward', 'pnl_pct']].to_dict('records'),
            'bot3': df.tail(3)[['id', 'mode', 'reward', 'pnl_pct']].to_dict('records'),
        })
    
    def _print_ranking(self, traders):
        """Print top 3 + bottom 3 at conclusion."""
        rows = []
        for t in traders:
            wealth = t.balance + t.position * 100
            pnl = wealth - 10000.0
            rows.append({
                'id': t.trader_id,
                'mode': t.mode,
                'reward': t.cumulative_reward,
                'pnl': pnl,
                'pnl_pct': pnl / 10000.0,
                'wealth': wealth,
            })
        
        df = pd.DataFrame(rows).sort_values('reward', ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        
        print(f"  {'Rk':>3} {'ID':>3} {'M':>1} {'Reward':>10} {'PnL':>10} {'PnL%':>8} {'Wealth':>10}")
        print(f"  {'─' * 50}")
        
        for rank, row in df.head(3).iterrows():
            print(f"  {rank:>3} {int(row['id']):>3} {row['mode']:>1} "
                  f"{row['reward']:>+10.4f} {row['pnl']:>+10.2f} "
                  f"{row['pnl_pct']:>+7.2%} {row['wealth']:>10,.2f}")
        
        if len(df) > 6:
            print(f"  {'...':>3}")
        
        for rank, row in df.tail(3).iterrows():
            print(f"  {rank:>3} {int(row['id']):>3} {row['mode']:>1} "
                  f"{row['reward']:>+10.4f} {row['pnl']:>+10.2f} "
                  f"{row['pnl_pct']:>+7.2%} {row['wealth']:>10,.2f}")
    
    def _print_convergence(self):
        """Detect convergence or paralysis from snapshot deltas."""
        if len(self.snapshots) < 3:
            return
        
        # Track top-1 reward across snapshots
        rewards = [s['top3'][0]['reward'] for s in self.snapshots]
        deltas = [rewards[i] - rewards[i - 1] for i in range(1, len(rewards))]
        
        # Check if top-1 ID is stable (convergence) or volatile (exploration)
        top_ids = [s['top3'][0]['id'] for s in self.snapshots]
        last_n = top_ids[-min(5, len(top_ids)):]
        stable = len(set(last_n)) == 1
        
        # Paralysis: reward deltas near zero for last N snapshots
        recent_deltas = deltas[-min(5, len(deltas)):]
        paralyzed = all(abs(d) < 1e-6 for d in recent_deltas)
        
        print(f"\n  ── Convergence ──")
        print(f"  Snapshots: {len(self.snapshots)} | "
              f"Reward Δ (last): {deltas[-1]:+.6f} | "
              f"Leader stable: {'YES' if stable else 'NO'}")
        
        if paralyzed:
            print(f"  ⚠  PARALYSIS: reward deltas near zero for last {len(recent_deltas)} snapshots")
        elif stable and abs(deltas[-1]) < 1e-4:
            print(f"  ✓  CONVERGED: leader stable, reward delta diminishing")
        else:
            print(f"  ↻  TRAINING: agents still differentiating")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRM Tradebot")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH"], help="Symbols to trade (Live Mode)")
    parser.add_argument("--interval", type=int, default=10, help="Loop interval in seconds")
    parser.add_argument("--lookback", type=int, default=300, help="Data lookback")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--days", type=int, default=365*5, help="Backtest duration in days")
    parser.add_argument("--traders", type=int, default=25, help="Number of agents")
    parser.add_argument("--report-every", type=int, default=5000, help="Ticks between ranking reports")
    
    args = parser.parse_args()
    
    if args.backtest:
        engine = BacktestEngine(
            n_traders=args.traders,
            days=args.days,
            report_every=args.report_every,
        )
        engine.run()
    else:
        run_bot(args.symbols, args.interval, args.lookback, paper_mode=args.paper)

