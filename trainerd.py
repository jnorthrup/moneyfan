#!/usr/bin/env python3
"""
MoneyFan Trainer Web Daemon (trainerd.py)
=========================================

A completely brainless Python HTTP server serving a static vanilla web console,
running the EpochEpisodeTrainer as a background singleton daemon.

Usage:
    python3 trainerd.py --port 8080
"""

import sys
import os
import json
import time
import argparse
import threading
import queue
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import EpochEpisodeTrainer, EpisodeTrainingConfig

# Global Singleton Trainer
trainer_instance = None
trainer_thread = None
latest_results = []
latest_samples = []
MAX_RESULTS_HISTORY = 50
MAX_SAMPLES_HISTORY = 100

# Global active transfers registry
active_transfers = {}
transfer_lock = threading.Lock()

# Global lock for thread-safe state access
state_lock = threading.Lock()

DRAWTHRU_DUCKDB_FILE = Path(__file__).resolve().parent / "data" / "binance" / "hrm_data.duckdb"


def load_drawthru_snapshot():
    try:
        import duckdb
    except Exception as e:
        return {"status": "unavailable", "error": f"duckdb import failed: {e}"}

    if not DRAWTHRU_DUCKDB_FILE.exists():
        return {"status": "missing", "db_path": str(DRAWTHRU_DUCKDB_FILE)}

    try:
        con = duckdb.connect(str(DRAWTHRU_DUCKDB_FILE), read_only=True)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "binance_sequences_import" in tables:
            table_name = "binance_sequences_import"
            where = ""
        elif "market_data" in tables:
            table_name = "market_data"
            where = "WHERE lower(coalesce(exchange, '')) = 'binance'"
        else:
            con.close()
            return {
                "status": "empty",
                "db_path": str(DRAWTHRU_DUCKDB_FILE),
                "tables": sorted(tables),
            }

        row = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(timestamp), MAX(timestamp)
            FROM {table_name}
            {where}
            """
        ).fetchone()
        top = con.execute(
            f"""
            SELECT symbol, COUNT(*) AS row_count, MAX(timestamp) AS last_ts
            FROM {table_name}
            {where}
            GROUP BY symbol
            ORDER BY row_count DESC, symbol ASC
            LIMIT 8
            """
        ).fetchall()

        preview_row = con.execute(
            f"""
            SELECT symbol, MAX(timestamp) AS last_ts
            FROM {table_name}
            {where}
            GROUP BY symbol
            ORDER BY last_ts DESC, symbol ASC
            LIMIT 1
            """
        ).fetchone()
        preview_symbol = preview_row[0] if preview_row else None
        preview_bars = []
        if preview_symbol:
            preview_q = con.execute(
                f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                {where}
                {"AND" if where else "WHERE"} symbol = ?
                ORDER BY timestamp DESC
                LIMIT 80
                """,
                [preview_symbol],
            ).fetchall()
            preview_bars = [
                {
                    "timestamp": None if ts is None else str(ts),
                    "open": None if o is None else float(o),
                    "high": None if h is None else float(h),
                    "low": None if l is None else float(l),
                    "close": None if c is None else float(c),
                    "volume": None if v is None else float(v),
                }
                for ts, o, h, l, c, v in reversed(preview_q)
            ]
        con.close()

        return {
            "status": "ok",
            "db_path": str(DRAWTHRU_DUCKDB_FILE),
            "table": table_name,
            "row_count": int(row[0] or 0),
            "symbol_count": int(row[1] or 0),
            "min_ts": None if row[2] is None else str(row[2]),
            "max_ts": None if row[3] is None else str(row[3]),
            "top_symbols": [
                {"symbol": s, "row_count": int(c), "last_ts": None if ts is None else str(ts)}
                for s, c, ts in top
            ],
            "preview_symbol": preview_symbol,
            "preview_bars": preview_bars,
        }
    except Exception as e:
        return {"status": "error", "db_path": str(DRAWTHRU_DUCKDB_FILE), "error": str(e)}

def start_background_trainer(config: EpisodeTrainingConfig):
    global trainer_instance
    trainer_instance = EpochEpisodeTrainer(config)

    # We need to drain the trainer's internal queue to keep state up to date
    def monitor_queue():
        global latest_results
        print("[Trainer HTTP Daemon] Monitor thread waiting for trainer to start...")
        while not trainer_instance or not trainer_instance.running:
            time.sleep(0.1)
        
        print("[Trainer HTTP Daemon] Monitor thread active. Draining event queue.")
        while trainer_instance.running:
            try:
                event_type, data = trainer_instance.event_queue.get(timeout=1.0)
                if event_type == 'episode_complete':
                    with state_lock:
                        latest_results.append(data)
                        if len(latest_results) > MAX_RESULTS_HISTORY:
                            latest_results = latest_results[-MAX_RESULTS_HISTORY:]
                elif event_type == 'sample_event':
                    with state_lock:
                        latest_samples.append(data)
                        if len(latest_samples) > MAX_SAMPLES_HISTORY:
                            latest_samples = latest_samples[-MAX_SAMPLES_HISTORY:]
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Trainer HTTP Daemon] Queue monitor error: {e}")
                time.sleep(1)

    print("[Trainer HTTP Daemon] Starting singleton trainer thread...")
    trainer_thread = threading.Thread(
        target=trainer_instance.run_episode_training,
        daemon=True
    )
    trainer_thread.start()

    monitor_thread = threading.Thread(
        target=monitor_queue,
        daemon=True
    )
    monitor_thread.start()
    print("[Trainer HTTP Daemon] Trainer running in background.")

class TrainerHTTPHandler(SimpleHTTPRequestHandler):
    """Serve the static console GUI and provide two JSON APIs"""

    def __init__(self, *args, **kwargs):
        # Serve static files from the 'console' directory
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "console"), **kwargs)

    def end_headers(self):
        # Disable caching for API and dev UX
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)

        # Basic routing
        if parsed_path.path == '/api/state':
            self.serve_api_state()
            return
        elif parsed_path.path == '/api/cache':
            self.serve_api_cache()
            return
        elif parsed_path.path == '/api/drawthru':
            self.serve_api_drawthru()
            return
        elif parsed_path.path == '/api/vqa':
            self.serve_api_vqa()
            return

        # Fallback to serving static files from /console (handled by super)
        super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/vqa':
            self.serve_api_vqa()
            return
        super().do_POST()

    def serve_api_vqa(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        
        try:
            req_data = json.loads(post_data)
        except:
            req_data = {}

        question = req_data.get('question', '').lower()
        
        # Simple Pilot reasoning logic (Brainless for now, but extensible)
        answer = "I'm monitoring the cockpit. Ask about win rate, PnL, or cache status."
        
        if "win rate" in question or "accuracy" in question:
            with state_lock:
                wr = latest_results[-1].get('hit_rate', 0.0) if latest_results else 0.0
                answer = f"Our current direction accuracy is {wr:.1%}. Tactical layer is holding steady."
        elif "pnl" in question or "profit" in question:
            with state_lock:
                pnl = sum([r.get('realized_pnl', 0.0) for r in latest_results])
                answer = f"Total net realized PnL for this session is ${pnl:.2f}. Capital is nominal."
        elif "cache" in question:
            with state_lock:
                size = len(trainer_instance.candle_cache.cache)
                answer = f"Stochastic Drawthru Cache is at {size} entries. Environmental data is piping through."
        elif "who" in question or "best" in question:
            with state_lock:
                hero = latest_results[-1].get('winning_agent', '--') if latest_results else "--"
                answer = f"Expert {hero} is currently leading the mission conviction matrix."

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"answer": answer}).encode('utf-8'))

    def serve_api_state(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        global trainer_instance
        if not trainer_instance:
            self.wfile.write(json.dumps({"status": "booting"}).encode('utf-8'))
            return

        with state_lock:
            # Safely capture top level stats
            response_data = {
                "status": "running" if trainer_instance.running else "stopped",
                "session_start_time": trainer_instance.session_start_time,
                "training_config": {
                    "optimizer_name": getattr(trainer_instance.config, "optimizer_name", "adamw"),
                    "learning_rate": getattr(trainer_instance.config, "learning_rate", None),
                    "weight_decay": getattr(trainer_instance.config, "weight_decay", None),
                },
                "history": latest_results, # Last N completed episodes
                "samples": latest_samples, # Last N sampling events
            }

            # If there's at least one result, attach a subset of the first/last elements to build global metrics easily
            if latest_results:
                response_data["latest_metrics"] = {
                    "total_trained": len(trainer_instance.results),
                    "current_capital": latest_results[-1].get("final_capital", 0.0),
                    "total_realized_pnl": sum([r.get('realized_pnl', 0.0) for r in trainer_instance.results if 'realized_pnl' in r]),
                }

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def serve_api_cache(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        global trainer_instance
        if not trainer_instance:
            response_data = {"cache_status": "offline"}
            with transfer_lock:
                trans_list = sorted(active_transfers.values(), key=lambda x: x.get('name', ''))
                response_data["transfers"] = trans_list
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

        cache_ref = trainer_instance.candle_cache

        with state_lock:
            # Peek at the cache contents safely
            keys = list(cache_ref.cache.keys())
            sizes = {k: len(cache_ref.cache[k]) for k in keys}
            response_data = {
                "cache_status": "online",
                "max_size": cache_ref.max_size,
                "current_size": len(keys),
                "keys": keys,
                "memory_rows": sum(sizes.values()),
                "access_order": cache_ref.access_order[:]
            }

        with transfer_lock:
            # We sort transfers so they are consistently ordered, eg by name
            trans_list = sorted(active_transfers.values(), key=lambda x: x.get('name', ''))
            response_data["transfers"] = trans_list

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def serve_api_drawthru(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(load_drawthru_snapshot()).encode('utf-8'))


def run_server(port: int = 8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, TrainerHTTPHandler)
    print(f"[Trainer HTTP Daemon] Brainless Web Server serving at http://localhost:{port}/")
    print("[Trainer HTTP Daemon] Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Trainer HTTP Daemon] Shutting down...")
        if trainer_instance:
            trainer_instance.running = False
        httpd.server_close()
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoneyFan Trainer Web Daemon")
    parser.add_argument("--port", type=int, default=8080, help="HTTP Server Port")
    parser.add_argument("--episodes", type=int, default=500, help="Epoch Episodes")
    parser.add_argument("--notional", type=float, default=100.0, help="Starting Notional")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "lion", "muon"],
                        help="MLX optimizer for HRM training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="MLX optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="MLX optimizer weight decay")
    args = parser.parse_args()

    # Determine background config
    config = EpisodeTrainingConfig(
        n_epoch_episodes=args.episodes,
        notional=args.notional,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Start the singleton background trainer
    start_background_trainer(config)

    # Start the basic HTTP server on the main thread
    run_server(args.port)
