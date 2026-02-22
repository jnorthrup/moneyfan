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
    args = parser.parse_args()

    # Determine background config
    config = EpisodeTrainingConfig(
        n_epoch_episodes=args.episodes,
        notional=args.notional
    )

    # Start the singleton background trainer
    start_background_trainer(config)

    # Start the basic HTTP server on the main thread
    run_server(args.port)
