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
from dataclasses import asdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
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
event_stream_log = []
next_event_seq = 1
MAX_EVENT_STREAM_HISTORY = 1000
event_stream_cond = threading.Condition()

# Global active transfers registry
active_transfers = {}
transfer_lock = threading.Lock()

# Global lock for thread-safe state access
state_lock = threading.Lock()

DRAWTHRU_DUCKDB_FILE = Path(__file__).resolve().parent / "data" / "binance" / "hrm_data.duckdb"
OPENAPI_SPEC_FILE = Path(__file__).resolve().parent / "trainerd.openapi.yaml"


RUNTIME_CONTROL_SCHEMA = {
    "notional": {"type": "float", "min": 1e-6, "max": 1e12},
    "pair_width": {"type": "int", "min": 1, "max": 512},
    "bar_sequences_per_episode": {"type": "int", "min": 1, "max": 100000},
    "epochs": {"type": "int", "min": 1, "max": 128},
    "min_bar_window": {"type": "int", "min": 8, "max": 8192},
    "max_bar_window": {"type": "int", "min": 8, "max": 8192},
    "cache_size": {"type": "int", "min": 1, "max": 500000},
    "candles_per_extent": {"type": "int", "min": -1, "max": 1000000},
    "shock_z_threshold": {"type": "float", "min": 0.0, "max": 100.0},
    "bar_shock_z_threshold": {"type": "float", "min": 0.0, "max": 100.0},
    "max_adaptive_replays": {"type": "int", "min": 0, "max": 1024},
    "use_mechanical_veto": {"type": "bool"},
    "replay_coalescing": {"type": "bool"},
    "replay_coalescing_chunk_size": {"type": "int", "min": 1, "max": 4096},
}


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("expected boolean")


def _coerce_control_value(field_name: str, raw_value):
    spec = RUNTIME_CONTROL_SCHEMA[field_name]
    expected = spec["type"]
    if expected == "bool":
        value = _coerce_bool(raw_value)
    elif expected == "int":
        if isinstance(raw_value, bool):
            raise ValueError("expected integer")
        value = int(raw_value)
    elif expected == "float":
        if isinstance(raw_value, bool):
            raise ValueError("expected number")
        value = float(raw_value)
    else:
        raise ValueError(f"unsupported control type: {expected}")

    if "min" in spec and value < spec["min"]:
        raise ValueError(f"must be >= {spec['min']}")
    if "max" in spec and value > spec["max"]:
        raise ValueError(f"must be <= {spec['max']}")
    return value


def _training_config_snapshot(config: EpisodeTrainingConfig):
    cfg = asdict(config)
    # Keep the payload small but include fields the realtime controls and telemetry need.
    return {
        "optimizer_name": cfg.get("optimizer_name"),
        "learning_rate": cfg.get("learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
        "notional": cfg.get("notional"),
        "pair_width": cfg.get("pair_width"),
        "bar_sequences_per_episode": cfg.get("bar_sequences_per_episode"),
        "epochs": cfg.get("epochs"),
        "min_bar_window": cfg.get("min_bar_window"),
        "max_bar_window": cfg.get("max_bar_window"),
        "cache_size": cfg.get("cache_size"),
        "candles_per_extent": cfg.get("candles_per_extent"),
        "shock_z_threshold": cfg.get("shock_z_threshold"),
        "bar_shock_z_threshold": cfg.get("bar_shock_z_threshold"),
        "max_adaptive_replays": cfg.get("max_adaptive_replays"),
        "use_mechanical_veto": cfg.get("use_mechanical_veto"),
        "replay_coalescing": cfg.get("replay_coalescing"),
        "replay_coalescing_chunk_size": cfg.get("replay_coalescing_chunk_size"),
        "runtime_control_fields": sorted(RUNTIME_CONTROL_SCHEMA.keys()),
    }


RUNTIME_CONTROL_DEFAULTS = _training_config_snapshot(EpisodeTrainingConfig())


def _event_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time()%1)*1000):03d}"


def _append_stream_event(event_type: str, data):
    global event_stream_log, next_event_seq
    event = {
        "seq": next_event_seq,
        "type": event_type,
        "timestamp": _event_now_iso(),
        "data": data,
    }
    next_event_seq += 1
    with event_stream_cond:
        event_stream_log.append(event)
        if len(event_stream_log) > MAX_EVENT_STREAM_HISTORY:
            event_stream_log = event_stream_log[-MAX_EVENT_STREAM_HISTORY:]
        event_stream_cond.notify_all()
    return event


def _read_stream_events(cursor=None, max_events=64):
    with event_stream_cond:
        if cursor is None:
            events = event_stream_log[-max_events:]
        else:
            events = [e for e in event_stream_log if int(e.get("seq", 0)) >= cursor][:max_events]
        next_cursor = (events[-1]["seq"] + 1) if events else next_event_seq
        return list(events), next_cursor, next_event_seq


def _wait_for_stream_events(cursor, timeout_ms=15000, max_events=64):
    deadline = time.time() + (max(0, timeout_ms) / 1000.0)
    with event_stream_cond:
        while True:
            events = [e for e in event_stream_log if int(e.get("seq", 0)) >= cursor][:max_events]
            if events:
                next_cursor = events[-1]["seq"] + 1
                return list(events), next_cursor, False

            remaining = deadline - time.time()
            if remaining <= 0:
                return [], next_event_seq, True
            event_stream_cond.wait(timeout=remaining)


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
        while trainer_instance and (trainer_instance.running or not trainer_instance.event_queue.empty()):
            try:
                event_type, data = trainer_instance.event_queue.get(timeout=1.0)
                if event_type == 'episode_complete':
                    with state_lock:
                        latest_results.append(data)
                        if len(latest_results) > MAX_RESULTS_HISTORY:
                            latest_results = latest_results[-MAX_RESULTS_HISTORY:]
                    _append_stream_event(event_type, data)
                elif event_type == 'sample_event':
                    with state_lock:
                        latest_samples.append(data)
                        if len(latest_samples) > MAX_SAMPLES_HISTORY:
                            latest_samples = latest_samples[-MAX_SAMPLES_HISTORY:]
                    _append_stream_event(event_type, data)
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
        elif parsed_path.path == '/api/events':
            self.serve_api_events(parsed_path)
            return
        elif parsed_path.path == '/api/ws':
            self.serve_api_ws_info()
            return
        elif parsed_path.path == '/api/openapi.yaml':
            self.serve_api_openapi_yaml()
            return

        # Fallback to serving static files from /console (handled by super)
        super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/control':
            self.serve_api_control()
            return
        super().do_POST()

    def _send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def serve_api_openapi_yaml(self):
        if not OPENAPI_SPEC_FILE.exists():
            self._send_json({
                "ok": False,
                "error": "spec_not_found",
                "message": f"OpenAPI spec file not found: {OPENAPI_SPEC_FILE}",
            }, status=404)
            return

        try:
            payload = OPENAPI_SPEC_FILE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/yaml')
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self._send_json({
                "ok": False,
                "error": "spec_read_failed",
                "message": str(e),
            }, status=500)

    def _stream_transport_info(self):
        host = self.headers.get("Host", "localhost:8080")
        return {
            "longpoll_url": f"http://{host}/api/events",
            "websocket_url": f"ws://{host}/api/ws",
            "websocket_status": "reserved_not_implemented",
            "recommended_poll_ms": 1000,
            "default_longpoll_timeout_ms": 15000,
            "max_longpoll_timeout_ms": 60000,
        }

    def serve_api_events(self, parsed_path):
        query = parse_qs(parsed_path.query or "", keep_blank_values=False)

        mode = (query.get("mode", ["snapshot"])[0] or "snapshot").strip().lower()
        if mode not in {"snapshot", "longpoll"}:
            self._send_json({
                "ok": False,
                "error": "invalid_mode",
                "message": "mode must be one of: snapshot, longpoll",
            }, status=400)
            return

        def _parse_int_param(name, default, min_v=None, max_v=None):
            raw = query.get(name, [None])[0]
            if raw is None or raw == "":
                value = default
            else:
                try:
                    value = int(raw)
                except Exception:
                    raise ValueError(f"{name} must be an integer")
            if value is None:
                return None
            if min_v is not None and value < min_v:
                value = min_v
            if max_v is not None and value > max_v:
                value = max_v
            return value

        try:
            cursor = _parse_int_param("cursor", None, min_v=1) if "cursor" in query else None
            max_events = _parse_int_param("max_events", 64, min_v=1, max_v=500)
            timeout_ms = _parse_int_param("timeout_ms", 15000, min_v=0, max_v=60000)
        except ValueError as e:
            self._send_json({
                "ok": False,
                "error": "invalid_query",
                "message": str(e),
            }, status=400)
            return

        if mode == "longpoll":
            wait_cursor = cursor if cursor is not None else _read_stream_events(cursor=None, max_events=1)[2]
            events, next_cursor_out, timed_out = _wait_for_stream_events(
                cursor=wait_cursor,
                timeout_ms=timeout_ms,
                max_events=max_events,
            )
            cursor_received = wait_cursor
        else:
            events, next_cursor_out, _stream_head = _read_stream_events(cursor=cursor, max_events=max_events)
            timed_out = False
            cursor_received = cursor

        response = {
            "ok": True,
            "mode": mode,
            "cursor_received": cursor_received,
            "next_cursor": next_cursor_out,
            "event_count": len(events),
            "timed_out": timed_out,
            "events": events,
            "transports": self._stream_transport_info(),
        }
        self._send_json(response, status=200)

    def serve_api_ws_info(self):
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if upgrade == "websocket":
            self._send_json({
                "ok": False,
                "error": "websocket_not_implemented",
                "message": "WebSocket upgrade endpoint is reserved but not implemented in trainerd.py yet. Use /api/events in longpoll mode.",
                "transports": self._stream_transport_info(),
            }, status=501)
            return

        self.send_response(426)
        self.send_header("Content-Type", "application/json")
        self.send_header("Upgrade", "websocket")
        self.end_headers()
        self.wfile.write(json.dumps({
            "ok": False,
            "error": "upgrade_required",
            "message": "Use a WebSocket client for /api/ws, or use /api/events?mode=longpoll for HTTP streaming.",
            "transports": self._stream_transport_info(),
        }).encode("utf-8"))

    def serve_api_state(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        global trainer_instance
        if not trainer_instance:
            self.wfile.write(json.dumps({
                "status": "booting",
                "runtime_control_defaults": RUNTIME_CONTROL_DEFAULTS,
                "stream_transports": self._stream_transport_info(),
            }).encode('utf-8'))
            return

        with state_lock:
            # Safely capture top level stats
            response_data = {
                "status": "running" if trainer_instance.running else "stopped",
                "session_start_time": trainer_instance.session_start_time,
                "training_config": _training_config_snapshot(trainer_instance.config),
                "runtime_control_defaults": RUNTIME_CONTROL_DEFAULTS,
                "stream_transports": self._stream_transport_info(),
                "history": latest_results, # Last N completed episodes
                "samples": latest_samples, # Last N sampling events
            }

            # If there's at least one result, attach a subset of the first/last elements to build global metrics easily
            if latest_results:
                response_data["latest_metrics"] = {
                    "total_trained": len(trainer_instance.results),
                    "current_capital": latest_results[-1].get("final_capital", latest_results[-1].get("capital", 0.0)),
                    "total_realized_pnl": sum([r.get('realized_pnl', 0.0) for r in trainer_instance.results if 'realized_pnl' in r]),
                }

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def serve_api_control(self):
        global trainer_instance
        if not trainer_instance:
            self._send_json({
                "ok": False,
                "error": "trainer_not_ready",
                "message": "Trainer instance has not started yet.",
            }, status=503)
            return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"

        try:
            payload = json.loads(post_data)
        except Exception:
            payload = {}

        if not isinstance(payload, dict):
            self._send_json({
                "ok": False,
                "error": "invalid_payload",
                "message": "Expected a JSON object or {\"updates\": {...}} payload.",
            }, status=400)
            return

        updates = payload.get("updates", payload)
        if not isinstance(updates, dict):
            self._send_json({
                "ok": False,
                "error": "invalid_payload",
                "message": "Expected a JSON object or {\"updates\": {...}} payload.",
            }, status=400)
            return

        allowed_fields = set(RUNTIME_CONTROL_SCHEMA.keys())
        errors = {}
        staged = {}

        for key, raw_value in updates.items():
            if key not in allowed_fields:
                errors[key] = "unsupported field"
                continue
            try:
                staged[key] = _coerce_control_value(key, raw_value)
            except Exception as e:
                errors[key] = str(e)

        # Cross-field validation against current config values.
        with state_lock:
            current_cfg = trainer_instance.config
            min_window = int(staged.get("min_bar_window", getattr(current_cfg, "min_bar_window", 64)))
            max_window = int(staged.get("max_bar_window", getattr(current_cfg, "max_bar_window", 256)))
            if min_window > max_window:
                errors["min_bar_window"] = "must be <= max_bar_window"
                errors["max_bar_window"] = "must be >= min_bar_window"

        if errors:
            self._send_json({
                "ok": False,
                "error": "validation_failed",
                "errors": errors,
                "training_config": _training_config_snapshot(trainer_instance.config),
            }, status=400)
            return

        warnings = []
        applied = {}
        with state_lock:
            cfg = trainer_instance.config
            for key, value in staged.items():
                setattr(cfg, key, value)
                applied[key] = value
                if key == "cache_size":
                    try:
                        trainer_instance.candle_cache.max_size = int(value)
                    except Exception:
                        warnings.append("cache_size updated in config but cache runtime object was not updated")

        if any(k in staged for k in ("optimizer_name", "learning_rate", "weight_decay")):
            warnings.append("optimizer changes may require trainer reinitialization to take effect")

        self._send_json({
            "ok": True,
            "applied": applied,
            "warnings": warnings,
            "training_config": _training_config_snapshot(trainer_instance.config),
            "message": "Runtime controls applied to trainer daemon.",
        }, status=200)

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
