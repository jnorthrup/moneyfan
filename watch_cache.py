#!/usr/bin/env python3
"""
Aria2c Download & DuckDB Cache Watcher
======================================

Animates the status of the singleton DuckDB cache and currently 
active aria2c downloads in the terminal.

Prerequisites:
  pip install aria2p rich
"""

import os
import time
from pathlib import Path

try:
    import aria2p
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich import box
except ImportError:
    print("Missing requirements. Run: pip install aria2p rich")
    exit(1)

# Path to the duckdb files and parquet directory
DUCKDB_PATHS = [
    Path("hrm/data/market.duckdb"),
    Path("hrm/data/coinbase.duckdb")
]

PARQUET_DIR = Path("hrm/data/parquet")

def get_file_size_formatted(path: Path) -> str:
    if not path.exists():
        return "0 B (Missing)"
    size = path.stat().st_size
    s = size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if s < 1024.0:
            return f"{s:.2f} {unit}"
        s /= 1024.0
    return f"{s:.2f} PB"

def get_cache_table() -> Table:
    table = Table(title="🪿 Singleton DuckDB Cache", box=box.ROUNDED, expand=True)
    table.add_column("Resource", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    # DuckDB files
    for db_path in DUCKDB_PATHS:
        table.add_row(f"🪙  {db_path.name}", get_file_size_formatted(db_path))
    
    # Parquet directory
    if PARQUET_DIR.exists():
        total_size = sum(f.stat().st_size for f in PARQUET_DIR.glob("**/*") if f.is_file())
        size_str = "0 B"
        s = total_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if s < 1024.0:
                size_str = f"{s:.2f} {unit}"
                break
            s /= 1024.0
        
        file_count = len(list(PARQUET_DIR.glob("**/*.parquet")))
        table.add_row("📦 Parquet Store", f"{size_str} ({file_count} files)")
    else:
        table.add_row("📦 Parquet Store", "Missing")
        
    return table

def get_aria2c_table() -> Table:
    table = Table(title="⬇️  Aria2p Downloads", box=box.ROUNDED, expand=True)
    table.add_column("GID", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Progress")
    table.add_column("Speed", justify="right")
    table.add_column("ETA", justify="right")
    
    try:
        # Initialize aria2p API (assumes default localhost:6800 RPC)
        client = aria2p.Client(
            host="http://localhost",
            port=6800,
            secret=""
        )
        api = aria2p.API(client)
        
        downloads = api.get_downloads()
        
        if not downloads:
            table.add_row("-", "No active downloads in RPC queue.", "-", "-", "-")
            return table
            
        for dl in downloads:
            # Format progress
            if dl.total_length == 0:
                prog_str = "0%"
            else:
                prog_str = f"{dl.progress:.1f}% ({dl.completed_length_string()} / {dl.total_length_string()})"
            
            table.add_row(
                dl.gid[:6] + "…",
                dl.name[:40] + ("…" if len(dl.name) > 40 else ""),
                prog_str,
                dl.download_speed_string(),
                dl.eta_string()
            )
            
    except Exception as e:
        table.add_row("-", f"[red]RPC Error:[/red] Is aria2c --enable-rpc running? ({type(e).__name__})", "-", "-", "-")
        
    return table

def generate_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=1),
        Layout(name="lower", ratio=2)
    )
    
    # Render tables directly
    layout["upper"].update(Panel(get_cache_table(), border_style="blue"))
    layout["lower"].update(Panel(get_aria2c_table(), border_style="magenta"))
    return layout

if __name__ == "__main__":
    print("Starting Watcher... (Ctrl+C to exit)")
    
    try:
        with Live(generate_layout(), refresh_per_second=4) as live:
            while True:
                time.sleep(0.5)
                live.update(generate_layout())
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
