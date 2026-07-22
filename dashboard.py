#!/usr/bin/env python3
"""
Training Dashboard (Streamlit)
==============================

Run with: streamlit run dashboard.py

Real-time visualization of 500 episode training runs.
Does not perform any training itself—merely a passive observer.
"""

import sys
import os
import threading
import queue
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from train import EpisodeTrainingConfig, EpochEpisodeTrainer


CHECKPOINT_FILE = Path("training_checkpoint.json")
RESULTS_FILE    = Path("training_results.json")
DRAWTHRU_DUCKDB_FILE = Path("data/binance/hrm_data.duckdb")


def _load_drawthru_snapshot():
    """
    Read a small DuckDB snapshot for immediate dashboard content before training results exist.

    Prefers `data/binance/hrm_data.duckdb.binance_sequences_import` which is populated from
    local parquet imports. Falls back to `market_data` if available.
    """
    try:
        import duckdb
    except Exception as e:
        return {"status": "unavailable", "error": f"duckdb import failed: {e}"}

    if not DRAWTHRU_DUCKDB_FILE.exists():
        return {"status": "missing", "db_path": str(DRAWTHRU_DUCKDB_FILE)}

    try:
        con = duckdb.connect(str(DRAWTHRU_DUCKDB_FILE), read_only=True)
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "binance_sequences_import" in tables:
            table_name = "binance_sequences_import"
            where_clause = ""
        elif "market_data" in tables:
            table_name = "market_data"
            where_clause = "WHERE lower(coalesce(exchange, '')) = 'binance'"
        else:
            con.close()
            return {
                "status": "empty",
                "db_path": str(DRAWTHRU_DUCKDB_FILE),
                "tables": sorted(tables),
            }

        def run_query(query, params=None, as_df=False):
            """Helper to run a query in its own DuckDB connection for thread-safety."""
            import duckdb
            # Use the global DRAWTHRU_DUCKDB_FILE
            conn = duckdb.connect(str(DRAWTHRU_DUCKDB_FILE), read_only=True)
            try:
                res = conn.execute(query, params)
                return res.df() if as_df else res.fetchone()
            finally:
                conn.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_stats = executor.submit(run_query, f"""
                SELECT
                  COUNT(*) AS row_count,
                  COUNT(DISTINCT symbol) AS symbol_count,
                  MIN(timestamp) AS min_ts,
                  MAX(timestamp) AS max_ts
                FROM {table_name}
                {where_clause}
            """)

            f_top = executor.submit(run_query, f"""
                SELECT
                  symbol,
                  COUNT(*) AS row_count,
                  MAX(timestamp) AS last_ts
                FROM {table_name}
                {where_clause}
                GROUP BY symbol
                ORDER BY row_count DESC, symbol ASC
                LIMIT 12
            """, as_df=True)

            f_preview_row = executor.submit(run_query, f"""
                SELECT symbol, MAX(timestamp) AS last_ts
                FROM {table_name}
                {where_clause}
                GROUP BY symbol
                ORDER BY last_ts DESC, symbol ASC
                LIMIT 1
            """)

            row_count, symbol_count, min_ts, max_ts = f_stats.result()
            top_symbols_df = f_top.result()
            preview_symbol_row = f_preview_row.result()

        preview_rows = []
        if preview_symbol_row:
            preview_symbol = preview_symbol_row[0]
            preview_df = con.execute(
                f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                {where_clause}
                {"AND" if where_clause else "WHERE"} symbol = ?
                ORDER BY timestamp DESC
                LIMIT 120
                """,
                [preview_symbol],
            ).df()
            if not preview_df.empty:
                preview_df = preview_df.sort_values("timestamp")
                preview_rows = preview_df.to_dict(orient="records")
        else:
            preview_symbol = None
        con.close()

        return {
            "status": "ok",
            "db_path": str(DRAWTHRU_DUCKDB_FILE),
            "table": table_name,
            "row_count": int(row_count or 0),
            "symbol_count": int(symbol_count or 0),
            "min_ts": None if min_ts is None else str(min_ts),
            "max_ts": None if max_ts is None else str(max_ts),
            "top_symbols": top_symbols_df.to_dict(orient="records"),
            "preview_symbol": preview_symbol,
            "preview_rows": preview_rows,
        }
    except Exception as e:
        return {
            "status": "error",
            "db_path": str(DRAWTHRU_DUCKDB_FILE),
            "error": str(e),
        }


def load_drawthru_snapshot():
    # Avoid hammering DuckDB each Streamlit rerun.
    if hasattr(st, "cache_data"):
        @st.cache_data(ttl=5, show_spinner=False)
        def _cached():
            return _load_drawthru_snapshot()
        return _cached()
    return _load_drawthru_snapshot()


def load_cli_checkpoint():
    """Return list of episode result dicts from the CLI trainer checkpoint, or []."""
    import json
    for p in (CHECKPOINT_FILE, RESULTS_FILE):
        if p.exists():
            try:
                data = json.loads(p.read_text())
                results = data.get('results', [])
                if results:
                    return results, data
            except Exception:
                pass
    return [], {}


def init_session_state():
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    if 'current_epoch_info' not in st.session_state:
        st.session_state.current_epoch_info = None
    if 'config' not in st.session_state:
        st.session_state.config = {}
    if 'cli_results' not in st.session_state:
        st.session_state.cli_results = []
    if 'cli_checkpoint_meta' not in st.session_state:
        st.session_state.cli_checkpoint_meta = {}


def main():
    st.set_page_config(
        page_title="MoneyFan Training",
        page_icon="💰",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("💰 MoneyFan Training Dashboard")
    st.markdown("### Unified Hierarchical Codec Training System")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        n_episodes = st.number_input("Epoch Episodes", min_value=1, value=1, key="n_episodes_input")
        notional_val = st.number_input("Starting Notional ($)", min_value=10, value=100, key="capital_input")
        pair_width_val = st.number_input("Pair Width (symbols)", min_value=5, value=30, key="pair_width_input")
        epochs = st.number_input("Epochs per Basket", min_value=1, value=1, key="epochs_input")
        per_extent_length = st.number_input("Extent Length (candles)", min_value=-1, value=1000, help="-1 for no limit", key="per_extent_length_input")
        
        st.divider()
        st.subheader("Sub-Basket Outliers")
        extent_outlier_z = st.slider("Extent Outlier Z-Score", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="extent_outlier_z_input")
        max_optimizer_replays = st.slider("Max Optimizer Replays", min_value=1, max_value=10, value=3, key="max_optimizer_replays_input")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                if not st.session_state.training_active:
                    config = EpisodeTrainingConfig(
                        n_epoch_episodes=n_episodes,
                        notional=notional_val,
                        pair_width=pair_width_val,
                        epochs=epochs,
                        candles_per_extent=per_extent_length,
                        shock_z_threshold=extent_outlier_z,
                        max_adaptive_replays=max_optimizer_replays
                    )
                    
                    st.session_state.trainer = EpochEpisodeTrainer(config)
                    st.session_state.config = config
                    st.session_state.results = []
                    st.session_state.training_active = True
                    
                    thread = threading.Thread(
                        target=st.session_state.trainer.run_episode_training,
                        daemon=True
                    )
                    thread.start()
                    st.session_state.training_thread = thread
                    
                    st.success("Training started!")
        
        with col2:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                if st.session_state.trainer:
                    st.session_state.trainer.running = False
                    st.session_state.training_active = False
                    st.warning("Training stopped")
    
    if st.session_state.trainer and st.session_state.training_active:
        while True:
            try:
                event_type, data = st.session_state.trainer.event_queue.get_nowait()
                if event_type == 'episode_complete':
                    st.session_state.results.append(data)
                elif event_type == 'epoch_complete':
                    st.session_state.current_epoch_info = data
            except queue.Empty:
                break
        
        if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
            st.session_state.training_active = False

    # ── CLI trainer passthrough ────────────────────────────────────────────────
    # Poll the checkpoint written by the background `python train.py` process.
    # This makes the dashboard a live observer even without clicking Start.
    cli_results, cli_meta = load_cli_checkpoint()
    if cli_results:
        st.session_state.cli_results = cli_results
        st.session_state.cli_checkpoint_meta = cli_meta
        completed_cli = cli_meta.get('completed_episodes', len(cli_results))
        total_cli     = cli_meta.get('total_episodes', 500)
        st.info(
            f"🖥️  **External CLI trainer detected** — "
            f"{completed_cli}/{total_cli} episodes completed "
            f"(`python train.py` → `training_checkpoint.json`)"
        )

    # ── Drawthru / DuckDB health snapshot ────────────────────────────────────
    drawthru = load_drawthru_snapshot()
    if drawthru.get("status") == "ok":
        st.subheader("🧭 Drawthru Data Health (DuckDB)")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.metric("DuckDB Rows", f"{drawthru.get('row_count', 0):,}")
        with dcol2:
            st.metric("Symbols", f"{drawthru.get('symbol_count', 0)}")
        with dcol3:
            st.metric("Latest Timestamp", drawthru.get("max_ts", "--"))
        with dcol4:
            st.metric("Table", drawthru.get("table", "--"))

        top_symbols = pd.DataFrame(drawthru.get("top_symbols", []))
        if not top_symbols.empty:
            dleft, dright = st.columns([2, 1])
            with dleft:
                fig_draw = px.bar(
                    top_symbols.sort_values("row_count", ascending=False),
                    x="symbol",
                    y="row_count",
                    title="Imported Rows by Symbol (Top 12)",
                )
                fig_draw.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_draw, use_container_width=True)
            with dright:
                st.caption(f"Source: `{drawthru.get('db_path', '')}`")
                st.dataframe(
                    top_symbols[["symbol", "row_count", "last_ts"]],
                    use_container_width=True,
                    hide_index=True,
                    height=260,
                )

        preview_rows = pd.DataFrame(drawthru.get("preview_rows", []))
        preview_symbol = drawthru.get("preview_symbol")
        if not preview_rows.empty and {"timestamp", "open", "high", "low", "close"}.issubset(preview_rows.columns):
            st.markdown(f"#### 📈 Live DuckDB Candle Preview ({preview_symbol})")
            preview_rows["timestamp"] = pd.to_datetime(preview_rows["timestamp"])
            fig_candle = go.Figure(
                data=[
                    go.Candlestick(
                        x=preview_rows["timestamp"],
                        open=preview_rows["open"],
                        high=preview_rows["high"],
                        low=preview_rows["low"],
                        close=preview_rows["close"],
                        name=preview_symbol or "preview",
                    )
                ]
            )
            fig_candle.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title=None,
                yaxis_title=None,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_candle, use_container_width=True)
        st.divider()
    elif drawthru.get("status") in {"missing", "error"}:
        st.warning(
            f"Drawthru DuckDB unavailable: {drawthru.get('error', drawthru.get('db_path', 'missing'))}"
        )
        st.divider()

    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completed = len(st.session_state.results)
        st.metric("Episodes Trained", f"{completed}/{n_episodes} (Size: {pair_width_val})", f"{completed/n_episodes*100:.1f}%")
    
    with col2:
        if st.session_state.results:
            total_pnl = sum(r.get('realized_pnl', 0.0) for r in st.session_state.results)
            st.metric("Total Realized PnL", f"${total_pnl:.2f}")
        else:
            st.metric("Total Realized PnL", "$0.00")
    
    with col3:
        if st.session_state.results:
            pnl_vals = [r.get('realized_pnl', 0.0) for r in st.session_state.results]
            avg_pnl = np.nanmean(pnl_vals) if pnl_vals else 0.0
            avg_pnl = 0.0 if np.isnan(avg_pnl) else avg_pnl
            st.metric("Avg Realized PnL/Episode", f"${avg_pnl:.2f}")
        else:
            st.metric("Avg Realized PnL/Episode", "$0.00")
    
    with col4:
        if st.session_state.results:
            wr_vals = [r.get('hit_rate', 0.0) for r in st.session_state.results]
            avg_wr = np.nanmean(wr_vals) if wr_vals else 0.0
            avg_wr = 0.0 if np.isnan(avg_wr) else avg_wr
            st.metric("Avg Hit Rate", f"{avg_wr:.1%}")
        else:
            st.metric("Avg Hit Rate", "0%")
            
    st.divider()
    
    if st.session_state.current_epoch_info and st.session_state.training_active:
        info = st.session_state.current_epoch_info
        st.subheader("🔄 Current Training Progress")
        
        ep_col1, ep_col2, ep_col3, ep_col4, ep_col5, ep_col6 = st.columns(6)
        with ep_col1:
            st.metric("Current Episode", f"#{info.get('episode_id', 0)}")
            st.caption(f"Symbols: {', '.join(info.get('symbols', [])[:3])}")
        with ep_col2:
            st.metric("Epoch", f"{info.get('epoch', 0)} / {info.get('total_epochs', 0)}")
        with ep_col3:
            st.metric("Episode PnL / HR", f"${info.get('realized_pnl', 0.0):.2f}", f"{info.get('hit_rate', 0.0):.1%}")
        with ep_col4:
            st.metric("Winning Agent", f"{info.get('winning_agent', 'N/A')}")
        with ep_col5:
            st.metric("HRM Score", f"{info.get('hrm_score', 0.0):.3f}")
        with ep_col6:
            st.metric("Pred Loss", f"{info.get('predictor_loss', 0.0):.4f}")
            
        progress = info.get('epoch', 0) / max(info.get('total_epochs', 1), 1)
        st.progress(progress)
    
    st.divider()

    # ── 24-Agent Leaderboard ─────────────────────────────────────────────────
    # Aggregate codec_scores across all available episodes (own trainer or CLI)
    active_results = st.session_state.results or st.session_state.cli_results
    all_codec_scores: Dict[str, float] = {}
    for r in active_results:
        for agent, score in r.get('codec_scores', {}).items():
            all_codec_scores[agent] = all_codec_scores.get(agent, 0.0) + score

    if all_codec_scores:
        st.subheader("🏆 24-Agent Leaderboard")
        lb_df = (
            pd.DataFrame.from_dict(all_codec_scores, orient='index', columns=['total_conviction'])
            .sort_values('total_conviction', ascending=True)     # ascending for horizontal bar
            .reset_index()
            .rename(columns={'index': 'agent'})
        )
        top_score = lb_df['total_conviction'].max()
        lb_df['pct'] = (lb_df['total_conviction'] / max(top_score, 1e-8)) * 100

        fig_lb = go.Figure(go.Bar(
            x=lb_df['total_conviction'],
            y=lb_df['agent'],
            orientation='h',
            marker=dict(
                color=lb_df['pct'],
                colorscale='Viridis',
                showscale=False,
            ),
            text=[f"{v:.1f}" for v in lb_df['total_conviction']],
            textposition='outside',
        ))
        fig_lb.update_layout(
            xaxis_title='Cumulative Conviction Score',
            yaxis_title='',
            height=max(400, len(lb_df) * 22),
            margin=dict(l=0, r=80, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        # Win-count column: how many episodes each agent was top scorer
        win_counts: Dict[str, int] = {}
        for r in active_results:
            wa = r.get('winning_agent')
            if wa:
                win_counts[wa] = win_counts.get(wa, 0) + 1
        if win_counts:
            wc_df = (
                pd.DataFrame.from_dict(win_counts, orient='index', columns=['episode_wins'])
                .sort_values('episode_wins', ascending=False)
                .reset_index()
                .rename(columns={'index': 'agent'})
            )
            st.caption(f"Episode wins — top agent: **{wc_df.iloc[0]['agent']}** "
                       f"({wc_df.iloc[0]['episode_wins']} wins / {len(active_results)} episodes)")
            st.dataframe(wc_df, use_container_width=True, hide_index=True)

    st.divider()

    if st.session_state.results:

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Cumulative Realized PnL")
            
            df = pd.DataFrame(st.session_state.results)
            # Defensive check if all episodes so far failed/errored out
            if not df.empty and 'realized_pnl' not in df.columns:
                df['realized_pnl'] = 0.0
            
            df['cumulative_pnl'] = df['realized_pnl'].fillna(0.0).cumsum() if not df.empty else pd.Series([0.0])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative Realized PnL',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                xaxis_title='Episode Number',
                yaxis_title='Cumulative Realized PnL ($)',
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Hit Rate Distribution")
            
            fig = go.Figure()
            
            if not df.empty and 'hit_rate' not in df.columns:
                df['hit_rate'] = 0.0
                
            hit_rates = df['hit_rate'].dropna() if not df.empty else [0.0]
            
            fig.add_trace(go.Histogram(
                x=hit_rates,
                nbinsx=20,
                marker_color='#00ff88',
                opacity=0.7
            ))
            
            fig.update_layout(
                xaxis_title='Hit Rate',
                yaxis_title='Count',
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("🌐 Breadth Score (Regime Coverage)")
        
        if not df.empty:
            if 'breadth_score' not in df.columns:
                # Fallback to zero if the trainer does not compute breadth
                df['breadth_score'] = 0.0

            fig_breadth = go.Figure()
            fig_breadth.add_trace(go.Scatter(
                y=df['breadth_score'],
                mode='lines+markers',
                name='Breadth Score',
                line=dict(color='#0088ff', width=2),
                marker=dict(size=4)
            ))
            fig_breadth.update_layout(
                xaxis_title='Episode Number',
                yaxis_title='Breadth Score (Coverage)',
                yaxis=dict(range=[0, 1]),
                height=300,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_breadth, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Realized PnL per Episode")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['realized_pnl'],
                mode='markers',
                name='Realized PnL',
                marker=dict(
                    color=df['realized_pnl'],
                    colorscale='RdYlGn',
                    size=8
                )
            ))
            
            fig.update_layout(
                xaxis_title='Episode Number',
                yaxis_title='Realized PnL ($)',
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎲 Capital Growth")
            
            fig = go.Figure()
            if not df.empty and 'final_capital' not in df.columns:
                target_capital = 100.0
                if 'config' in st.session_state and hasattr(st.session_state.config, 'capital'):
                    target_capital = float(st.session_state.config.capital)
                elif 'config' in st.session_state and isinstance(st.session_state.config, dict):
                    target_capital = float(st.session_state.config.get('capital', 100))
                df['final_capital'] = target_capital
                
            fig.add_trace(go.Scatter(
                y=df['final_capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                y=[100] * len(df),
                mode='lines',
                name='Starting Capital',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title='Episode Number',
                yaxis_title='Capital ($)',
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(x=0, y=1, xanchor='left', yanchor='top')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Recent Results")
        
        # Defensive schema enforcement to prevent KeyErrors on errored episodes
        required_cols = [
            'episode_id', 'symbols', 'final_capital', 'realized_pnl', 'hit_rate', 
            'winning_agent', 'hrm_score', 'predictor_loss', 
            'outlier_extents', 'optimizer_replays', 'total_trades'
        ]
        for col in required_cols:
            if col not in df.columns:
                if col == 'symbols':
                    df[col] = [[]] * len(df)
                elif col in ['outlier_extents', 'optimizer_replays', 'total_trades', 'episode_id']:
                    df[col] = 0
                elif col == 'final_capital':
                    df[col] = 100.0
                elif col in ['realized_pnl', 'hit_rate', 'hrm_score', 'predictor_loss']:
                    df[col] = 0.0
                else:
                    df[col] = "N/A"
        
        display_df = df[['episode_id', 'symbols', 'final_capital', 'realized_pnl', 'hit_rate', 'winning_agent', 'hrm_score', 'predictor_loss', 'outlier_extents', 'optimizer_replays', 'total_trades']].tail(20)
        display_df = display_df.copy()
        display_df['symbols'] = display_df['symbols'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        display_df = display_df.round(3)
        
        st.dataframe(
            display_df.drop(columns=['equity_curve'], errors='ignore'),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        st.subheader("🔍 Micro Replay (3D Realtime Spark)")
        
        # User selector for episode drill-down
        with st.expander("🔍 Deep Dive: Replay Specific Episodes"):
            # Get valid episodes that actually track an equity curve > 1 step
            valid_episodes = [r for r in st.session_state.results if 'equity_curve' in r and len(r['equity_curve']) > 1]
            
            if valid_episodes:
                # Sort by PnL to help rank discovery
                ranked_episodes = sorted(valid_episodes, key=lambda x: x.get('realized_pnl', 0), reverse=True)
                
                # Format options for the selectbox
                options = {
                    f"#{e['episode_id']} (PnL: ${e.get('realized_pnl', 0):.2f})": e
                    for e in ranked_episodes
                }
                
                selected_label = st.selectbox(
                    "Drill down into specific episode:",
                    options=list(options.keys())
                )
                
                if selected_label:
                    selected_episode = options[selected_label]
                    # Calculate sequence step X-axis
                    x_vals = list(range(len(eq_curve)))
                    
                    # 1. 2D Portfolio Health (True Equity Curve + Drawdown)
                    st.markdown("#### 📉 Portfolio Health (True Equity & Drawdown)")
                    
                    fig_eq = go.Figure()
                    
                    # Main Equity line
                    fig_eq.add_trace(go.Scatter(
                        x=x_vals,
                        y=eq_curve,
                        mode='lines',
                        name='HRM Capital ($)',
                        line=dict(color='#00ff88', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 136, 0.1)'
                    ))
                    
                    # Buy and Hold Benchmark (if available, otherwise flat line)
                    bh_curve = selected_episode.get('benchmark_curve', [selected_episode.get('final_capital', 100)] * len(eq_curve))
                    if len(bh_curve) == len(eq_curve):
                        fig_eq.add_trace(go.Scatter(
                            x=x_vals, y=bh_curve, mode='lines',
                            name='Buy & Hold (Equally Weighted)',
                            line=dict(color='gray', width=1, dash='dash')
                        ))

                    fig_eq.update_layout(
                        title=f"Episode #{selected_episode['episode_id']} Equity Curve",
                        xaxis_title="Bar/Step",
                        yaxis_title="Notional Value ($)",
                        height=350, margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    # Compute Drawdown
                    eq_arr = np.array(eq_curve)
                    peak_arr = np.maximum.accumulate(eq_arr)
                    dd_pct = np.zeros_like(eq_arr)
                    valid_idx = peak_arr > 0
                    dd_pct[valid_idx] = (eq_arr[valid_idx] - peak_arr[valid_idx]) / peak_arr[valid_idx]

                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=x_vals, y=dd_pct, mode='lines', name='Drawdown',
                        line=dict(color='#ff3366', width=1),
                        fill='tozeroy', fillcolor='rgba(255, 51, 102, 0.3)'
                    ))
                    fig_dd.update_layout(
                        title="Underwater Curve (Drawdown Depth)",
                        xaxis_title="Bar/Step",
                        yaxis_title="Drawdown (%)",
                        yaxis_tickformat='.1%',
                        height=200, margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

                    # 2. Episode Predictor Truth Scatter (if available)
                    # If this episode logged raw trades/bars (in future trainer iter), show them
                    trades = selected_episode.get('trades', [])
                    if trades and isinstance(trades, list) and len(trades) > 0 and 'hrm_score' in trades[0]:
                        st.markdown("#### ⚖️ Predictor Truth (Score vs PnL)")

                        scores = [t.get('hrm_score', 0) for t in trades]
                        pnls = [t.get('pnl', 0) for t in trades]

                        fig_scatter = px.scatter(
                            x=scores, y=pnls,
                            color=np.sign(pnls),
                            color_continuous_scale=['#ff3366', 'gray', '#00ff88'],
                            labels={'x': 'HRM Conviction Score', 'y': 'Realised Trade PnL ($)', 'color': 'PnL Sign'}
                        )
                        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_scatter.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                    # 3. Codec Allocator Imprint (Which agents won this pair basket?)
                    if 'codec_scores' in selected_episode:
                        st.markdown("#### 🧠 Allocator Weights (Episode Conviction)")
                        cs = selected_episode['codec_scores']
                        if cs:
                            cs_df = pd.DataFrame(list(cs.items()), columns=['Agent', 'Total Conviction']).sort_values('Total Conviction', ascending=True)
                            fig_bar = px.bar(
                                cs_df, x='Total Conviction', y='Agent', orientation='h',
                                color='Total Conviction', color_continuous_scale='Viridis'
                            )
                            fig_bar.update_layout(height=max(300, len(cs)*20), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.info("No detailed equity curves available for replay yet.")
                
        st.divider()
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cap = df['final_capital'].sum() if not df.empty else 0.0
            std_pnl = df['realized_pnl'].std() if not df.empty else 0.0
            std_pnl = 0.0 if pd.isna(std_pnl) else std_pnl
            st.metric("Total Capital", f"${total_cap:.2f}")
            st.metric("Std Realized PnL", f"${std_pnl:.2f}")
        
        with col2:
            if not df.empty:
                winning = df[df['realized_pnl'] > 0]
                st.metric("Winning Episodes", f"{len(winning)}/{len(df)}")
                st.metric("Winning %", f"{len(winning)/max(len(df), 1)*100:.1f}%")
            else:
                st.metric("Winning Episodes", "0/0")
                st.metric("Winning %", "0.0%")
        
        with col3:
            if not df.empty and len(winning := df[df['realized_pnl'] > 0]) > 0:
                st.metric("Best Episode PnL", f"${winning['realized_pnl'].max():.2f}")
            else:
                st.metric("Best Episode PnL", "$0.00")
            
            if not df.empty and len(losing := df[df['realized_pnl'] < 0]) > 0:
                st.metric("Worst Episode PnL", f"${losing['realized_pnl'].min():.2f}")
            else:
                st.metric("Worst Episode PnL", "$0.00")
    
    else:
        # Show CLI trainer results as a fallback if the dashboard hasn't started its own run
        if st.session_state.cli_results:
            cli_df = pd.DataFrame(st.session_state.cli_results)
            st.subheader("📡 Live Results from CLI Trainer (`train.py`)")
            required_cols = ['episode_id', 'realized_pnl', 'hit_rate', 'total_trades',
                             'predictor_loss', 'outlier_extents', 'optimizer_replays']
            for col in required_cols:
                if col not in cli_df.columns:
                    cli_df[col] = 0
            cli_df['cumulative_pnl'] = cli_df['realized_pnl'].fillna(0.0).cumsum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Episodes", len(cli_df))
            m2.metric("Cumulative PnL", f"${cli_df['cumulative_pnl'].iloc[-1]:.2f}")
            m3.metric("Avg Hit Rate", f"{cli_df['hit_rate'].mean():.1%}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=cli_df['cumulative_pnl'], mode='lines',
                line=dict(color='#00ff88', width=2)
            ))
            fig.update_layout(
                xaxis_title='Episode', yaxis_title='Cumul. PnL ($)',
                height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                cli_df[required_cols].tail(20).round(3),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No training results yet. Click 'Start' to begin, or run `python train.py` in a terminal.")
    
    if st.session_state.training_active or CHECKPOINT_FILE.exists():
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
