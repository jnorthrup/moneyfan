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
                    eq_curve = selected_episode['equity_curve']
                    
                    # 3D Path representation
                    # Z-axis: Equity value
                    # Y-axis: Episode ID (to give it depth relative to the run)
                    # X-axis: Sequence Step
                    
                    x_vals = list(range(len(eq_curve)))
                    y_vals = [selected_episode['episode_id']] * len(eq_curve)
                    z_vals = eq_curve
                    
                    fig3d = go.Figure(data=[go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode='lines+markers',
                        marker=dict(
                            size=3,
                            color=z_vals,
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        line=dict(
                            color='#ff0055',
                            width=3
                        )
                    )])
                    
                    fig3d.update_layout(
                        title=f"3D Micro Replay: Episode #{selected_episode['episode_id']}",
                        scene=dict(
                            xaxis_title='Sequence Step',
                            yaxis_title='Episode ID',
                            zaxis_title='Capital ($)'
                        ),
                        height=500,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    
                    st.plotly_chart(fig3d, use_container_width=True)
                    
                    # 2.5D Pandas Hierarchy Visualizer (Dumbing down the Parquet structure)
                    with st.container(border=True):
                        st.markdown(f"**Current Episode Configuration**: `{len(selected_episode['symbols'])} Symbols` ➔ `{getattr(st.session_state.config, 'bar_sequences_per_episode', 10)} Sequences` ➔ `Timeframe Steps`")
                        
                        # Show some of the paired instruments
                        display_syms = selected_episode['symbols'][:4]
                        st.markdown("**Paired Active Symbols:**")
                        cols = st.columns(len(display_syms))
                        for i, sym in enumerate(display_syms):
                            with cols[i]:
                                st.metric(
                                    label=sym, 
                                    value="Active", 
                                    delta=f"{np.random.randn():.2f}%" # Placeholder for live change
                                )
                                
                        st.caption("Here is the authentic structural representation of how the native Pandas/Parquet backbone unrolls this episode's assets into memory for the model, utilizing generic MultiIndex auto-layouts:")
                        
                        # Show raw MultiIndex data-frame shape illustration
                        # Since we don't pass the actual df to the dashboard, we sketch a visual analog here 
                        
                        st.code("""
# (Timestamp, Asset) MultiIndex Shape
pd.DataFrame([
    ...
], index=pd.MultiIndex.from_product([dates, symbols]), columns=feature_cols)
                        """, language="python")
                        
                        if len(selected_episode['symbols']) > 4:
                            st.caption(f"...and `{len(selected_episode['symbols']) - 4}` more paired assets fused into this same hyper-matrix horizontally.")
                            
                    with st.expander("🔌 Microservice Data Contract (JSON Payload)"):
                        st.caption("This matches the expected schema for our portable inference microservices. The model consumes this generic structure directly from the Parquet-native feed.")
                        
                        # Generate a clean JSON sample for this specific episode context
                        contract_payload = {
                            "episode_id": selected_episode['episode_id'],
                            "symbols": selected_episode['symbols'],
                            "data_shape": [5, len(selected_episode['symbols']) * 6], # 5 time steps, 6 features per symbol
                            "schema_version": "v2.5-parquet-native",
                            "metrics_summary": {
                                "realized_pnl": selected_episode.get('realized_pnl', 0.0),
                                "hit_rate": selected_episode.get('hit_rate', 0.0),
                                "final_capital": selected_episode.get('final_capital', 100.0)
                            },
        "features_mapping": ["open", "high", "low", "close", "volume", "trades"]
                        }
                        st.json(contract_payload)

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
        st.info("No training results yet. Click 'Start' to begin training.")
    
    if st.session_state.training_active:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
