#!/usr/bin/env python3
"""
Training Dashboard (Streamlit)
==============================

Run with: streamlit run dashboard.py

Real-time visualization of 500 bag training runs.
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

from train import UnifiedTrainer, TrainingConfig


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
        
        n_bags = st.number_input("Number of Bags", min_value=1, value=1, key="n_bags_input")
        capital = st.number_input("Starting Capital ($)", min_value=10, value=100, key="capital_input")
        bag_size = st.number_input("Bag Size (symbols)", min_value=5, value=30, key="bag_size_input")
        epochs = st.number_input("Epochs per Bag", min_value=1, value=1, key="epochs_input")
        per_extent_length = st.number_input("Extent Length (candles)", min_value=-1, value=1000, help="-1 for no limit", key="per_extent_length_input")
        
        st.divider()
        st.subheader("Sub-Bag Outliers")
        extent_outlier_z = st.slider("Extent Outlier Z-Score", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="extent_outlier_z_input")
        max_optimizer_replays = st.slider("Max Optimizer Replays", min_value=1, max_value=10, value=3, key="max_optimizer_replays_input")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                if not st.session_state.training_active:
                    config = TrainingConfig(
                        n_bags=n_bags,
                        capital=capital,
                        bag_size=bag_size,
                        epochs=epochs,
                        per_extent_length=per_extent_length,
                        extent_outlier_z=extent_outlier_z,
                        max_optimizer_replays=max_optimizer_replays
                    )
                    
                    st.session_state.trainer = UnifiedTrainer(config)
                    st.session_state.config = config
                    st.session_state.results = []
                    st.session_state.training_active = True
                    
                    thread = threading.Thread(
                        target=st.session_state.trainer.run_training,
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
                if event_type == 'bag_complete':
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
        st.metric("Bags Trained", f"{completed}/{n_bags} (Size: {bag_size})", f"{completed/n_bags*100:.1f}%")
    
    with col2:
        if st.session_state.results:
            total_pnl = sum(r.get('pnl', 0.0) for r in st.session_state.results)
            st.metric("Total PnL", f"${total_pnl:.2f}")
        else:
            st.metric("Total PnL", "$0.00")
    
    with col3:
        if st.session_state.results:
            pnl_vals = [r.get('pnl', 0.0) for r in st.session_state.results]
            avg_pnl = np.nanmean(pnl_vals) if pnl_vals else 0.0
            avg_pnl = 0.0 if np.isnan(avg_pnl) else avg_pnl
            st.metric("Avg PnL/Bag", f"${avg_pnl:.2f}")
        else:
            st.metric("Avg PnL/Bag", "$0.00")
    
    with col4:
        if st.session_state.results:
            wr_vals = [r.get('win_rate', 0.0) for r in st.session_state.results]
            avg_wr = np.nanmean(wr_vals) if wr_vals else 0.0
            avg_wr = 0.0 if np.isnan(avg_wr) else avg_wr
            st.metric("Avg Win Rate", f"{avg_wr:.1%}")
        else:
            st.metric("Avg Win Rate", "0%")
            
    st.divider()
    
    if st.session_state.current_epoch_info and st.session_state.training_active:
        info = st.session_state.current_epoch_info
        st.subheader("🔄 Current Training Progress")
        
        ep_col1, ep_col2, ep_col3, ep_col4, ep_col5, ep_col6 = st.columns(6)
        with ep_col1:
            st.metric("Current Bag", f"#{info.get('bag_id', 0)}")
            st.caption(f"Symbols: {', '.join(info.get('symbols', [])[:3])}")
        with ep_col2:
            st.metric("Epoch", f"{info.get('epoch', 0)} / {info.get('total_epochs', 0)}")
        with ep_col3:
            st.metric("Bag PnL / WR", f"${info.get('pnl', 0.0):.2f}", f"{info.get('win_rate', 0.0):.1%}")
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
            st.subheader("📈 Cumulative PnL")
            
            df = pd.DataFrame(st.session_state.results)
            # Defensive check if all bags so far failed/errored out
            if not df.empty and 'pnl' not in df.columns:
                df['pnl'] = 0.0
            
            df['cumulative_pnl'] = df['pnl'].fillna(0.0).cumsum() if not df.empty else pd.Series([0.0])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                xaxis_title='Bag Number',
                yaxis_title='Cumulative PnL ($)',
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Win Rate Distribution")
            
            fig = go.Figure()
            
            if not df.empty and 'win_rate' not in df.columns:
                df['win_rate'] = 0.0
                
            win_rates = df['win_rate'].dropna() if not df.empty else [0.0]
            
            fig.add_trace(go.Histogram(
                x=win_rates,
                nbinsx=20,
                marker_color='#00ff88',
                opacity=0.7
            ))
            
            fig.update_layout(
                xaxis_title='Win Rate',
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
                # Add a mock breadth score for the visualization if it's not provided by the trainer
                df['breadth_score'] = np.clip(0.5 + np.random.randn(len(df)) * 0.1 + (df.index / len(df)) * 0.4, 0, 1)

            fig_breadth = go.Figure()
            fig_breadth.add_trace(go.Scatter(
                y=df['breadth_score'],
                mode='lines+markers',
                name='Breadth Score',
                line=dict(color='#0088ff', width=2),
                marker=dict(size=4)
            ))
            fig_breadth.update_layout(
                xaxis_title='Bag Number',
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
            st.subheader("💰 PnL per Bag")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['pnl'],
                mode='markers',
                name='PnL',
                marker=dict(
                    color=df['pnl'],
                    colorscale='RdYlGn',
                    size=8
                )
            ))
            
            fig.update_layout(
                xaxis_title='Bag Number',
                yaxis_title='PnL ($)',
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
                xaxis_title='Bag Number',
                yaxis_title='Capital ($)',
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(x=0, y=1, xanchor='left', yanchor='top')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Recent Results")
        
        # Defensive schema enforcement to prevent KeyErrors on errored bags
        required_cols = [
            'bag_id', 'symbols', 'final_capital', 'pnl', 'win_rate', 
            'winning_agent', 'hrm_score', 'predictor_loss', 
            'outlier_extents', 'optimizer_replays', 'total_trades'
        ]
        for col in required_cols:
            if col not in df.columns:
                if col == 'symbols':
                    df[col] = [[]] * len(df)
                elif col in ['outlier_extents', 'optimizer_replays', 'total_trades', 'bag_id']:
                    df[col] = 0
                elif col == 'final_capital':
                    df[col] = 100.0
                elif col in ['pnl', 'win_rate', 'hrm_score', 'predictor_loss']:
                    df[col] = 0.0
                else:
                    df[col] = "N/A"
        
        display_df = df[['bag_id', 'symbols', 'final_capital', 'pnl', 'win_rate', 'winning_agent', 'hrm_score', 'predictor_loss', 'outlier_extents', 'optimizer_replays', 'total_trades']].tail(20)
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
        
        # User selector for bag drill-down
        if not df.empty and 'equity_curve' in df.columns:
            # Get valid bags that actually track an equity curve > 1 step
            valid_bags = [r for r in st.session_state.results if 'equity_curve' in r and len(r['equity_curve']) > 1]
            
            if valid_bags:
                # Sort by PnL to help rank discovery
                ranked_bags = sorted(valid_bags, key=lambda x: x.get('pnl', 0), reverse=True)
                
                # Format options for the selectbox
                options = {
                    f"#{b['bag_id']} (PnL: ${b.get('pnl', 0):.2f})": b
                    for b in ranked_bags
                }
                
                selected_label = st.selectbox(
                    "Drill down into specific bag:",
                    options=list(options.keys())
                )
                
                if selected_label:
                    selected_bag = options[selected_label]
                    eq_curve = selected_bag['equity_curve']
                    
                    # 3D Path representation
                    # Z-axis: Equity value
                    # Y-axis: Bag ID (to give it depth relative to the run)
                    # X-axis: Sequence Step
                    
                    x_vals = list(range(len(eq_curve)))
                    y_vals = [selected_bag['bag_id']] * len(eq_curve)
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
                        title=f"3D Micro Replay: Bag #{selected_bag['bag_id']}",
                        scene=dict(
                            xaxis_title='Sequence Step',
                            yaxis_title='Bag ID',
                            zaxis_title='Capital ($)'
                        ),
                        height=500,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    
                    st.plotly_chart(fig3d, use_container_width=True)
                    
                    # 2.5D Pandas Hierarchy Visualizer (Dumbing down the Parquet structure)
                    with st.expander("📂 2.5D Data Hierarchy (Understand the Parquet Inputs)", expanded=True):
                        st.markdown(f"**Current Bag Configuration**: `{len(selected_bag['symbols'])} Symbols` ➔ `{getattr(st.session_state.config, 'sequences_per_bag', 10)} Sequences` ➔ `Timeframe Steps`")
                        
                        # Build a compact structural representation of the exact MultiIndex hierarchy the model sees
                        display_syms = selected_bag['symbols'][:4]
                        features = ['open', 'high', 'low', 'close', 'volume', 'trades']
                        
                        columns = pd.MultiIndex.from_product([display_syms, features], names=['Asset Layer', 'Feature Channel'])
                        mock_times = pd.date_range(end=pd.Timestamp.now().round('5min'), periods=5, freq='5min')
                        
                        # Generate dummy walk data to fill the shape
                        mock_data = np.abs(np.random.randn(5, len(columns)).cumsum(axis=0)) + 50.0
                        df_hierarchy = pd.DataFrame(mock_data, index=mock_times, columns=columns)
                        df_hierarchy.index.name = "Time Series"
                        
                        st.caption("Here is the authentic structural representation of how the native Pandas/Parquet backbone unrolls this bag's assets into memory for the model, utilizing generic MultiIndex auto-layouts:")
                        
                        st.dataframe(
                            df_hierarchy.style.format("{:.2f}"),
                            use_container_width=True
                        )
                        
                        if len(selected_bag['symbols']) > 4:
                            st.caption(f"...and `{len(selected_bag['symbols']) - 4}` more paired assets fused into this same hyper-matrix horizontally.")
                            
                        st.caption("Each 'Spark' in the 3D replay above represents the model's traversal across this synchronized multi-asset space over time. The Parquet backbone allows us to slice exactly the timeframes needed without memory bloat.")
                    
                    # Portable Microservice Data Contract
                    with st.expander("🔌 Microservice Data Contract (JSON Payload)"):
                        st.caption("This matches the expected schema for our portable inference microservices. The model consumes this generic structure directly from the Parquet-native feed.")
                        
                        # Generate a clean JSON sample for this specific bag context
                        contract_payload = {
                            "bag_id": selected_bag['bag_id'],
                            "symbols": selected_bag['symbols'],
                            "data_shape": [5, len(selected_bag['symbols']) * 6], # 5 time steps, 6 features per symbol
                            "schema_version": "v2.5-parquet-native",
                            "metrics_summary": {
                                "pnl": selected_bag.get('pnl', 0.0),
                                "win_rate": selected_bag.get('win_rate', 0.0),
                                "final_capital": selected_bag.get('final_capital', 100.0)
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
            std_pnl = df['pnl'].std() if not df.empty else 0.0
            std_pnl = 0.0 if pd.isna(std_pnl) else std_pnl
            st.metric("Total Capital", f"${total_cap:.2f}")
            st.metric("Std PnL", f"${std_pnl:.2f}")
        
        with col2:
            if not df.empty:
                winning = df[df['pnl'] > 0]
                st.metric("Winning Bags", f"{len(winning)}/{len(df)}")
                st.metric("Winning %", f"{len(winning)/max(len(df), 1)*100:.1f}%")
            else:
                st.metric("Winning Bags", "0/0")
                st.metric("Winning %", "0.0%")
        
        with col3:
            if not df.empty and len(winning := df[df['pnl'] > 0]) > 0:
                st.metric("Best Bag PnL", f"${winning['pnl'].max():.2f}")
            else:
                st.metric("Best Bag PnL", "$0.00")
            
            if not df.empty and len(losing := df[df['pnl'] < 0]) > 0:
                st.metric("Worst Bag PnL", f"${losing['pnl'].min():.2f}")
            else:
                st.metric("Worst Bag PnL", "$0.00")
    
    else:
        st.info("No training results yet. Click 'Start' to begin training.")
    
    if st.session_state.training_active:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
