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
        
        n_bags = st.number_input("Number of Bags", min_value=1, max_value=1000, value=500)
        capital = st.number_input("Starting Capital ($)", min_value=10, value=100)
        bag_size = st.number_input("Bag Size (symbols)", min_value=5, max_value=50, value=30)
        epochs = st.number_input("Epochs per Bag", min_value=1, max_value=1000, value=3)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                if not st.session_state.training_active:
                    config = TrainingConfig(
                        n_bags=n_bags,
                        capital=capital,
                        bag_size=bag_size,
                        epochs=epochs
                    )
                    
                    st.session_state.trainer = UnifiedTrainer(config)
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
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
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
            fig.add_trace(go.Histogram(
                x=df['win_rate'],
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
        
        # Make sure columns exist from legacy events
        for col in ['winning_agent', 'hrm_score', 'predictor_loss']:
            if col not in df.columns:
                df[col] = "N/A"
        
        display_df = df[['bag_id', 'symbols', 'final_capital', 'pnl', 'win_rate', 'winning_agent', 'hrm_score', 'predictor_loss', 'total_trades']].tail(20)
        display_df = display_df.copy()
        display_df['symbols'] = display_df['symbols'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        display_df = display_df.round(3)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
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
