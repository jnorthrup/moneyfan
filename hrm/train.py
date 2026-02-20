"""
Train 24 codecs independently, then train HRM to emulate them.

GOALS.md:
- 24 Codecs: Small 2-layer ML models, trained independently
- HRM: Learns to emulate all 24 codec outputs
- Test-time: Only run HRM (no codecs)

Training Flow:
1. Phase 1: Train 24 codec models independently
2. Phase 2: Train HRM to emulate 24 codec outputs
3. Phase 3: Deploy HRM only
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from hrm.codecs import CodecCollection, HRM, CodecConfig, HRMConfig
from hrm.duck_store import DuckStore
from hrm.features import compute_all_features
from hrm.trade_pair_muxer import stochastic_bag, MuxerRegistry
import time
from pathlib import Path


def load_arrow_data(arrow_dir: str, n_samples: int = 10) -> list:
    """
    Load sample data from Arrow directory using DuckStore
    """
    store = DuckStore(arrow_dir=arrow_dir)
    data_list = []
    
    # Get list of all symbols
    all_files = list(Path(arrow_dir).glob("*.feather"))
    if not all_files:
        print(f"No .feather files found in {arrow_dir}")
        return []
    
    # Sample random files
    import random
    sample_files = random.sample(all_files, min(n_samples, len(all_files)))
    
    for feather_file in sample_files:
        try:
            symbol = feather_file.stem.replace("_", "-")
            df = store.load(symbol)
            if len(df) > 50:
                df['symbol'] = symbol
                data_list.append(df)
                print(f"  Loaded {symbol}: {len(df)} rows via DuckStore")
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
    
    return data_list


def create_training_data(data_list: list, n_batches: int = 500) -> list:
    """
    Create training batches from stochastic bags
    
    GOALS.md:
    - 30 pairs + USD in stochastic bag
    - Stochastic extent: 32-256 time steps
    - Stochastic length: 64-256 total steps
    - Up to 75% missing data allowed
    """
    if not data_list:
        return []
    
    batches = []
    
    for _ in range(n_batches):
        # Stochastic bag: 2-30 symbols (some may be empty - 75% dropout allowed)
        n_select = np.random.randint(2, 31)
        
        if len(data_list) < n_select:
            continue
        
        # Randomly select symbols for this bag
        import random
        bag_data = random.sample(data_list, n_select)
        
        if not bag_data or len(bag_data) < 2:
            continue
        
        # Stochastic extent: 32-256 time steps
        extent_length = np.random.randint(32, 257)
        
        # Get features for each symbol
        all_metrics = []
        for df in bag_data:
            if len(df) < extent_length:
                # Missing data - up to 75% allowed
                if np.random.random() > 0.25:
                    continue
            
            # Random start position (ensure valid range)
            max_start = max(0, len(df) - extent_length)
            if max_start <= 0:
                continue
            
            start_idx = np.random.randint(0, max_start)
            extent_df = df.iloc[start_idx:start_idx + extent_length]
            
            # Compute features - ensure asset column exists
            extent_df = extent_df.copy()
            if 'symbol' in extent_df.columns:
                extent_df['asset'] = extent_df['symbol']
            else:
                extent_df['asset'] = 'UNKNOWN'
            
            features_df = compute_all_features(extent_df)
            
            if features_df is not None and len(features_df) > 0:
                # Get last row's features
                last_row = features_df.iloc[-1]
                metric_cols = [c for c in features_df.columns if c not in ['asset', 'timestamp', 'time', 'symbol']]
                all_metrics.append(last_row[metric_cols].values)
        
        if len(all_metrics) >= 2:
            # Average metrics across instruments (simplified)
            avg_metrics = np.mean(all_metrics, axis=0)
            batches.append(avg_metrics)
    
    print(f"Created {len(batches)} training batches")
    return batches


def train_codecs_and_hrm():
    """Train 24 codecs, then train HRM to emulate them"""
    
    print("=" * 70)
    print("  TRAINING 24 CODECS + HRM EMULATION")
    print("=" * 70)
    
    # Create models
    print("\nCreating models...")
    codec_config = CodecConfig()
    codecs = CodecCollection(n_codecs=24, config=codec_config)
    print(f"  24 Codecs: {codec_config.n_inputs} inputs -> {codec_config.hidden_dim} hidden -> 3 outputs")
    
    hrm_config = HRMConfig(n_codecs=24)
    hrm = HRM(hrm_config)
    print(f"  1 HRM: {hrm_config.n_inputs} inputs -> {hrm_config.hidden_dim} hidden -> 72 outputs")
    
    # Load training data
    print("\nLoading training data from Arrow directory...")
    data_list = load_arrow_data("hrm/data/arrow", n_samples=10)
    
    if len(data_list) == 0:
        print("  No training data found!")
        print("  Please ensure hrm/data/arrow/ contains .feather files")
        return
    
    # Create training batches
    batches = create_training_data(data_list, n_batches=500)
    
    if len(batches) == 0:
        print("  No training batches created!")
        return
    
    # Prepare training data
    n_inputs = codec_config.n_inputs
    n_codecs = 24
    
    # Create dummy codec targets (in real training, these would be computed from SOTA strategies)
    print("\nTraining Phase 1: 24 Codecs independently")
    print("-" * 50)
    
    for i, batch_metrics in enumerate(batches):
        if i >= 10:  # Limit for testing
            break
        
        # Skip if wrong size
        if len(batch_metrics) != n_inputs:
            continue
        
        # Create inputs and dummy targets
        inputs = torch.tensor([batch_metrics], dtype=torch.float32)
        
        # Create random targets for each codec (placeholder)
        # In real training, these would be computed from SOTA strategies
        targets = {}
        for agent_name, codec in codecs.codecs.items():
            # Random targets: confidence [0,1], direction [-1,1], regime_fit [0,1]
            targets[agent_name] = torch.randn(1, 3)
            targets[agent_name][:, 0] = torch.sigmoid(targets[agent_name][:, 0])  # confidence
            targets[agent_name][:, 1] = torch.tanh(targets[agent_name][:, 1])     # direction
            targets[agent_name][:, 2] = torch.sigmoid(targets[agent_name][:, 2])  # regime_fit
        
        # Train codecs
        for agent_name, codec in codecs.codecs.items():
            codec.train()
            codec.optimizer.zero_grad()
            loss = codec.compute_loss(inputs, targets[agent_name])
            loss.backward()
            codec.optimizer.step()
        
        if i % 10 == 0:
            print(f"  Batch {i}: trained 24 codecs")
    
    print("\nTraining Phase 2: HRM emulation")
    print("-" * 50)
    
    for i, batch_metrics in enumerate(batches):
        if i >= 10:  # Limit for testing
            break
        
        if len(batch_metrics) != n_inputs:
            continue
        
        inputs = torch.tensor([batch_metrics], dtype=torch.float32)
        
        # Get codec outputs (as training targets for HRM)
        codec_outputs = codecs.forward_all(inputs)
        
        # Train HRM to emulate codec outputs
        hrm.train()
        hrm.optimizer.zero_grad()
        loss = hrm.compute_loss(inputs, codec_outputs)
        loss.backward()
        hrm.optimizer.step()
        
        if i % 10 == 0:
            print(f"  Batch {i}: HRM loss = {loss.item():.4f}")
    
    # Save models
    save_dir = Path("hrm/checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    
    # Save codecs
    for agent_name, codec in codecs.codecs.items():
        codec_path = save_dir / f"codec_{agent_name}_{timestamp}.pt"
        torch.save({
            'model_state_dict': codec.state_dict(),
            'config': codec.config,
            'agent_name': agent_name,
        }, codec_path)
        print(f"  Saved: {codec_path.name}")
    
    # Save HRM
    hrm_path = save_dir / f"hrm_{timestamp}.pt"
    torch.save({
        'model_state_dict': hrm.state_dict(),
        'config': hrm_config,
        'agent_names': hrm.agent_names,
    }, hrm_path)
    print(f"  Saved: {hrm_path.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  24 Codecs trained independently")
    print(f"  1 HRM trained to emulate 24 codecs")
    print(f"  Test-time: Only run HRM (no codecs)")
    print("=" * 70)


if __name__ == "__main__":
    train_codecs_and_hrm()
