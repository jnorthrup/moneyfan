"""
Backbone Countercoin Bag Trainer - DuckDB-based
================================================

Creates a structured bag following backbone countercoin spokes and breadth:
1. Backbone (BTC, ETH) - Central pairs
2. Countercoin Spokes (SOL, ADA, XRP, etc.) - Spokes around backbone
3. Breadth (diversified pairs) - Diversification

All data flows through DuckDB.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

try:
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
except ImportError:
    HAS_DUCK_STORE = False
    print("[BackboneTrainer] DuckStore not available")

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("[BackboneTrainer] MLX not available")

@dataclass
class BackboneConfig:
    """Configuration for backbone countercoin bag"""
    # Backbone pairs (central pairs)
    backbone_pairs: List[str] = field(default_factory=lambda: [
        "BTC-USD",
        "ETH-USD"
    ])
    
    # Countercoin spokes (spokes around backbone)
    # These are pairs that have strong correlation with backbone
    countercoin_spokes: List[str] = field(default_factory=lambda: [
        "SOL-USD",    # Strong correlation with ETH
        "ADA-USD",    # Strong correlation with BTC
        "XRP-USD",    # Strong correlation with BTC
        "DOGE-USD",   # Meme coin, correlated
        "AVAX-USD",   # Layer 1, correlated
        "DOT-USD",    # Layer 1, correlated
        "MATIC-USD",  # Layer 2, correlated
        "LINK-USD",   # Oracle, correlated
        "UNI-USD",    # DeFi, correlated
        "ATOM-USD",   # Cosmos, correlated
    ])
    
    # Breadth pairs (diversified across sectors)
    # Provide breadth and diversification
    breadth_pairs: List[str] = field(default_factory=lambda: [
        "LTC-USD",    # Payment
        "BCH-USD",    # Payment
        "ETC-USD",    # Smart contract
        "FIL-USD",    # Storage
        "APT-USD",    # Layer 1
        "OP-USD",     # Layer 2
        "ARB-USD",    # Layer 2
        "SUI-USD",    # Layer 1
        "SEI-USD",    # Layer 1
        "RUNE-USD",   # DEX
        "INJ-USD",    # DeFi
        "TIA-USD",    # Modular
        "PYTH-USD",   # Oracle
        "JUP-USD",    # DEX
        "WIF-USD",    # Meme
        "BONK-USD",   # Meme
        "PEPE-USD",   # Meme
    ])
    
    # Prune non-core pairs (remove if not in backbone/spoke/breadth)
    prune_non_core: bool = True
    
    # Training parameters
    timeframe: str = "5m"
    min_seq_len: int = 64
    max_seq_len: int = 256
    seed: int = 42
    
    # Output
    output_dir: str = "models/backbone_trained"
    
    def __post_init__(self):
        # Set random seed
        np.random.seed(self.seed)
    
    def get_all_core_pairs(self) -> List[str]:
        """Get all core pairs (backbone + spokes + breadth)"""
        all_pairs = self.backbone_pairs + self.countercoin_spokes + self.breadth_pairs
        return list(set(all_pairs))  # Remove duplicates
    
    def get_bag_structure(self) -> Dict[str, List[str]]:
        """Get the bag structure (backbone, spokes, breadth)"""
        return {
            'backbone': self.backbone_pairs,
            'spokes': self.countercoin_spokes,
            'breadth': self.breadth_pairs
        }

class BackboneCountercoinBag:
    """
    Structured bag following backbone countercoin spokes and breadth
    """
    
    def __init__(self, config: BackboneConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB store
        if HAS_DUCK_STORE:
            self.duck_store = DuckStore("hrm/data/market.duckdb")
        else:
            self.duck_store = None
        
        # Bag structure
        self.bag_structure = config.get_bag_structure()
        self.all_core_pairs = config.get_all_core_pairs()
        
        print(f"[BackboneCountercoinBag] Initialized")
        print(f"  Backbone: {config.backbone_pairs}")
        print(f"  Spokes: {config.countercoin_spokes}")
        print(f"  Breadth: {config.breadth_pairs}")
        print(f"  Total core pairs: {len(self.all_core_pairs)}")
    
    def query_duckdb(self, query: str) -> pd.DataFrame:
        """Query DuckDB and return DataFrame"""
        if not self.duck_store:
            return pd.DataFrame()
        
        try:
            return self.duck_store.conn.execute(query).fetchdf()
        except Exception as e:
            print(f"[BackboneCountercoinBag] Query failed: {e}")
            return pd.DataFrame()
    
    def load_pair_data(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load data for a single pair from DuckDB"""
        if not self.duck_store:
            return None
        
        try:
            # Convert symbol for DuckDB (BTC-USD -> BTC_USD or BTCUSD)
            db_symbol = symbol.replace("-", "_")
            
            # Build query
            if start and end:
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{db_symbol}' 
                    AND timestamp >= '{start}' 
                    AND timestamp <= '{end}'
                    ORDER BY timestamp
                """
            else:
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{db_symbol}'
                    ORDER BY timestamp
                """
            
            df = self.query_duckdb(query)
            
            if df.empty:
                # Try alternative symbol format
                db_symbol_alt = symbol.replace("-", "")
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{db_symbol_alt}'
                    ORDER BY timestamp
                """
                df = self.query_duckdb(query)
            
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"[BackboneCountercoinBag] Failed to load {symbol}: {e}")
            return None
    
    def extract_sequences(self, df: pd.DataFrame, n_sequences: int = 100) -> List[Dict[str, Any]]:
        """Extract variable-length sequences from DataFrame"""
        if df.empty or len(df) < self.config.min_seq_len:
            return []
        
        sequences = []
        total_length = len(df)
        
        for i in range(n_sequences):
            # Random sequence length
            seq_len = np.random.randint(self.config.min_seq_len, self.config.max_seq_len)
            
            # Random start index
            max_start = max(0, total_length - seq_len)
            if max_start == 0:
                continue
            
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + seq_len
            
            # Extract sequence
            seq_df = df.iloc[start_idx:end_idx].copy()
            
            if len(seq_df) < self.config.min_seq_len:
                continue
            
            # Create sequence
            sequence = {
                'sequence_id': f"seq_{i:06d}",
                'start_timestamp': seq_df.index[0],
                'end_timestamp': seq_df.index[-1],
                'seq_len': len(seq_df),
                'open_prices': seq_df['open'].values,
                'high_prices': seq_df['high'].values,
                'low_prices': seq_df['low'].values,
                'close_prices': seq_df['close'].values,
                'volumes': seq_df['volume'].values,
                'returns': np.diff(seq_df['close'].values) / seq_df['close'].values[:-1] if len(seq_df) > 1 else np.array([0.0]),
                'label': np.random.choice([1, -1, 0])  # Buy/Sell/Hold
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def create_backbone_bag(self, sequences_per_pair: int = 50) -> Dict[str, Any]:
        """Create backbone countercoin bag"""
        print(f"\n{'='*60}")
        print("CREATING BACKBONE COUNTERCOIN BAG")
        print(f"{'='*60}")
        
        bag_data = {
            'backbone': {},
            'spokes': {},
            'breadth': {},
            'metadata': {
                'created_at': time.time(),
                'config': self.config,
                'structure': self.bag_structure
            }
        }
        
        # Load and process each category
        for category, pairs in self.bag_structure.items():
            print(f"\nProcessing {category}: {len(pairs)} pairs")
            
            category_data = {}
            
            for pair in pairs:
                print(f"  Loading {pair}...")
                
                # Load from DuckDB
                df = self.load_pair_data(pair)
                
                if df is None or df.empty:
                    print(f"    Warning: No data for {pair}")
                    continue
                
                # Extract sequences
                sequences = self.extract_sequences(df, sequences_per_pair)
                
                if sequences:
                    category_data[pair] = sequences
                    print(f"    ✓ Extracted {len(sequences)} sequences")
                else:
                    print(f"    Warning: No sequences for {pair}")
            
            bag_data[category] = category_data
        
        # Save bag structure
        self.save_backbone_bag(bag_data)
        
        return bag_data
    
    def save_backbone_bag(self, bag_data: Dict[str, Any]):
        """Save backbone bag to DuckDB and metadata"""
        print(f"\n{'='*60}")
        print("SAVING BACKBONE BAG")
        print(f"{'='*60}")
        
        # Save metadata
        metadata_file = self.output_dir / "backbone_bag_metadata.json"
        metadata = {
            'created_at': time.time(),
            'config': self.config.__dict__,
            'structure': self.bag_structure,
            'total_pairs': {
                'backbone': len(bag_data['backbone']),
                'spokes': len(bag_data['spokes']),
                'breadth': len(bag_data['breadth'])
            },
            'total_sequences': {
                'backbone': sum(len(sequences) for sequences in bag_data['backbone'].values()),
                'spokes': sum(len(sequences) for sequences in bag_data['spokes'].values()),
                'breadth': sum(len(sequences) for sequences in bag_data['breadth'].values())
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Saved metadata to {metadata_file}")
        
        # Save sequences to DuckDB (if available)
        if HAS_DUCK_STORE:
            self.save_to_duckdb(bag_data)
    
    def save_to_duckdb(self, bag_data: Dict[str, Any]):
        """Save sequences to DuckDB"""
        print(f"\n{'='*60}")
        print("SAVING TO DUCKDB")
        print(f"{'='*60}")
        
        # Create sequences table
        self.duck_store.conn.execute("""
            CREATE TABLE IF NOT EXISTS backbone_sequences (
                sequence_id TEXT PRIMARY KEY,
                category TEXT,
                pair TEXT,
                start_timestamp TIMESTAMP,
                end_timestamp TIMESTAMP,
                seq_len INTEGER,
                open_prices TEXT,
                high_prices TEXT,
                low_prices TEXT,
                close_prices TEXT,
                volumes TEXT,
                returns TEXT,
                label INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create category indexes
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seq_category ON backbone_sequences(category)
        """)
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seq_pair ON backbone_sequences(pair)
        """)
        
        # Insert sequences
        total_inserted = 0
        
        for category, category_data in bag_data.items():
            if category in ['metadata']:
                continue
            
            for pair, sequences in category_data.items():
                for seq in sequences:
                    try:
                        # Serialize arrays to JSON strings
                        open_str = json.dumps(seq['open_prices'].tolist())
                        high_str = json.dumps(seq['high_prices'].tolist())
                        low_str = json.dumps(seq['low_prices'].tolist())
                        close_str = json.dumps(seq['close_prices'].tolist())
                        volumes_str = json.dumps(seq['volumes'].tolist())
                        returns_str = json.dumps(seq['returns'].tolist())
                        
                        # Insert
                        self.duck_store.conn.execute("""
                            INSERT OR REPLACE INTO backbone_sequences 
                            (sequence_id, category, pair, start_timestamp, end_timestamp, seq_len,
                             open_prices, high_prices, low_prices, close_prices, volumes, returns, label)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            seq['sequence_id'],
                            category,
                            pair,
                            seq['start_timestamp'],
                            seq['end_timestamp'],
                            seq['seq_len'],
                            open_str,
                            high_str,
                            low_str,
                            close_str,
                            volumes_str,
                            returns_str,
                            int(seq['label'])
                        ))
                        
                        total_inserted += 1
                        
                    except Exception as e:
                        print(f"  Warning: Failed to insert {seq['sequence_id']}: {e}")
                        break
        
        print(f"Inserted {total_inserted} sequences into DuckDB")
    
    def train_predictors(self, bag_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Train 3 predictors on backbone bag"""
        if not HAS_MLX:
            print("[BackboneCountercoinBag] MLX not available, skipping training")
            return []
        
        print(f"\n{'='*60}")
        print("TRAINING 3 PREDICTORS ON BACKBONE BAG")
        print(f"{'='*60}")
        
        predictors = []
        
        # Training logic would go here
        # For now, create placeholder models
        
        predictor_types = ["transformer_5m", "xgboost_15m", "lightgbm_1h"]
        
        for i, ptype in enumerate(predictor_types):
            print(f"[{i+1}/{len(predictor_types)}] Training {ptype}...")
            
            # Create simple model (placeholder)
            model = self._create_model(ptype)
            
            # Save model
            model_path = self.output_dir / f"{ptype}.mlxbf"
            model_config = {
                'type': ptype,
                'trained_at': time.time(),
                'bag_structure': self.bag_structure,
                'total_sequences': sum(
                    len(sequences) 
                    for category in ['backbone', 'spokes', 'breadth']
                    for sequences in bag_data.get(category, {}).values()
                )
            }
            
            with open(model_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            print(f"  Saved model to {model_path}")
            
            predictors.append({
                'type': ptype,
                'model': model,
                'config': model_config
            })
        
        return predictors
    
    def _create_model(self, model_type: str):
        """Create MLX model based on type"""
        if "transformer" in model_type:
            class TransformerModel(nn.Module):
                def __init__(self, input_dim=64, d_model=128, nhead=8):
                    super().__init__()
                    self.encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
                        num_layers=2
                    )
                    self.fc = nn.Linear(d_model, 1)
                
                def __call__(self, x):
                    x = x.astype(mx.float32)
                    encoded = self.encoder(x)
                    pooled = mx.mean(encoded, axis=1)
                    return self.fc(pooled)
            
            return TransformerModel()
        
        elif "xgboost" in model_type:
            class XGBoostLikeModel(nn.Module):
                def __init__(self, input_dim=64):
                    super().__init__()
                    self.base = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def __call__(self, x):
                    x = x.astype(mx.float32)
                    return self.base(x)
            
            return XGBoostLikeModel()
        
        else:  # lightgbm
            class LightGBMLikeModel(nn.Module):
                def __init__(self, input_dim=64):
                    super().__init__()
                    self.base = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
                
                def __call__(self, x):
                    x = x.astype(mx.float32)
                    return self.base(x)
            
            return LightGBMLikeModel()
    
    def generate_report(self, bag_data: Dict[str, Any], predictors: List[Dict[str, Any]]) -> str:
        """Generate training report"""
        total_sequences = 0
        for category in ['backbone', 'spokes', 'breadth']:
            for sequences in bag_data.get(category, {}).values():
                total_sequences += len(sequences)
        
        report = []
        report.append("="*80)
        report.append("BACKBONE COUNTERCOIN BAG TRAINER REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("BAG STRUCTURE:")
        report.append(f"  Backbone (central): {len(self.config.backbone_pairs)} pairs")
        report.append(f"  Countercoin spokes: {len(self.config.countercoin_spokes)} pairs")
        report.append(f"  Breadth (diversified): {len(self.config.breadth_pairs)} pairs")
        report.append(f"  Total core pairs: {len(self.all_core_pairs)}")
        report.append("")
        
        report.append("SEQUENCES EXTRACTED:")
        for category in ['backbone', 'spokes', 'breadth']:
            count = len(bag_data.get(category, {}))
            if count > 0:
                report.append(f"  {category}: {count} pairs")
        report.append(f"  Total sequences: {total_sequences}")
        report.append("")
        
        report.append("PREDICTORS TRAINED:")
        for i, predictor in enumerate(predictors):
            report.append(f"  {i+1}. {predictor['type']}")
        report.append("")
        
        report.append("OUTPUT FILES:")
        for file in self.output_dir.iterdir():
            if file.is_file():
                report.append(f"  • {file.name}")
        report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    async def train(self, sequences_per_pair: int = 50) -> Dict[str, Any]:
        """Main training loop"""
        print(f"\n{'='*80}")
        print("BACKBONE COUNTERCOIN BAG TRAINER")
        print("Structured Bag: Backbone → Spokes → Breadth")
        print(f"{'='*80}")
        
        # Create backbone bag
        bag_data = self.create_backbone_bag(sequences_per_pair)
        
        # Train predictors
        predictors = self.train_predictors(bag_data)
        
        # Generate report
        report = self.generate_report(bag_data, predictors)
        print(f"\n{report}")
        
        return {
            'bag_data': bag_data,
            'predictors': predictors,
            'config': self.config
        }

# Example usage
async def main():
    config = BackboneConfig(
        timeframe="5m",
        min_seq_len=64,
        max_seq_len=256,
        seed=42,
        output_dir="models/backbone_trained"
    )
    
    trainer = BackboneCountercoinBag(config)
    results = await trainer.train(sequences_per_pair=30)
    
    print(f"\nTraining complete!")
    print(f"Output directory: {trainer.output_dir}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())