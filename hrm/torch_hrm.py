"""
Minimal Torch HRM (Hashed Relational Model) Module

A simple PyTorch implementation of a hashed relational model for
deterministic smoke testing.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TorchHrmConfig:
    """Configuration for Torch HRM model."""
    
    def __init__(
        self,
        num_entities: int = 100,
        num_relations: int = 10,
        embedding_dim: int = 32,
        num_hash_buckets: int = 64,
        hidden_dim: int = 64,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_hash_buckets = num_hash_buckets
        self.hidden_dim = hidden_dim


class TorchHrmEmbedding(nn.Module):
    """HRM embedding layer with hash-based lookup."""
    
    def __init__(self, config: TorchHrmConfig):
        super().__init__()
        self.config = config
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(
            config.num_entities, 
            config.embedding_dim
        )
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(
            config.num_relations,
            config.embedding_dim
        )
        
        # Hash function for embedding modulation
        self.hash_weights = nn.Linear(config.embedding_dim, config.num_hash_buckets)
    
    def forward(
        self, 
        entity_ids: torch.Tensor, 
        relation_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            entity_ids: (batch_size, seq_len) entity indices
            relation_ids: (batch_size, seq_len) relation indices
            
        Returns:
            entity_emb: (batch_size, seq_len, embedding_dim)
            relation_emb: (batch_size, seq_len, embedding_dim)
        """
        entity_emb = self.entity_embeddings(entity_ids)
        relation_emb = self.relation_embeddings(relation_ids)
        
        return entity_emb, relation_emb


class TorchHrmModule(nn.Module):
    """Minimal Torch HRM model for smoke testing."""
    
    def __init__(self, config: Optional[TorchHrmConfig] = None):
        super().__init__()
        self.config = config or TorchHrmConfig()
        
        # Embedding layer
        self.embedding = TorchHrmEmbedding(self.config)
        
        # Simple MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.config.embedding_dim * 3, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
        
        # Initialize weights for reproducibility
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with deterministic seed."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        head_ids: torch.Tensor, 
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for HRM model.
        
        Args:
            head_ids: (batch_size, seq_len) head entity indices
            relation_ids: (batch_size, seq_len) relation indices
            tail_ids: (batch_size, seq_len) tail entity indices
            
        Returns:
            scores: (batch_size, seq_len, 1) predicted scores
        """
        # Get embeddings
        head_emb, relation_emb = self.embedding(head_ids, relation_ids)
        tail_emb, _ = self.embedding(tail_ids, relation_ids)
        
        # Concatenate head + relation + tail for scoring
        combined = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        
        # Simple scoring
        scores = self.decoder(combined)
        
        return scores
    
    def get_embeddings(
        self, 
        entity_ids: torch.Tensor, 
        relation_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get entity and relation embeddings."""
        return self.embedding(entity_ids, relation_ids)


def create_hrm_model(
    device: str = "cpu",
    seed: int = 42,
    **config_kwargs
) -> Tuple[TorchHrmModule, TorchHrmConfig]:
    """
    Create a deterministic HRM model.
    
    Args:
        device: "cpu", "mps", or "cuda"
        seed: Random seed for reproducibility
        **config_kwargs: Additional config parameters
        
    Returns:
        model: The HRM model
        config: The configuration used
    """
    # Set seeds for determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    config = TorchHrmConfig(**config_kwargs)
    model = TorchHrmModule(config)
    model = model.to(device)
    model.eval()
    
    return model, config


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string: "mps", "cuda", or "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
