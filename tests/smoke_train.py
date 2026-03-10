"""
Deterministic Smoke Training Helper for Torch HRM

Provides a deterministic training loop with CPU and MPS device selection
for smoke testing the HRM module.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

from hrm.torch_hrm import (
    TorchHrmModule, 
    TorchHrmConfig, 
    create_hrm_model, 
    get_device
)


class DeterministicSmokeTrainer:
    """Deterministic smoke trainer for HRM model."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        seed: int = 42,
        **config_kwargs
    ):
        """
        Initialize the deterministic smoke trainer.
        
        Args:
            device: Device to use ("cpu", "mps", "cuda"). If None, auto-detect.
            seed: Random seed for reproducibility
            **config_kwargs: Additional config parameters
        """
        # Set device
        if device is None:
            device = get_device()
        self.device = device
        
        # Set seeds for full determinism
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.manual_seed(seed)
        # MPS also benefits from manual_seed
        torch.manual_seed(seed)
        
        # Create model
        self.model, self.config = create_hrm_model(
            device=device,
            seed=seed,
            **config_kwargs
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training history
        self.history: Dict[str, list] = {
            "epoch": [],
            "loss": [],
            "device": [],
        }
        
        # Metadata
        self.metadata = {
            "device": device,
            "seed": seed,
            "config": self.config.__dict__,
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        }
    
    def create_sample_data(
        self, 
        batch_size: int = 8, 
        seq_len: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create deterministic sample data for training.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (head_ids, relation_ids, tail_ids, target_scores)
        """
        # Use fixed seed for data generation
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + 100)
        
        num_entities = self.config.num_entities
        num_relations = self.config.num_relations
        
        head_ids = torch.randint(
            0, num_entities, 
            (batch_size, seq_len), 
            generator=gen,
            device=self.device
        )
        
        relation_ids = torch.randint(
            0, num_relations,
            (batch_size, seq_len),
            generator=gen,
            device=self.device
        )
        
        tail_ids = torch.randint(
            0, num_entities,
            (batch_size, seq_len),
            generator=gen,
            device=self.device
        )
        
        # Target scores (random but deterministic)
        target_scores = torch.randn(
            batch_size, seq_len, 1,
            generator=gen,
            device=self.device
        )
        
        return head_ids, relation_ids, tail_ids, target_scores
    
    def train_step(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        target_scores: torch.Tensor
    ) -> float:
        """
        Perform one training step.
        
        Args:
            head_ids: Head entity indices
            relation_ids: Relation indices
            tail_ids: Tail entity indices
            target_scores: Target scores to predict
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(head_ids, relation_ids, tail_ids)
        
        # Compute loss
        loss = self.criterion(predictions, target_scores)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        target_scores: torch.Tensor
    ) -> float:
        """
        Evaluate the model.
        
        Args:
            head_ids: Head entity indices
            relation_ids: Relation indices
            tail_ids: Tail entity indices
            target_scores: Target scores to predict
            
        Returns:
            Loss value
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(head_ids, relation_ids, tail_ids)
            loss = self.criterion(predictions, target_scores)
        
        return loss.item()
    
    def run_smoke_training(
        self,
        num_epochs: int = 5,
        batch_size: int = 8,
        seq_len: int = 4,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run smoke training for a few epochs.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            seq_len: Sequence length
            verbose: Whether to print progress
            
        Returns:
            Training results dictionary
        """
        if verbose:
            print(f"Starting smoke training on device: {self.device}")
            print(f"Model config: {self.config.__dict__}")
            print(f"Seed: {self.seed}")
        
        # Create data
        head_ids, relation_ids, tail_ids, target_scores = self.create_sample_data(
            batch_size=batch_size,
            seq_len=seq_len
        )
        
        results = {
            "device": self.device,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seed": self.seed,
            "epochs": [],
            "initial_loss": None,
            "final_loss": None,
            "loss_reduced": None,
        }
        
        for epoch in range(num_epochs):
            # Training step
            loss = self.train_step(head_ids, relation_ids, tail_ids, target_scores)
            
            # Record history
            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["device"].append(self.device)
            
            results["epochs"].append({
                "epoch": epoch,
                "loss": loss,
            })
            
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")
            
            # Store initial and final loss
            if epoch == 0:
                results["initial_loss"] = loss
            if epoch == num_epochs - 1:
                results["final_loss"] = loss
        
        # Determine if loss was reduced
        if results["initial_loss"] is not None and results["final_loss"] is not None:
            results["loss_reduced"] = results["final_loss"] < results["initial_loss"]
        
        self.metadata["training_results"] = results
        
        return results


def run_smoke_test_on_device(
    device: str,
    num_epochs: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run smoke test on a specific device.
    
    Args:
        device: Device to use ("cpu" or "mps")
        num_epochs: Number of epochs
        seed: Random seed
        
    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"Running smoke test on device: {device}")
    print(f"{'='*60}")
    
    try:
        trainer = DeterministicSmokeTrainer(
            device=device,
            seed=seed,
            num_entities=50,
            num_relations=5,
            embedding_dim=16,
            num_hash_buckets=32,
            hidden_dim=32,
        )
        
        results = trainer.run_smoke_training(
            num_epochs=num_epochs,
            batch_size=4,
            seq_len=2,
            verbose=True
        )
        
        results["status"] = "passed"
        results["error"] = None
        
        return results
        
    except Exception as e:
        return {
            "device": device,
            "status": "failed",
            "error": str(e),
        }


def main():
    """Main entry point for smoke training."""
    print("Torch HRM Deterministic Smoke Training")
    print("=" * 40)
    
    # Get available device
    device = get_device()
    print(f"Best available device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run on available device
    results = run_smoke_test_on_device(device, num_epochs=5, seed=42)
    
    print(f"\n{'='*60}")
    print("SMOKE TEST RESULTS")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2, default=str))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"smoke_test_results_{device}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
