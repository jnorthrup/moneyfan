
"""
HRM Interfaces (Polymorphic Protocols)
=====================================

Strict contract definitions for the Nexus Architecture.
Adheres to Interface Segregation Principle (ISP).
"""

from typing import Protocol, Dict, List, Optional, Any, runtime_checkable
import pandas as pd
import numpy as np
from datetime import datetime

@runtime_checkable
class ITimeProvider(Protocol):
    """Source of truth for time."""
    def now(self) -> datetime: ...

@runtime_checkable
class IMemoryStore(Protocol):
    """
    Abstract Data Store (Arrow/SQLite/Parquet).
    Responsible for persistence and retrieval of raw market data.
    """
    def load(self, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """Load Dataframe for a symbol."""
        ...

    def list_symbols(self) -> List[str]:
        """List all available symbols."""
        ...

@runtime_checkable
class ISignalGenerator(Protocol):
    """
    Abstract Signal Engine (The Senses).
    Produces normalized signal vectors from market data.
    """
    def compute_signals(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute signal map {name: array} from dataframe."""
        ...
    
    @property
    def signal_names(self) -> List[str]:
        """List of signal names produced."""
        ...

@runtime_checkable
class ICognitiveModel(Protocol):
    """
    Abstract Reasoning Model (The Brain).
    Consumes signals and produces high-level intent/alpha.
    """
    def predict(self, tensor: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            tensor: [batch, seq_len, features]
            context: [batch, context_dim]
        Returns:
            alpha: [batch, output_dim]
        """
        ...

@runtime_checkable
class IRoutingEngine(Protocol):
    """
    Abstract Topology Engine (The Map).
    Calculates paths between assets.
    """
    def find_best_route(self, source: str, target: str, amount: float) -> List[Any]:
        """Find optimal execution path."""
        ...

@runtime_checkable
class IExecutionBackend(Protocol):
    """
    Abstract Executioner (The Hands).
    Executes orders against a venue (Coinbase, Paper, Simulation).
    """
    def execute_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute atomic order."""
        ...
    
    def get_balance(self, asset: str) -> float:
        """Get current balance."""
        ...

@runtime_checkable
class INexus(Protocol):
    """
    Central Nervous System.
    Coordinates the flow between components.
    """
    memory: IMemoryStore
    senses: ISignalGenerator
    brain: ICognitiveModel
    router: IRoutingEngine
    executor: IExecutionBackend
    
    def pulse(self) -> None:
        """Atomic system step."""
        ...
