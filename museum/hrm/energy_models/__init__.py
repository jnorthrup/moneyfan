"""
Energy Quant Models - Stateful Signal Executors

Each model is STATEFUL with full duplex I/O contract with HRM.

INBOUND (from HRM):
  - weight: How much to scale this model's signals
  - directive: Meta-control (regime hint, risk limit, etc.)

OUTBOUND (to HRM):
  - report: State summary for HRM to consume
    - energy_level: Signal momentum (rolling)
    - signal_entropy: Diversity of signals across assets
    - regime_confidence: How sure model is about current regime
    - performance_estimate: Self-assessed expected return
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


# =============================================================================
# I/O CONTRACTS
# =============================================================================

@dataclass
class ModelDirective:
    """HRM → Model: Control signals"""
    regime_hint: float = 0.0      # -1 (trending) to +1 (ranging)
    risk_limit: float = 1.0       # Scale factor for risk
    focus_assets: Optional[np.ndarray] = None  # Which assets to focus on


@dataclass  
class ModelReport:
    """Model → HRM: State summary"""
    energy_level: float = 0.0
    signal_entropy: float = 0.0
    regime_confidence: float = 0.5
    performance_estimate: float = 0.0
    n_active_assets: int = 0
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.energy_level,
            self.signal_entropy,
            self.regime_confidence,
            self.performance_estimate,
            self.n_active_assets / 43.0  # Normalized
        ], dtype=np.float32)


@dataclass
class ModelState:
    """Internal state carried across calls"""
    energy: float = 0.0
    signal_history: List[np.ndarray] = field(default_factory=list)
    regime_belief: float = 0.0
    lookback: int = 20
    
    def update_energy(self, new_signals: np.ndarray):
        """Rolling energy: momentum of signal changes"""
        if len(self.signal_history) > 0:
            prev = self.signal_history[-1]
            delta = new_signals - prev
            # Energy = accumulated signal momentum
            self.energy = 0.9 * self.energy + 0.1 * np.mean(np.abs(delta))
        
        self.signal_history.append(new_signals.copy())
        if len(self.signal_history) > self.lookback:
            self.signal_history.pop(0)


# =============================================================================
# BASE CLASS
# =============================================================================

class EnergyModel(ABC):
    """
    Base class for stateful energy models.
    
    Full duplex I/O:
    - Receives weight + directive from HRM
    - Sends report back to HRM
    """
    
    def __init__(self, name: str, n_assets: int = 43, lookback: int = 20):
        self.name = name
        self.n_assets = n_assets
        self.state = ModelState(lookback=lookback)
        self.weight = 1.0 / 3  # Default equal weight
    
    def compute_signals(self, 
                        instrument_data: np.ndarray,
                        weight: float = None,
                        directive: ModelDirective = None) -> Tuple[np.ndarray, ModelReport]:
        """
        Main entry point with full duplex I/O.
        
        Args:
            instrument_data: [n_assets, n_features] market state
            weight: Model weight from HRM
            directive: Control signals from HRM
        
        Returns:
            signals: [n_assets] position signals
            report: State summary for HRM
        """
        if weight is not None:
            self.weight = weight
        
        # Core signal computation (implemented by subclass)
        raw_signals = self._compute_raw_signals(instrument_data)
        
        # Apply directive constraints
        if directive is not None:
            raw_signals = self._apply_directive(raw_signals, directive)
        
        # Scale by weight
        signals = raw_signals * self.weight
        
        # Update internal state
        self.state.update_energy(signals)
        
        # Generate report
        report = self._generate_report(signals)
        
        return signals, report
    
    @abstractmethod
    def _compute_raw_signals(self, instrument_data: np.ndarray) -> np.ndarray:
        """Subclass implements core signal logic"""
        pass
    
    def _apply_directive(self, signals: np.ndarray, directive: ModelDirective) -> np.ndarray:
        """Apply HRM control signals"""
        # Risk limiting
        signals = signals * directive.risk_limit
        
        # Focus on specific assets if requested
        if directive.focus_assets is not None:
            mask = np.zeros(self.n_assets)
            mask[directive.focus_assets] = 1.0
            signals = signals * mask
        
        return signals
    
    def _generate_report(self, signals: np.ndarray) -> ModelReport:
        """Generate state summary for HRM"""
        # Energy from accumulated momentum
        energy_level = np.clip(self.state.energy, 0, 1)
        
        # Signal entropy (diversity)
        abs_signals = np.abs(signals) + 1e-8
        p = abs_signals / np.sum(abs_signals)
        entropy = -np.sum(p * np.log(p + 1e-8)) / np.log(self.n_assets)
        
        # Regime confidence (how much signal agreement)
        regime_confidence = np.abs(np.mean(np.sign(signals)))
        
        # Performance estimate (self-assessed)
        if len(self.state.signal_history) > 1:
            # Estimate from signal stability
            recent = np.array(self.state.signal_history[-5:])
            stability = 1.0 - np.mean(np.std(recent, axis=0))
            performance_estimate = stability * np.mean(np.abs(signals))
        else:
            performance_estimate = 0.0
        
        # Active assets
        n_active = np.sum(np.abs(signals) > 0.01)
        
        return ModelReport(
            energy_level=energy_level,
            signal_entropy=entropy,
            regime_confidence=regime_confidence,
            performance_estimate=performance_estimate,
            n_active_assets=n_active
        )
    
    def reset_state(self):
        """Reset internal state"""
        self.state = ModelState(lookback=self.state.lookback)


# =============================================================================
# CONCRETE MODELS
# =============================================================================

class VolatilityBreakoutModel(EnergyModel):
    """
    Volatility × Breakout (PROVEN WINNER: $37,308 vs Grid $21,800)
    
    Signal = volatility_signal × breakout_direction
    
    Stateful: Tracks signal history and energy momentum.
    """
    
    def __init__(self, n_assets: int = 43, lookback: int = 20):
        super().__init__("volatility_breakout", n_assets, lookback)
    
    def _compute_raw_signals(self, instrument_data: np.ndarray) -> np.ndarray:
        """
        instrument_data columns:
        [open, high, low, close, volume, returns, volatility, momentum, rsi, ma_ratio]
        """
        signals = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            # Volatility signal (normalized 0-1)
            volatility = instrument_data[i, 6]
            vol_signal = np.clip(volatility, 0, 1)
            
            # Breakout direction: where is price in recent range?
            high = instrument_data[i, 1]
            low = instrument_data[i, 2]
            close = instrument_data[i, 3]
            
            price_position = (close - low) / (high - low + 1e-8)
            breakout_signal = 2 * price_position - 1  # -1 to 1
            
            # Multiply (AND logic - both must agree)
            signals[i] = vol_signal * breakout_signal
        
        return signals


class MomentumTrendModel(EnergyModel):
    """
    Momentum × Trend
    
    Signal = momentum_strength × trend_direction
    """
    
    def __init__(self, n_assets: int = 43, lookback: int = 20):
        super().__init__("momentum_trend", n_assets, lookback)
    
    def _compute_raw_signals(self, instrument_data: np.ndarray) -> np.ndarray:
        signals = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            momentum = instrument_data[i, 7]
            ma_ratio = instrument_data[i, 9]
            
            # Trend direction from MA ratio
            trend = np.sign(ma_ratio - 1.0)
            
            # Momentum strength
            strength = np.clip(np.abs(momentum), 0, 1)
            
            signals[i] = strength * trend
        
        return signals


class MeanReversionModel(EnergyModel):
    """
    Mean Reversion
    
    Signal = -deviation_from_mean (buy dips, sell rips)
    """
    
    def __init__(self, n_assets: int = 43, lookback: int = 20, threshold: float = 0.02):
        super().__init__("mean_reversion", n_assets, lookback)
        self.threshold = threshold
    
    def _compute_raw_signals(self, instrument_data: np.ndarray) -> np.ndarray:
        signals = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            ma_ratio = instrument_data[i, 9]
            rsi = instrument_data[i, 8]
            
            # Deviation from mean
            deviation = ma_ratio - 1.0
            
            # RSI signal
            rsi_signal = (50 - rsi) / 50  # Oversold = bullish
            
            # Mean reversion
            if np.abs(deviation) > self.threshold:
                signals[i] = -np.sign(deviation) * min(np.abs(deviation) / self.threshold, 1.0)
            
            # Blend with RSI
            signals[i] = 0.5 * signals[i] + 0.5 * rsi_signal
        
        return signals


# =============================================================================
# SWARM (Collection of Models)
# =============================================================================

class SwarmModel:
    """
    Swarm of Stateful Energy Models with Full Duplex I/O.
    
    Collects signals from all models, aggregates by HRM weights,
    and provides combined reports back to HRM.
    """
    
    def __init__(self, n_assets: int = 43, lookback: int = 20):
        self.models = [
            VolatilityBreakoutModel(n_assets, lookback),
            MomentumTrendModel(n_assets, lookback),
            MeanReversionModel(n_assets, lookback),
        ]
        self.n_models = len(self.models)
        self.n_assets = n_assets
        
        # Report dimension for each model
        self.report_dim = 5  # energy, entropy, confidence, perf, active_ratio
    
    def compute_signals(self,
                        instrument_data: np.ndarray,
                        weights: np.ndarray = None,
                        directives: List[ModelDirective] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full duplex signal computation.
        
        Args:
            instrument_data: [n_assets, n_features]
            weights: [n_models] from HRM
            directives: [n_models] from HRM
        
        Returns:
            combined_signals: [n_assets]
            reports: [n_models, report_dim] for HRM
        """
        if weights is None:
            weights = np.ones(self.n_models) / self.n_models
        
        if directives is None:
            directives = [ModelDirective() for _ in range(self.n_models)]
        
        all_signals = []
        all_reports = []
        
        for i, (model, w, d) in enumerate(zip(self.models, weights, directives)):
            signals, report = model.compute_signals(instrument_data, weight=w, directive=d)
            all_signals.append(signals)
            all_reports.append(report.to_array())
        
        # Combine signals (weighted sum)
        combined = np.sum(np.array(all_signals) * weights[:, np.newaxis], axis=0)
        
        # Stack reports for HRM
        reports = np.stack(all_reports, axis=0)
        
        return combined, reports
    
    def reset_all_states(self):
        """Reset all model states"""
        for model in self.models:
            model.reset_state()
    
    def get_model_names(self) -> List[str]:
        return [m.name for m in self.models]


# =============================================================================
# CONVERSION UTILS
# =============================================================================

def reports_to_hrm_input(reports: np.ndarray) -> np.ndarray:
    """
    Convert model reports to HRM input format.
    
    Args:
        reports: [n_models, report_dim]
    
    Returns:
        hrm_input: [n_models * report_dim] flattened
    """
    return reports.flatten()


def hrm_output_to_weights(output: np.ndarray, n_models: int = 3) -> np.ndarray:
    """
    Convert HRM output to model weights (softmax normalized).
    
    Args:
        output: [n_models] or larger
    
    Returns:
        weights: [n_models] valid probability distribution
    """
    raw = output[:n_models]
    exp = np.exp(raw - np.max(raw))
    return exp / np.sum(exp)


if __name__ == "__main__":
    print("Testing Stateful Energy Models with Full Duplex I/O...\n")
    
    # Create swarm
    swarm = SwarmModel(n_assets=43, lookback=20)
    
    # Fake instrument data [43 assets, 10 features]
    np.random.seed(42)
    instrument_data = np.random.randn(43, 10) * 0.1 + 0.5
    
    # Test with equal weights
    signals, reports = swarm.compute_signals(instrument_data)
    
    print(f"Model names: {swarm.get_model_names()}")
    print(f"Instrument data shape: {instrument_data.shape}")
    print(f"Combined signals shape: {signals.shape}")
    print(f"Reports shape: {reports.shape}")
    print()
    
    print("Per-model reports:")
    for i, name in enumerate(swarm.get_model_names()):
        print(f"  {name}:")
        print(f"    energy: {reports[i, 0]:.3f}")
        print(f"    entropy: {reports[i, 1]:.3f}")
        print(f"    confidence: {reports[i, 2]:.3f}")
        print(f"    perf_est: {reports[i, 3]:.3f}")
    
    print()
    print("Signal range:", signals.min().round(3), "to", signals.max().round(3))
    
    # Test with custom weights and directives
    print("\n--- With custom weights and directives ---")
    weights = np.array([0.7, 0.2, 0.1])  # Favor volatility_breakout
    directives = [
        ModelDirective(regime_hint=-0.5, risk_limit=0.8),  # Trending, reduce risk
        ModelDirective(regime_hint=0.0, risk_limit=1.0),
        ModelDirective(regime_hint=0.5, risk_limit=0.5),   # Ranging, reduce risk
    ]
    
    signals2, reports2 = swarm.compute_signals(instrument_data, weights, directives)
    print(f"Signals with custom weights: {signals2[:5]}")
