"""
Signal generation module - creates trading signals from data.

Pure logic, no framework dependencies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class SignalConfig:
    """Signal generation configuration"""
    n_models: int = 5
    n_regimes: int = 6
    confidence_threshold: float = 0.3
    signal_decay: float = 0.9  # Weight for recent signals


@dataclass
class TradingSignal:
    """Trading signal structure"""
    timestamp: datetime
    symbol: str
    regime: str  # TREND, MEAN_REVERSION, VOLATILITY, STAT_ARB, SYSTEMATIC, ML
    signal_strength: float  # [-1, 1] short to long
    confidence: float  # [0, 1]
    model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """
    Generate trading signals from data.
    
    Responsibilities:
    - Signal computation from features
    - Regime classification
    - Confidence calculation
    - Signal aggregation
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        
    def generate_signals(self,
                        features: np.ndarray,
                        timestamp: datetime,
                        symbol: str,
                        metadata: Dict[str, Any] = None) -> List[TradingSignal]:
        """
        Generate signals for given features.
        
        Args:
            features: Feature array
            timestamp: Current timestamp
            symbol: Asset symbol
            metadata: Additional metadata
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Generate signals for each model
        for model_idx in range(self.config.n_models):
            signal = self._generate_single_signal(
                features, timestamp, symbol, model_idx, metadata
            )
            if signal:
                signals.append(signal)
        
        return signals
    
    def _generate_single_signal(self,
                               features: np.ndarray,
                               timestamp: datetime,
                               symbol: str,
                               model_idx: int,
                               metadata: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal for single model"""
        if metadata is None:
            metadata = {}
        
        # Determine regime based on features
        regime = self._classify_regime(features, model_idx)
        
        # Generate signal based on regime
        signal_strength, confidence = self._compute_signal_and_confidence(
            features, regime, model_idx
        )
        
        # Apply confidence threshold
        if abs(signal_strength) < self.config.confidence_threshold:
            return None
        
        # Create signal
        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            signal_strength=signal_strength,
            confidence=confidence,
            model_id=f"model_{model_idx}",
            metadata=metadata
        )
        
        # Store in history
        key = f"{symbol}_{model_idx}"
        if key not in self.signal_history:
            self.signal_history[key] = []
        self.signal_history[key].append(signal)
        
        # Keep history manageable
        if len(self.signal_history[key]) > 100:
            self.signal_history[key] = self.signal_history[key][-100:]
        
        return signal
    
    def _classify_regime(self, features: np.ndarray, model_idx: int) -> str:
        """Classify market regime based on features"""
        if len(features.shape) > 1:
            # Use last timestep
            features = features[-1]
        
        # Simple regime classification based on feature patterns
        # This would be more sophisticated in production
        
        regimes = ["trend", "mean_reversion", "volatility", "stat_arb", "systematic", "ml"]
        
        # Determine which features are prominent
        if len(features) >= 15:
            # Assuming first few features are returns, volatility, etc.
            returns = features[5] if len(features) > 5 else 0.0
            volatility = features[6] if len(features) > 6 else 0.0
            rsi = features[7] if len(features) > 7 else 0.5
            
            if abs(returns) > 0.02:
                return "trend"
            elif volatility > 0.05:
                return "volatility"
            elif rsi < 0.3 or rsi > 0.7:
                return "mean_reversion"
            elif model_idx == 3:
                return "stat_arb"
            elif model_idx == 4:
                return "systematic"
            else:
                return "ml"
        
        return regimes[model_idx % len(regimes)]
    
    def _compute_signal_and_confidence(self,
                                      features: np.ndarray,
                                      regime: str,
                                      model_idx: int) -> Tuple[float, float]:
        """Compute signal strength and confidence"""
        if len(features.shape) > 1:
            features = features[-1]
        
        # Base signal computation
        if regime == "trend":
            # Momentum-based
            if len(features) > 5:
                signal = 0.5 * features[5]  # Returns
            else:
                signal = np.random.uniform(-0.3, 0.3)
            confidence = 0.6
        
        elif regime == "mean_reversion":
            # Mean reversion
            if len(features) > 7:
                rsi = features[7]
                if rsi < 0.3:
                    signal = 0.8  # Oversold, buy
                elif rsi > 0.7:
                    signal = -0.8  # Overbought, sell
                else:
                    signal = 0.0
            else:
                signal = np.random.uniform(-0.2, 0.2)
            confidence = 0.7
        
        elif regime == "volatility":
            # Volatility breakout
            if len(features) > 6:
                volatility = features[6]
                if volatility > 0.05:
                    signal = 0.6 if np.random.random() > 0.5 else -0.6
                else:
                    signal = 0.0
            else:
                signal = np.random.uniform(-0.4, 0.4)
            confidence = 0.5
        
        elif regime == "stat_arb":
            # Statistical arbitrage (simplified)
            if len(features) > 10:
                # Use correlation or mean reversion patterns
                signal = np.random.uniform(-0.5, 0.5)
            else:
                signal = np.random.uniform(-0.3, 0.3)
            confidence = 0.4
        
        elif regime == "systematic":
            # Systematic strategy (DCA, rebalance)
            if len(features) > 8:
                price_trend = features[8] if len(features) > 8 else 0.0
                if abs(price_trend) > 0.01:
                    signal = 0.4 * np.sign(price_trend)
                else:
                    signal = 0.0
            else:
                signal = np.random.uniform(-0.2, 0.2)
            confidence = 0.5
        
        else:  # ML
            # ML-based signals
            if len(features) > 0:
                signal = np.random.uniform(-0.6, 0.6)
            else:
                signal = 0.0
            confidence = 0.3
        
        # Apply model-specific adjustments
        confidence *= (1.0 - 0.1 * model_idx)  # Older models less confident
        
        # Ensure bounds
        signal = np.clip(signal, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return signal, confidence
    
    def aggregate_signals(self,
                         signals: List[TradingSignal],
                         weights: Optional[np.ndarray] = None) -> Tuple[float, float, str]:
        """
        Aggregate multiple signals.
        
        Args:
            signals: List of signals to aggregate
            weights: Optional weights for each signal
            
        Returns:
            (aggregated_signal, aggregated_confidence, dominant_regime)
        """
        if not signals:
            return 0.0, 0.0, "none"
        
        if weights is None:
            weights = np.ones(len(signals)) / len(signals)
        
        # Weighted aggregation
        total_signal = 0.0
        total_confidence = 0.0
        
        for signal, weight in zip(signals, weights):
            total_signal += signal.signal_strength * weight * signal.confidence
            total_confidence += signal.confidence * weight
        
        # Normalize
        total_confidence = min(total_confidence, 1.0)
        
        # Determine dominant regime
        regime_counts: Dict[str, float] = {}
        for signal in signals:
            regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + signal.confidence
        
        if regime_counts:
            dominant_regime = max(regime_counts, key=regime_counts.get)
        else:
            dominant_regime = "none"
        
        return total_signal, total_confidence, dominant_regime
    
    def compute_regime_weights(self,
                              signals: List[TradingSignal],
                              current_regime: str = None) -> np.ndarray:
        """
        Compute regime weights from signals.
        
        Args:
            signals: List of signals
            current_regime: Current market regime
            
        Returns:
            Regime weights [n_regimes]
        """
        n_regimes = self.config.n_regimes
        regime_names = ["trend", "mean_reversion", "volatility", "stat_arb", "systematic", "ml"]
        
        # Initialize weights
        weights = np.zeros(n_regimes)
        
        if not signals:
            # Return balanced weights
            return np.ones(n_regimes) / n_regimes
        
        # Aggregate by regime
        regime_confidence: Dict[str, float] = {}
        for signal in signals:
            if signal.regime in regime_names:
                idx = regime_names.index(signal.regime)
                weights[idx] += signal.confidence
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Balanced weights
            weights = np.ones(n_regimes) / n_regimes
        
        return weights
    
    def filter_signals(self,
                      signals: List[TradingSignal],
                      min_confidence: float = 0.0,
                      min_signal_strength: float = 0.0) -> List[TradingSignal]:
        """Filter signals by confidence and signal strength"""
        filtered = []
        for signal in signals:
            if (abs(signal.confidence) >= min_confidence and 
                abs(signal.signal_strength) >= min_signal_strength):
                filtered.append(signal)
        return filtered
    
    def get_recent_signals(self,
                          symbol: str,
                          model_idx: Optional[int] = None,
                          limit: int = 10) -> List[TradingSignal]:
        """Get recent signals for symbol"""
        if model_idx is not None:
            key = f"{symbol}_{model_idx}"
            return self.signal_history.get(key, [])[-limit:]
        
        # Combine all models
        all_signals = []
        for model_idx in range(self.config.n_models):
            key = f"{symbol}_{model_idx}"
            if key in self.signal_history:
                all_signals.extend(self.signal_history[key])
        
        # Sort by timestamp
        all_signals.sort(key=lambda s: s.timestamp)
        return all_signals[-limit:]


class SignalAggregator:
    """
    Aggregates signals from multiple sources.
    
    Handles combining signals from different models and regimes.
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        
    def aggregate_from_features(self,
                               features: np.ndarray,
                               timestamp: datetime,
                               symbol: str) -> TradingSignal:
        """
        Generate and aggregate signals from features.
        
        Args:
            features: Feature array
            timestamp: Current timestamp
            symbol: Asset symbol
            
        Returns:
            Aggregated trading signal
        """
        # Generate signals
        signals = self.signal_generator.generate_signals(features, timestamp, symbol)
        
        # Filter weak signals
        filtered = self.signal_generator.filter_signals(
            signals, 
            min_confidence=self.config.confidence_threshold
        )
        
        # Aggregate
        signal_strength, confidence, regime = self.signal_generator.aggregate_signals(filtered)
        
        # Create aggregated signal
        aggregated_signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            signal_strength=signal_strength,
            confidence=confidence,
            model_id="aggregated",
            metadata={
                'n_signals': len(filtered),
                'source_models': [s.model_id for s in filtered]
            }
        )
        
        return aggregated_signal
    
    def compute_regime_strategy(self,
                               signals: List[TradingSignal],
                               regime_weights: np.ndarray) -> Dict[str, Any]:
        """
        Compute strategy based on regime weights.
        
        Args:
            signals: List of signals
            regime_weights: Regime weights
            
        Returns:
            Strategy parameters
        """
        if not signals:
            return {
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'regime_focus': 'neutral'
            }
        
        # Aggregate signals per regime
        regime_signals: Dict[str, List[TradingSignal]] = {}
        for signal in signals:
            if signal.regime not in regime_signals:
                regime_signals[signal.regime] = []
            regime_signals[signal.regime].append(signal)
        
        # Compute regime-specific strategies
        regime_strategies = {}
        for regime, regime_sig_list in regime_signals.items():
            avg_signal = np.mean([s.signal_strength for s in regime_sig_list])
            avg_confidence = np.mean([s.confidence for s in regime_sig_list])
            
            regime_strategies[regime] = {
                'signal': avg_signal,
                'confidence': avg_confidence,
                'weight': regime_weights[regime] if regime in regime_weights else 0.0
            }
        
        # Determine dominant regime
        max_weight_idx = np.argmax(regime_weights)
        regimes = ["trend", "mean_reversion", "volatility", "stat_arb", "systematic", "ml"]
        dominant_regime = regimes[max_weight_idx]
        
        # Compute strategy parameters based on dominant regime
        if dominant_regime == "trend":
            position_size = 0.8
            stop_loss = 0.02
            take_profit = 0.05
        elif dominant_regime == "mean_reversion":
            position_size = 0.5
            stop_loss = 0.03
            take_profit = 0.04
        elif dominant_regime == "volatility":
            position_size = 0.3
            stop_loss = 0.04
            take_profit = 0.08
        elif dominant_regime == "stat_arb":
            position_size = 0.4
            stop_loss = 0.025
            take_profit = 0.035
        elif dominant_regime == "systematic":
            position_size = 0.6
            stop_loss = 0.015
            take_profit = 0.025
        else:  # ML
            position_size = 0.2
            stop_loss = 0.05
            take_profit = 0.1
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'dominant_regime': dominant_regime,
            'regime_strategies': regime_strategies
        }


# Factory functions
def create_signal_generator(config: SignalConfig = None) -> SignalGenerator:
    """Factory function to create signal generator"""
    if config is None:
        config = SignalConfig()
    return SignalGenerator(config)


def create_signal_aggregator(config: SignalConfig = None) -> SignalAggregator:
    """Factory function to create signal aggregator"""
    if config is None:
        config = SignalConfig()
    return SignalAggregator(config)