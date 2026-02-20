"""
Unit tests for high-level HRM module.
"""
import pytest
import numpy as np
from core.hrm.high_level import (
    HighLevelModule, 
    HighLevelConfig,
    HighLevelDecision,
    HighLevelController
)


class TestHighLevelModule:
    """Test high-level module"""
    
    def test_initialization(self):
        """Test module initialization"""
        config = HighLevelConfig()
        module = HighLevelModule(config)
        
        assert module.config.n_regimes == 6
        assert np.all(module.state == 0.0)
        assert len(module.memory) == 0
    
    def test_forward_pass(self):
        """Test forward pass"""
        config = HighLevelConfig()
        module = HighLevelModule(config)
        
        low_level_output = np.random.randn(config.hidden_dim)
        context = np.random.randn(config.hidden_dim)
        
        regime_weights, metadata = module.forward(low_level_output, context)
        
        # Check output shape
        assert regime_weights.shape == (config.n_regimes,)
        
        # Check probabilities sum to 1
        assert np.isclose(regime_weights.sum(), 1.0)
        
        # Check confidence in metadata
        assert 'state_norm' in metadata
        assert 'memory_size' in metadata
        
    def test_reset(self):
        """Test reset functionality"""
        config = HighLevelConfig()
        module = HighLevelModule(config)
        
        # Add some state
        module.state = np.ones(config.hidden_dim)
        module.memory.append(np.ones(config.hidden_dim))
        
        # Reset
        module.reset()
        
        # Check state is reset
        assert np.all(module.state == 0.0)
        assert len(module.memory) == 0
    
    def test_compute_confidence(self):
        """Test confidence computation"""
        config = HighLevelConfig()
        controller = HighLevelController(config)
        
        # Test with uniform weights
        uniform_weights = np.ones(config.n_regimes) / config.n_regimes
        confidence = controller._compute_confidence(uniform_weights)
        
        # Low confidence for uniform distribution
        assert confidence < 0.5
        
        # Test with concentrated weights
        concentrated_weights = np.zeros(config.n_regimes)
        concentrated_weights[0] = 1.0
        confidence = controller._compute_confidence(concentrated_weights)
        
        # High confidence for concentrated distribution
        assert confidence > 0.8


class TestHighLevelController:
    """Test high-level controller"""
    
    def test_initialization(self):
        """Test controller initialization"""
        config = HighLevelConfig()
        controller = HighLevelController(config)
        
        assert len(controller.modules) == 2
        assert controller.active_module == 'primary'
    
    def test_decide(self):
        """Test decision making"""
        config = HighLevelConfig()
        controller = HighLevelController(config)
        
        signals = np.random.randn(config.n_models, 10)
        market_state = np.random.randn(20)
        
        decision = controller.decide(signals, market_state)
        
        # Check decision properties
        assert decision.regime_weights.shape == (config.n_regimes,)
        assert 0.0 <= decision.confidence <= 1.0
        assert 'regime_distribution' in decision.metadata
    
    def test_switch_module(self):
        """Test module switching"""
        config = HighLevelConfig()
        controller = HighLevelController(config)
        
        assert controller.active_module == 'primary'
        
        controller.switch_module()
        assert controller.active_module == 'backup'
        
        controller.switch_module()
        assert controller.active_module == 'primary'
    
    def test_reset(self):
        """Test reset functionality"""
        config = HighLevelConfig()
        controller = HighLevelController(config)
        
        # Modify state
        controller.modules['primary'].state = np.ones(config.hidden_dim)
        
        # Reset
        controller.reset()
        
        # Check all modules reset
        for module in controller.modules.values():
            assert np.all(module.state == 0.0)


class TestHighLevelDecision:
    """Test high-level decision class"""
    
    def test_decision_creation(self):
        """Test decision creation"""
        config = HighLevelConfig()
        regime_weights = np.ones(config.n_regimes) / config.n_regimes
        confidence = 0.8
        metadata = {'test': 'value'}
        
        decision = HighLevelDecision(regime_weights, confidence, metadata)
        
        assert np.allclose(decision.regime_weights, regime_weights)
        assert decision.confidence == confidence
        assert decision.metadata == metadata
    
    def test_to_dict(self):
        """Test conversion to dict"""
        config = HighLevelConfig()
        regime_weights = np.ones(config.n_regimes) / config.n_regimes
        confidence = 0.8
        metadata = {'test': 'value'}
        
        decision = HighLevelDecision(regime_weights, confidence, metadata)
        decision_dict = decision.to_dict()
        
        assert 'regime_weights' in decision_dict
        assert 'confidence' in decision_dict
        assert 'metadata' in decision_dict
        assert decision_dict['confidence'] == confidence


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_high_level_module(self):
        """Test creating high-level module"""
        from core.hrm.high_level import create_high_level_module
        
        module = create_high_level_module()
        assert isinstance(module, HighLevelModule)
    
    def test_create_high_level_controller(self):
        """Test creating high-level controller"""
        from core.hrm.high_level import create_high_level_controller
        
        controller = create_high_level_controller()
        assert isinstance(controller, HighLevelController)


if __name__ == "__main__":
    pytest.main([__file__])