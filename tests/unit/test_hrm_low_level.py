"""
Unit tests for low-level HRM module.
"""
import pytest
import numpy as np
from core.hrm.low_level import (
    LowLevelModule,
    LowLevelConfig,
    LowLevelFeature,
    LowLevelProcessor
)


class TestLowLevelModule:
    """Test low-level module"""
    
    def test_initialization(self):
        """Test module initialization"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        assert module.config.n_features == 15
        assert module.buffer.shape == (config.lookback, config.hidden_dim)
        assert module.buffer_idx == 0
    
    def test_forward_pass(self):
        """Test forward pass"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        input_features = np.random.randn(config.n_assets, config.n_features)
        context = np.random.randn(config.hidden_dim)
        
        processed, metadata = module.forward(input_features, context)
        
        # Check output shape
        assert processed.shape == (config.hidden_dim,)
        
        # Check metadata
        assert 'buffer_size' in metadata
        assert 'extracted_norm' in metadata
    
    def test_extract_features(self):
        """Test feature extraction"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        # Test with single asset
        single_asset = np.random.randn(config.n_features)
        extracted = module._extract_features(single_asset)
        
        assert extracted.shape == (config.hidden_dim,)
        
        # Test with multiple assets
        multi_asset = np.random.randn(config.n_assets, config.n_features)
        extracted = module._extract_features(multi_asset)
        
        assert extracted.shape == (config.hidden_dim,)
    
    def test_update_buffer(self):
        """Test buffer update"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        features = np.random.randn(config.hidden_dim)
        
        # Update buffer multiple times
        for i in range(config.lookback + 5):
            module._update_buffer(features)
        
        # Check buffer is not all zeros
        assert not np.all(module.buffer == 0.0)
    
    def test_process_buffer(self):
        """Test buffer processing"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        # Fill buffer with some data
        for i in range(10):
            features = np.random.randn(config.hidden_dim)
            module._update_buffer(features)
        
        processed = module._process_buffer()
        
        # Check output shape
        assert processed.shape == (config.hidden_dim,)
    
    def test_reset(self):
        """Test reset functionality"""
        config = LowLevelConfig()
        module = LowLevelModule(config)
        
        # Modify buffer
        module.buffer = np.ones((config.lookback, config.hidden_dim))
        module.buffer_idx = 5
        
        # Reset
        module.reset()
        
        # Check buffer is reset
        assert np.all(module.buffer == 0.0)
        assert module.buffer_idx == 0


class TestLowLevelProcessor:
    """Test low-level processor"""
    
    def test_initialization(self):
        """Test processor initialization"""
        config = LowLevelConfig()
        processor = LowLevelProcessor(config)
        
        assert len(processor.modules) == 2
        assert processor.active_module == 'primary'
    
    def test_process(self):
        """Test processing"""
        config = LowLevelConfig()
        processor = LowLevelProcessor(config)
        
        input_data = np.random.randn(config.n_assets, config.n_features)
        
        feature = processor.process(input_data)
        
        # Check feature properties
        assert feature.features.shape == (config.hidden_dim,)
        assert 0.0 <= feature.confidence <= 1.0
        assert 'extracted_norm' in feature.metadata
    
    def test_switch_module(self):
        """Test module switching"""
        config = LowLevelConfig()
        processor = LowLevelProcessor(config)
        
        assert processor.active_module == 'primary'
        
        processor.switch_module()
        assert processor.active_module == 'secondary'
        
        processor.switch_module()
        assert processor.active_module == 'primary'
    
    def test_reset(self):
        """Test reset functionality"""
        config = LowLevelConfig()
        processor = LowLevelProcessor(config)
        
        # Modify state
        processor.modules['primary'].buffer = np.ones((config.lookback, config.hidden_dim))
        
        # Reset
        processor.reset()
        
        # Check all modules reset
        for module in processor.modules.values():
            assert np.all(module.buffer == 0.0)


class TestLowLevelFeature:
    """Test low-level feature class"""
    
    def test_feature_creation(self):
        """Test feature creation"""
        config = LowLevelConfig()
        features = np.random.randn(config.hidden_dim)
        confidence = 0.8
        metadata = {'test': 'value'}
        
        feature = LowLevelFeature(features, confidence, metadata)
        
        assert np.allclose(feature.features, features)
        assert feature.confidence == confidence
        assert feature.metadata == metadata
    
    def test_to_dict(self):
        """Test conversion to dict"""
        config = LowLevelConfig()
        features = np.random.randn(config.hidden_dim)
        confidence = 0.8
        metadata = {'test': 'value'}
        
        feature = LowLevelFeature(features, confidence, metadata)
        feature_dict = feature.to_dict()
        
        assert 'features' in feature_dict
        assert 'confidence' in feature_dict
        assert 'metadata' in feature_dict
        assert feature_dict['confidence'] == confidence


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_low_level_module(self):
        """Test creating low-level module"""
        from core.hrm.low_level import create_low_level_module
        
        module = create_low_level_module()
        assert isinstance(module, LowLevelModule)
    
    def test_create_low_level_processor(self):
        """Test creating low-level processor"""
        from core.hrm.low_level import create_low_level_processor
        
        processor = create_low_level_processor()
        assert isinstance(processor, LowLevelProcessor)


if __name__ == "__main__":
    pytest.main([__file__])