"""
Feature Extraction Module
===========================

Clean separation between:
- PandasFeatureExtractor: Computes technical indicators via pandas
- HRMFeatureEncoder: Encodes features for HRM model consumption

Usage:
    from feature_extraction import PandasFeatureExtractor, HRMFeatureEncoder

    extractor = PandasFeatureExtractor()
    encoded = extractor.encode(df)

    encoder = HRMFeatureEncoder(n_codecs=24)
    features = encoder.prepare(encoded)
"""

from .pandas_extractor import PandasFeatureExtractor, EncodedDataFrame
from .hrm_encoder import HRMFeatureEncoder

__all__ = [
    'PandasFeatureExtractor',
    'EncodedDataFrame',
    'HRMFeatureEncoder',
]