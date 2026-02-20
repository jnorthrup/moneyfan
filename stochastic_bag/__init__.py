"""
Stochastic Bag and Compass Module
==================================

Implements:
1. Dirichlet sampling for codec weights
2. Bag resampling with correlation matrix
3. GBM and OU equations for price path simulation
4. Correlation matrix calculations
"""

from .compass import StochasticCompass
from .resampler import StochasticBagResampler
from .correlation import CorrelationMatrix

__all__ = ['StochasticCompass', 'StochasticBagResampler', 'CorrelationMatrix']