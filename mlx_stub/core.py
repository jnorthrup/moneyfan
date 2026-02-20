"""Minimal stub for mlx.core used in tests.
Provides numpy-backed equivalents for required functions.
"""
import numpy as np

# Device placeholder
class _Device:
    pass

gpu = _Device()

def set_default_device(dev):
    # No-op for stub
    return None

# Array creation / conversion
array = np.array
zeros = np.zeros

# Random utilities
class random:
    @staticmethod
    def normal(shape, dtype=np.float32):
        return np.random.normal(size=shape).astype(dtype)

# Math utilities
clip = np.clip
cos = np.cos
sin = np.sin
exp = np.exp
log = np.log
arange = np.arange
outer = np.outer
concatenate = np.concatenate
sqrt = np.sqrt
rsqrt = lambda x: 1.0/np.sqrt(x)
mean = np.mean
square = np.square

# Softmax implementation
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# No-op gradient utilities
stop_gradient = lambda x: x

def eval(*args, **kwargs):
    return None

# value_and_grad returns a function that returns (loss, dummy grads dict)
def value_and_grad(f):
    def wrapper(*a, **kw):
        loss = f(*a, **kw)
        # Return empty dict for grads to satisfy usage
        return loss, {}
    return wrapper

# Context manager for no_grad
class _NoGrad:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def no_grad():
    return _NoGrad()
