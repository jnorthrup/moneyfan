"""Minimal stub for mlx.optimizers used in tests.
Provides placeholder optimizer classes with no-op update.
"""

class Adam:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
    def update(self, model, grads):
        # No-op: pretend parameters are updated
        return None

class AdamW(Adam):
    def __init__(self, learning_rate=1e-3, weight_decay=0.0):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
