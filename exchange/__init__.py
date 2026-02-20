"""
Exchange Integration Module
===========================

Handles communication between Python MLX system and execution layer.

Architecture:
- Python/MLX: Generates signals via 24 codecs
- SignalWriter: Writes signals to stdout as JSON
- Execution: Receives signals and executes via exchange API
"""

from .signal_writer import SignalWriter

__all__ = ['SignalWriter']