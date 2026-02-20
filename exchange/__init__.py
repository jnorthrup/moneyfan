"""
Exchange Integration Module
===========================

Handles communication between Python MLX system and Kotlin execution layer.

Architecture:
- Python/MLX: Generates signals via 24 codecs
- SignalWriter: Writes signals to stdout as JSON
- Kotlin: Reads stdin, executes via coinbaseXChangeBot.main.kts
"""

from .signal_writer import SignalWriter

__all__ = ['SignalWriter']