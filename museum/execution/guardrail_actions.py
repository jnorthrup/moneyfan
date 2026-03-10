"""
Guardrail Actions Module
========================

Maps guardrail states to runtime actions that modify trading behavior.
This module provides deterministic action mapping for drawdown guardrail states.

Guardrail States:
- normal: No action required, trading proceeds normally
- warn: Log warning, allow trading but with increased monitoring
- derisk: Reduce position sizes, limit new entries, increase confidence threshold
- halt: Stop all trading activity immediately

Runtime Actions per State:
- normal: Full trading allowed
- warn: Log warning, may optionally increase monitoring but allow normal trading
- derisk: 
  - Scale down position sizes (e.g., 50% of normal)
  - Increase confidence threshold
  - Limit top-k new entries per iteration
- halt:
  - Stop all new position entries
  - Close existing positions based on halt policy
  - Save state and exit

Usage:
    from execution.guardrail_actions import GuardrailActionMapper
    
    mapper = GuardrailActionMapper(config)
    action = mapper.get_action_for_state(guardrail_state)
    
    # Apply action modifications to trading parameters
    modified_params = mapper.apply_action(
        current_params,
        action,
        top_k=engine.config.top_k,
        risk_per_trade=engine.config.risk_per_trade,
        signal_threshold=engine.config.signal_threshold
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class GuardrailState(str, Enum):
    """Guardrail state enumeration matching TradingEngine.guardrail_state."""
    NORMAL = "normal"
    WARN = "warn"
    DERISK = "derisk"
    HALT = "halt"


class GuardrailAction(str, Enum):
    """Runtime action to take based on guardrail state."""
    NORMAL = "normal"
    WARN = "warn"
    DERISK = "derisk"
    HALT = "halt"


@dataclass
class GuardrailActionConfig:
    """Configuration for guardrail action behavior."""
    # De-risk scaling factors
    derisk_position_size_scale: float = 0.5  # Scale position sizes to 50% when derisking
    derisk_top_k_scale: float = 0.5  # Scale top-k to 50% when derisking
    derisk_confidence_boost: float = 0.1  # Increase confidence threshold by this amount
    
    # Warn behavior
    warn_log_only: bool = True  # Warn state only logs, doesn't modify behavior
    
    # Halt behavior
    halt_close_positions: bool = False  # Whether to close existing positions on halt


@dataclass
class GuardrailActionResult:
    """Result of applying guardrail action to trading parameters."""
    action: GuardrailAction
    position_size_scale: float
    top_k: int
    signal_threshold: float
    allow_new_entries: bool
    guardrail_state: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "position_size_scale": self.position_size_scale,
            "top_k": self.top_k,
            "signal_threshold": self.signal_threshold,
            "allow_new_entries": self.allow_new_entries,
            "guardrail_state": self.guardrail_state,
        }


class GuardrailActionMapper:
    """
    Maps guardrail states to runtime action modifications.
    
    This mapper provides deterministic behavior for each guardrail state,
    allowing the trading engine to automatically adjust risk parameters
    when drawdown thresholds are breached.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        action_config: Optional[GuardrailActionConfig] = None,
    ):
        """
        Initialize the guardrail action mapper.
        
        Args:
            config: Trading config dict with guardrail settings
            action_config: Optional custom configuration for action behavior
        """
        self.config = config or {}
        self.action_config = action_config or GuardrailActionConfig()
        
        # Extract guardrail settings from config if provided
        self._guardrail_enabled = bool(self.config.get("guardrail_enabled", False))
    
    def get_action_for_state(self, state: str) -> GuardrailAction:
        """
        Map guardrail state string to corresponding action enum.
        
        Args:
            state: Guardrail state string ("normal", "warn", "derisk", "halt")
            
        Returns:
            GuardrailAction enum value
        """
        state_lower = str(state).lower().strip()
        
        if state_lower == "halt":
            return GuardrailAction.HALT
        elif state_lower == "derisk":
            return GuardrailAction.DERISK
        elif state_lower == "warn":
            return GuardrailAction.WARN
        else:
            return GuardrailAction.NORMAL
    
    def apply_action(
        self,
        current_params: Dict[str, Any],
        action: Optional[GuardrailAction] = None,
        guardrail_state: Optional[str] = None,
        top_k: int = 1,
        risk_per_trade: float = 0.01,
        signal_threshold: float = 0.65,
        max_positions: int = 10,
    ) -> GuardrailActionResult:
        """
        Apply guardrail action to modify trading parameters.
        
        Args:
            current_params: Current trading parameters (for context)
            action: The guardrail action to apply
            guardrail_state: The current guardrail state string
            top_k: Current top-k setting for new entries per iteration
            risk_per_trade: Current risk per trade
            signal_threshold: Current signal confidence threshold
            max_positions: Current max positions limit
            
        Returns:
            GuardrailActionResult with modified parameters
        """
        # Determine action from state if not provided
        if action is None and guardrail_state is not None:
            action = self.get_action_for_state(guardrail_state)
        elif action is None:
            action = GuardrailAction.NORMAL
        
        state_str = guardrail_state or "normal"
        
        # Apply state-specific modifications
        if action == GuardrailAction.HALT:
            # Halt: No new entries allowed
            return GuardrailActionResult(
                action=GuardrailAction.HALT,
                position_size_scale=0.0,
                top_k=0,
                signal_threshold=1.0,  # Require perfect confidence (effectively block)
                allow_new_entries=False,
                guardrail_state=state_str,
            )
        
        elif action == GuardrailAction.DERISK:
            # De-risk: Reduce position sizes, limit entries, increase confidence
            scaled_top_k = max(1, int(top_k * self.action_config.derisk_top_k_scale))
            adjusted_threshold = min(
                1.0, 
                signal_threshold + self.action_config.derisk_confidence_boost
            )
            return GuardrailActionResult(
                action=GuardrailAction.DERISK,
                position_size_scale=self.action_config.derisk_position_size_scale,
                top_k=scaled_top_k,
                signal_threshold=adjusted_threshold,
                allow_new_entries=True,
                guardrail_state=state_str,
            )
        
        elif action == GuardrailAction.WARN:
            # Warn: Log only (optional behavior), otherwise normal
            if self.action_config.warn_log_only:
                return GuardrailActionResult(
                    action=GuardrailAction.WARN,
                    position_size_scale=1.0,
                    top_k=top_k,
                    signal_threshold=signal_threshold,
                    allow_new_entries=True,
                    guardrail_state=state_str,
                )
            else:
                # If warn_log_only is False, apply mild derisk
                return self.apply_action(
                    current_params,
                    action=GuardrailAction.DERISK,
                    guardrail_state=guardrail_state,
                    top_k=top_k,
                    risk_per_trade=risk_per_trade,
                    signal_threshold=signal_threshold,
                    max_positions=max_positions,
                )
        
        else:
            # Normal: No modifications
            return GuardrailActionResult(
                action=GuardrailAction.NORMAL,
                position_size_scale=1.0,
                top_k=top_k,
                signal_threshold=signal_threshold,
                allow_new_entries=True,
                guardrail_state="normal",
            )
    
    def should_allow_entry(
        self,
        guardrail_state: str,
        current_positions: int,
        max_positions: int = 10,
    ) -> tuple[bool, str]:
        """
        Determine if a new position entry should be allowed.
        
        Args:
            guardrail_state: Current guardrail state
            current_positions: Number of current open positions
            max_positions: Maximum allowed positions
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        action = self.get_action_for_state(guardrail_state)
        
        if action == GuardrailAction.HALT:
            return False, "halted: guardrail halt state active"
        
        if current_positions >= max_positions:
            return False, "max_positions_reached"
        
        if action == GuardrailAction.DERISK:
            # Check if derisk limits have been reached
            derisk_max_positions = int(max_positions * self.action_config.derisk_position_size_scale)
            if current_positions >= derisk_max_positions:
                return False, f"derisk: position limit ({derisk_max_positions}) reached"
        
        return True, "allowed"
    
    def get_effective_risk_per_trade(
        self,
        base_risk: float,
        guardrail_state: str,
    ) -> float:
        """
        Get the effective risk per trade after guardrail modifications.
        
        Args:
            base_risk: Base risk per trade from config
            guardrail_state: Current guardrail state
            
        Returns:
            Effective risk per trade
        """
        action = self.get_action_for_state(guardrail_state)
        
        if action == GuardrailAction.DERISK:
            return base_risk * self.action_config.derisk_position_size_scale
        elif action == GuardrailAction.HALT:
            return 0.0
        
        return base_risk


def create_guardrail_action_mapper_from_config(config: Any) -> GuardrailActionMapper:
    """
    Factory function to create GuardrailActionMapper from TradingConfig.
    
    Args:
        config: TradingConfig object or dict with guardrail settings
        
    Returns:
        Configured GuardrailActionMapper instance
    """
    if hasattr(config, "__dict__"):
        # It's a dataclass/object
        config_dict = {
            "guardrail_enabled": getattr(config, "guardrail_enabled", False),
            "guardrail_warn_drawdown_pct": getattr(config, "guardrail_warn_drawdown_pct", 0.05),
            "guardrail_derisk_drawdown_pct": getattr(config, "guardrail_derisk_drawdown_pct", 0.08),
            "guardrail_halt_drawdown_pct": getattr(config, "guardrail_halt_drawdown_pct", 0.12),
            "guardrail_confirmation_window": getattr(config, "guardrail_confirmation_window", 1),
        }
    else:
        config_dict = config or {}
    
    return GuardrailActionMapper(config=config_dict)
