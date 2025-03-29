"""RF effects modeling for electronic systems."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np

from helios.core.data_structures import Waveform, Signal

class RFEffectType(Enum):
    """Types of RF effects on electronic systems."""
    INTERFERENCE = auto()    # Temporary disruption
    UPSET = auto()           # State change requiring reset
    LATCH_UP = auto()        # Persistent high-current state
    BURNOUT = auto()         # Permanent damage

class RFEffectSeverity(Enum):
    """Severity levels for RF effects."""
    NONE = auto()
    MINOR = auto()
    MODERATE = auto()
    MAJOR = auto()
    CRITICAL = auto()

@dataclass
class RFEffect:
    """Models an RF effect on a component."""
    effect_type: RFEffectType
    severity: RFEffectSeverity
    component_id: str
    waveform: Waveform
    duration: float  # Effect duration in seconds
    recovery_time: float = 0.0  # Time to recover after effect ends

class RFEffectsModel:
    """Predicts RF effects on electronic systems."""
    
    def predict_effects(self, signal: Signal, component_id: str) -> List[RFEffect]:
        """
        Predict RF effects on a component from a signal.
        
        Args:
            signal: RF signal to analyze
            component_id: Target component identifier
            
        Returns:
            List of predicted effects (empty if no effects)
        """
        effects = []
        
        # Calculate power at component (simplified)
        power = signal.power  # Should use distance-based calculation in real implementation
        
        # Determine effects based on power and waveform characteristics
        if power > 10:  # Threshold for interference (dBm)
            effects.append(RFEffect(
                effect_type=RFEffectType.INTERFERENCE,
                severity=RFEffectSeverity.MINOR,
                component_id=component_id,
                waveform=signal.waveform,
                duration=1.0
            ))
            
        if (power > 20 and 
            signal.waveform.modulation_type in ['PULSED', 'HPM']):
            effects.append(RFEffect(
                effect_type=RFEffectType.UPSET,
                severity=RFEffectSeverity.MODERATE,
                component_id=component_id,
                waveform=signal.waveform,
                duration=5.0,
                recovery_time=30.0
            ))
            
        return effects
