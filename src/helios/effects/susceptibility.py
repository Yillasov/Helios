"""HPM effects modeling for electronic components susceptibility."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Assuming ModulationType is defined elsewhere, e.g., helios.core.data_structures
# from helios.core.data_structures import ModulationType 

from helios.utils.logger import get_logger

logger = get_logger(__name__)

class EffectType(Enum):
    """Types of effects that can occur on electronic components."""
    NONE = auto()
    INTERFERENCE = auto()  # Temporary interference
    UPSET = auto()         # Temporary malfunction requiring reset
    LATCH_UP = auto()      # Semi-permanent state requiring power cycle
    BURNOUT = auto()       # Permanent damage
    FUNCTIONAL_DAMAGE = auto()  # Partial damage affecting functionality
    PHYSICAL_DAMAGE = auto()    # Complete physical damage


@dataclass
class ComponentSusceptibility:
    """Defines susceptibility parameters for an electronic component."""
    component_id: str
    component_type: str  # e.g., "processor", "memory", "power_supply"
    
    # Frequency-dependent thresholds (frequency in Hz, power in dBm)
    frequency_thresholds: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    # Default thresholds if frequency-specific ones aren't available
    upset_threshold: float = 20.0      # dBm
    damage_threshold: float = 40.0     # dBm
    
    # Recovery characteristics
    recovery_time: Dict[EffectType, float] = field(default_factory=dict)
    
    # Modulation-dependent sensitivity factors (ModulationType name -> factor)
    # Factor < 1 means lower threshold (more susceptible) for that modulation
    modulation_factors: Dict[str, float] = field(default_factory=dict)
    
    # Additional parameters
    polarization_sensitivity: float = 1.0  # 0-1 scale
    pulse_width_sensitivity: Dict[str, float] = field(default_factory=dict)  # Effect type to sensitivity
    
    def __post_init__(self):
        """Initialize default recovery times if not provided."""
        if not self.recovery_time:
            self.recovery_time = {
                EffectType.INTERFERENCE: 0.1,    # 100ms
                EffectType.UPSET: 1.0,           # 1s
                EffectType.LATCH_UP: 10.0,       # 10s (requires power cycle)
                EffectType.BURNOUT: float('inf'),  # Permanent
                EffectType.FUNCTIONAL_DAMAGE: float('inf'),  # Permanent
                EffectType.PHYSICAL_DAMAGE: float('inf')     # Permanent
            }
        
        if not self.pulse_width_sensitivity:
            # Default pulse width sensitivity (shorter pulses may be less effective)
            self.pulse_width_sensitivity = {
                "upset": 0.5,     # Sensitivity factor for upset effects
                "damage": 0.8     # Sensitivity factor for damage effects
            }

