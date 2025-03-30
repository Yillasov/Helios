"""Defines Electronic Countermeasures (ECM) techniques."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

class ECMTechniqueType(Enum):
    """Types of common ECM techniques."""
    NOISE_JAMMING = auto()      # Barrage, spot, swept noise
    DECEPTION_JAMMING = auto()  # False targets, range/angle deception
    CHAFF = auto()              # Physical RF reflector clouds (simplified as ECM)
    INFRARED_CM = auto()        # Flares (outside RF scope but listed for completeness)

@dataclass
class ECMParameters:
    """Parameters describing an active ECM technique."""
    technique_type: ECMTechniqueType
    power_erp: float = 0.0  # Effective Radiated Power in Watts
    bandwidth: Optional[float] = None # Jamming bandwidth in Hz (for noise)
    duty_cycle: float = 1.0 # ECM signal duty cycle
    parameters: Dict[str, Any] = field(default_factory=dict) # Technique-specific params

    def __str__(self) -> str:
        return f"{self.technique_type.name}(ERP={self.power_erp:.1f}W)"