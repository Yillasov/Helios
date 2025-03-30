"""Defines Electronic Counter-Countermeasures (ECCM) techniques."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

class ECCMTechniqueType(Enum):
    """Types of common ECCM techniques."""
    FREQUENCY_HOPPING = auto()
    PULSE_COMPRESSION = auto()
    SIDELOBE_CANCELLATION = auto()
    POLARIZATION_FILTERING = auto()
    HOME_ON_JAM = auto() # Guidance technique, but relevant ECCM

@dataclass
class ECCMParameters:
    """Parameters describing active ECCM capabilities or techniques."""
    supported_techniques: List[ECCMTechniqueType] = field(default_factory=list)
    active_technique: Optional[ECCMTechniqueType] = None
    parameters: Dict[str, Any] = field(default_factory=dict) # Technique-specific params

    def is_active(self, technique: ECCMTechniqueType) -> bool:
        """Check if a specific ECCM technique is currently active."""
        return self.active_technique == technique

    def supports(self, technique: ECCMTechniqueType) -> bool:
        """Check if a specific ECCM technique is supported."""
        return technique in self.supported_techniques

    def __str__(self) -> str:
        active_str = self.active_technique.name if self.active_technique else "None"
        supported_str = ', '.join(t.name for t in self.supported_techniques)
        return f"Active: {active_str}, Supported: [{supported_str}]"