"""EMP hardening analysis tools for military systems."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from helios.utils.logger import get_logger
from helios.armament.hardening import HardeningLevel, HardeningProfile, EMPProtection
from helios.design.military_component_library import MilitaryComponentLibrary

logger = get_logger(__name__)

class EMPThreatLevel(Enum):
    """Standard EMP threat levels."""
    COMMERCIAL = auto()       # Typical industrial/commercial EMP
    MILITARY = auto()         # Standard military-grade EMP
    NUCLEAR = auto()          # High-altitude nuclear EMP (HEMP)
    SUPER_EMP = auto()        # Enhanced EMP weapons
    EXTREME = auto()          # Worst-case scenario

@dataclass
class EMPThreatProfile:
    """Profile defining EMP threat characteristics."""
    level: EMPThreatLevel
    peak_field: float                # V/m
    rise_time: float                 # seconds
    frequency_components: List[Tuple[float, float]]  # (frequency, weight) pairs
    duration: float                  # seconds

class EMPHardeningAnalyzer:
    """Tool for analyzing EMP hardening effectiveness."""
    
    def __init__(self, system_id: str):
        """Initialize analyzer with system ID."""
        self.system_id = system_id
        self.hardening_profile = HardeningProfile(system_id)
        self.emp_protection = EMPProtection(system_id)
        self.component_lib = MilitaryComponentLibrary()
        
    def analyze_system(self, 
                      threat: EMPThreatProfile,
                      components: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze system vulnerability to specified EMP threat.
        
        Args:
            threat: EMP threat profile
            components: Optional list of specific component IDs to analyze
            
        Returns:
            Dictionary of component IDs to survival probabilities (0-1)
        """
        results = {}
        
        # Calculate overall system protection
        system_protection = self._calculate_system_protection(threat)
        
        # If no specific components provided, analyze all military components
        if not components:
            components = list(self.component_lib.component_templates.keys())
            
        for component_id in components:
            # Get component-specific vulnerability factors
            component_vuln = self._get_component_vulnerability(component_id)
            
            # Calculate effective threat intensity after protection
            effective_threat = threat.peak_field * (1 - system_protection)
            
            # Calculate survival probability
            survival_prob = self._calculate_survival_probability(
                effective_threat, 
                component_vuln
            )
            
            results[component_id] = survival_prob
            
        return results
    
    def _calculate_system_protection(self, threat: EMPThreatProfile) -> float:
        """Calculate overall system protection against EMP threat."""
        # Base protection from hardening profile
        protection = self.hardening_profile.level.value / HardeningLevel.TEMPEST.value
        
        # Add EMP-specific protection
        protection += self.emp_protection.protection_level * 0.5
        
        # Frequency-weighted protection
        freq_protection = 0.0
        for freq, weight in threat.frequency_components:
            freq_protection += weight * self.hardening_profile.calculate_protection(freq)
        
        return min(0.99, protection + (freq_protection / 100))
    
    def _get_component_vulnerability(self, component_id: str) -> float:
        """Get vulnerability factor for a component (0-1 scale)."""
        template = self.component_lib.component_templates.get(component_id)
        if not template:
            logger.warning(f"Component template {component_id} not found")
            return 1.0  # Default to fully vulnerable
            
        params = template['parameters']
        
        # Calculate vulnerability based on hardening parameters
        vuln = 1.0
        if 'radiation_hardness_tid' in params:
            vuln -= params['radiation_hardness_tid'] / 1000  # Normalize to 0-1
        if 'vibration_spec' in params:
            vuln -= 0.2 if "MIL-STD" in params['vibration_spec'] else 0
        
        return max(0.0, min(1.0, vuln))
    
    def _calculate_survival_probability(self, 
                                      effective_threat: float, 
                                      component_vuln: float) -> float:
        """Calculate component survival probability."""
        # Simple model: threat * vulnerability determines survival
        threat_factor = effective_threat / 1e6  # Normalize
        survival = 1.0 - (threat_factor * component_vuln)
        return max(0.0, min(1.0, survival))

# Standard EMP threat profiles
STANDARD_EMP_THREATS = {
    "commercial": EMPThreatProfile(
        level=EMPThreatLevel.COMMERCIAL,
        peak_field=1e3,  # 1 kV/m
        rise_time=1e-6,  # 1 μs
        frequency_components=[(1e6, 0.7), (10e6, 0.3)],  # Mostly 1-10 MHz
        duration=1e-3    # 1 ms
    ),
    "military": EMPThreatProfile(
        level=EMPThreatLevel.MILITARY,
        peak_field=5e4,  # 50 kV/m
        rise_time=1e-8,  # 10 ns
        frequency_components=[(10e6, 0.5), (100e6, 0.3), (1e9, 0.2)],
        duration=1e-3    # 1 ms
    ),
    "nuclear": EMPThreatProfile(
        level=EMPThreatLevel.NUCLEAR,
        peak_field=5e4,  # 50 kV/m
        rise_time=5e-9,  # 5 ns
        frequency_components=[(1e6, 0.2), (10e6, 0.3), (100e6, 0.3), (1e9, 0.2)],
        duration=1e-6    # 1 μs
    )
}