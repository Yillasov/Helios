"""Hardening profiles for RF-based armament systems."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from helios.utils.logger import get_logger

logger = get_logger(__name__)

class HardeningLevel(Enum):
    """Levels of electromagnetic hardening."""
    NONE = 0
    MINIMAL = 1
    MODERATE = 2
    SUBSTANTIAL = 3
    MILITARY_GRADE = 4
    TEMPEST = 5


@dataclass
class FrequencyBand:
    """Frequency band with associated protection level."""
    min_freq: float  # Hz
    max_freq: float  # Hz
    attenuation: float  # dB


class HardeningProfile:
    """Profile for electromagnetic hardening of systems."""
    
    def __init__(self, system_id: str, level: HardeningLevel = HardeningLevel.MODERATE):
        """Initialize the hardening profile.
        
        Args:
            system_id: ID of the system being hardened
            level: Overall hardening level
        """
        self.system_id = system_id
        self.level = level
        self.protected_bands: List[FrequencyBand] = []
        self.shielding_effectiveness = 20.0 * (level.value / 2)  # dB
        self.filter_effectiveness = 10.0 * (level.value / 2)  # dB
        self.grounding_quality = min(0.95, 0.5 + 0.1 * level.value)  # 0-1 scale
        
    def add_protected_band(self, min_freq: float, max_freq: float, attenuation: float):
        """Add a protected frequency band.
        
        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            attenuation: Attenuation in dB
        """
        band = FrequencyBand(min_freq=min_freq, max_freq=max_freq, attenuation=attenuation)
        self.protected_bands.append(band)
        logger.debug(f"Added protected band {min_freq/1e6}-{max_freq/1e6} MHz to {self.system_id}")
        
    def calculate_protection(self, frequency: float) -> float:
        """Calculate protection level at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Protection level in dB
        """
        # Base protection from shielding
        protection = self.shielding_effectiveness
        
        # Add band-specific protection
        for band in self.protected_bands:
            if band.min_freq <= frequency <= band.max_freq:
                protection += band.attenuation
                break
                
        return protection
        
    def upgrade_to_level(self, level: HardeningLevel):
        """Upgrade hardening to a specific level.
        
        Args:
            level: Target hardening level
        """
        if level.value <= self.level.value:
            logger.warning(f"System {self.system_id} already at level {self.level.name} or higher")
            return
            
        old_level = self.level
        self.level = level
        
        # Update protection values
        self.shielding_effectiveness = 20.0 * (level.value / 2)
        self.filter_effectiveness = 10.0 * (level.value / 2)
        self.grounding_quality = min(0.95, 0.5 + 0.1 * level.value)
        
        logger.info(f"Upgraded {self.system_id} hardening from {old_level.name} to {level.name}")


class EMPProtection:
    """Specialized protection against electromagnetic pulse (EMP) effects."""
    
    def __init__(self, system_id: str):
        """Initialize EMP protection.
        
        Args:
            system_id: ID of the system being protected
        """
        self.system_id = system_id
        self.has_faraday_cage = False
        self.has_surge_protection = False
        self.has_optical_isolation = False
        self.has_emp_filters = False
        self.protection_level = 0.0  # 0-1 scale
        
    def add_faraday_cage(self):
        """Add Faraday cage protection."""
        self.has_faraday_cage = True
        self._update_protection_level()
        logger.info(f"Added Faraday cage to {self.system_id}")
        
    def add_surge_protection(self):
        """Add surge protection devices."""
        self.has_surge_protection = True
        self._update_protection_level()
        logger.info(f"Added surge protection to {self.system_id}")
        
    def add_optical_isolation(self):
        """Add optical isolation for critical interfaces."""
        self.has_optical_isolation = True
        self._update_protection_level()
        logger.info(f"Added optical isolation to {self.system_id}")
        
    def add_emp_filters(self):
        """Add specialized EMP filters."""
        self.has_emp_filters = True
        self._update_protection_level()
        logger.info(f"Added EMP filters to {self.system_id}")
        
    def _update_protection_level(self):
        """Update the overall protection level based on components."""
        # Simple additive model with diminishing returns
        level = 0.0
        
        if self.has_faraday_cage:
            level += 0.4
        
        if self.has_surge_protection:
            level += 0.2
        
        if self.has_optical_isolation:
            level += 0.2
        
        if self.has_emp_filters:
            level += 0.3
            
        # Apply diminishing returns
        self.protection_level = 1.0 - (1.0 - level) ** 2
        
    def get_survival_probability(self, emp_intensity: float) -> float:
        """Calculate probability of surviving an EMP.
        
        Args:
            emp_intensity: EMP intensity (normalized 0-1 scale)
            
        Returns:
            Survival probability (0-1)
        """
        # Simple model: protection reduces effective intensity
        effective_intensity = emp_intensity * (1.0 - self.protection_level)
        
        # Survival probability decreases with effective intensity
        survival_prob = 1.0 - effective_intensity ** 2
        
        return max(0.0, min(1.0, survival_prob))