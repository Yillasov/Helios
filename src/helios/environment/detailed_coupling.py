"""Detailed power coupling models for HPM effects simulation."""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field

from helios.core.data_structures import Position, Signal, HPMWaveform
from helios.environment.em_coupling import EMField, CouplingResult
from helios.environment.em_structures import Enclosure, Aperture
from helios.environment.hpm_coupling import CouplingPath, HPMEffect
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FrequencyResponse:
    """Frequency-dependent coupling response."""
    frequency: float  # Center frequency in Hz
    coupling_coefficient: float  # dB
    bandwidth: float  # Response bandwidth in Hz
    q_factor: float = 10.0  # Quality factor of resonance

    def get_response(self, freq: float) -> float:
        """Calculate coupling response at a specific frequency."""
        # Simple resonant response model
        normalized_freq_diff = (freq - self.frequency) / (self.bandwidth / 2)
        response = self.coupling_coefficient / (1 + (normalized_freq_diff ** 2) * self.q_factor)
        return response

@dataclass
class DetailedCouplingPath(CouplingPath):
    """Enhanced coupling path with detailed frequency and polarization response."""
    # Inherit basic properties from CouplingPath
    frequency_responses: List[FrequencyResponse] = field(default_factory=list)
    polarization_response: Dict[str, float] = field(default_factory=dict)
    material_attenuation: float = 0.0  # Additional attenuation in dB
    
    def calculate_coupling(self, signal: Signal) -> float:
        """
        Calculate detailed coupling coefficient for a specific signal.
        
        Args:
            signal: The incident signal
            
        Returns:
            Effective coupling coefficient in dB
        """
        if not isinstance(signal.waveform, HPMWaveform):
            logger.warning("Signal does not contain an HPM waveform")
            return -100.0  # Very low coupling for non-HPM signals
            
        # Start with base coupling coefficient
        effective_coupling = self.coupling_coefficient
        
        # Apply frequency-dependent response
        if self.frequency_responses:
            freq_coupling = self._calculate_frequency_response(signal.waveform.center_frequency)
            effective_coupling = freq_coupling
        
        # Apply polarization effects
        if signal.polarization and self.polarization_response:
            pol_factor = self.polarization_response.get(signal.polarization, 1.0)
            effective_coupling += 10 * np.log10(pol_factor)  # Convert to dB
            
        # Apply material attenuation
        effective_coupling -= self.material_attenuation
        
        return effective_coupling
        
    def _calculate_frequency_response(self, frequency: float) -> float:
        """Calculate combined frequency response at a specific frequency."""
        if not self.frequency_responses:
            return self.coupling_coefficient
            
        # Find the maximum response across all frequency responses
        responses = [resp.get_response(frequency) for resp in self.frequency_responses]
        return max(responses)


class DetailedHPMCouplingModel:
    """
    Enhanced HPM coupling model with detailed power coupling calculations.
    Accounts for frequency-dependent coupling, polarization effects, and material properties.
    """
    
    def __init__(self):
        """Initialize the detailed HPM coupling model."""
        self.coupling_paths: Dict[str, List[DetailedCouplingPath]] = {}
        self.active_effects: List[Tuple[float, HPMEffect]] = []
        
    def add_coupling_path(self, platform_id: str, path: DetailedCouplingPath):
        """Add a detailed coupling path to a platform."""
        if platform_id not in self.coupling_paths:
            self.coupling_paths[platform_id] = []
        
        self.coupling_paths[platform_id].append(path)
        logger.debug(f"Added detailed coupling path '{path.name}' to platform {platform_id}")
    
    def calculate_coupling(self, signal: Signal, platform_id: str) -> List[HPMEffect]:
        """
        Calculate HPM coupling effects with detailed models.
        
        Args:
            signal: The incident signal
            platform_id: ID of the platform potentially affected
            
        Returns:
            List of HPM effects generated
        """
        if platform_id not in self.coupling_paths:
            return []
        
        effects = []
        
        # Check if signal has HPM waveform
        if not hasattr(signal.waveform, 'peak_power'):
            logger.debug(f"Signal {signal.id} is not an HPM signal")
            return []
            
        for path in self.coupling_paths[platform_id]:
            # Check if signal frequency is in coupling path's range
            signal_freq = signal.waveform.center_frequency
            min_freq, max_freq = path.frequency_range
            
            if not (min_freq <= signal_freq <= max_freq):
                continue
            
            # Calculate detailed coupling
            effective_coupling = path.calculate_coupling(signal)
            
            # Calculate coupled power
            coupled_power = signal.power + effective_coupling
            
            # Check if power exceeds threshold
            if coupled_power > path.threshold_power:
                # Calculate effect severity based on power above threshold
                power_ratio = 10**((coupled_power - path.threshold_power) / 10)
                severity = min(1.0, power_ratio / 100)  # Normalize to 0-1
                
                # Determine effect type and duration based on severity
                if severity > 0.8:
                    effect_type = "damage"
                    duration = float('inf')  # Permanent damage
                elif severity > 0.5:
                    effect_type = "upset"
                    duration = 30.0  # 30 seconds of system upset
                else:
                    effect_type = "interference"
                    duration = 5.0  # 5 seconds of interference
                
                # Create effect
                effect = HPMEffect(
                    system_id=path.system_id,
                    effect_type=effect_type,
                    severity=severity,
                    duration=duration,
                    description=f"HPM effect via {path.name} coupling path"
                )
                
                effects.append(effect)
        
        return effects


class ResonantCavityCoupling:
    """
    Models coupling into resonant cavities like equipment enclosures.
    Accounts for cavity resonances, Q-factor, and field enhancement.
    """
    
    def __init__(self, cavity_dimensions: Tuple[float, float, float]):
        """
        Initialize the resonant cavity coupling model.
        
        Args:
            cavity_dimensions: (width, height, depth) in meters
        """
        self.dimensions = cavity_dimensions
        self.q_factor = 50.0  # Quality factor of the cavity
        self.resonant_modes = self._calculate_resonant_modes()
        
    def _calculate_resonant_modes(self, max_mode: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Calculate resonant modes of the cavity.
        
        Args:
            max_mode: Maximum mode number to calculate
            
        Returns:
            List of ((m, n, p), frequency) tuples
        """
        c = 3e8  # Speed of light in m/s
        w, h, d = self.dimensions
        
        modes = []
        for m in range(max_mode + 1):
            for n in range(max_mode + 1):
                for p in range(max_mode + 1):
                    # Skip (0,0,0) mode
                    if m == 0 and n == 0 and p == 0:
                        continue
                        
                    # Calculate resonant frequency
                    freq = (c/2) * np.sqrt((m/w)**2 + (n/h)**2 + (p/d)**2)
                    modes.append(((m, n, p), freq))
        
        # Sort by frequency
        modes.sort(key=lambda x: x[1])
        return modes
    
    def calculate_field_enhancement(self, frequency: float) -> float:
        """
        Calculate field enhancement factor at a specific frequency.
        
        Args:
            frequency: Signal frequency in Hz
            
        Returns:
            Field enhancement factor (linear)
        """
        # Find closest resonant mode
        closest_mode = min(self.resonant_modes, key=lambda x: abs(x[1] - frequency))
        mode, mode_freq = closest_mode
        
        # Calculate enhancement based on Q-factor and frequency difference
        normalized_diff = abs(frequency - mode_freq) / (mode_freq / self.q_factor)
        enhancement = self.q_factor / (1 + normalized_diff**2)
        
        return enhancement