"""Continuous clutter models for RF environment simulation."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.environment.clutter import ClutterPatch, DiscreteClutterModel
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TerrainPatch:
    """Represents a terrain patch with continuous properties."""
    center: Position
    size_x: float  # Size in meters (x-direction)
    size_y: float  # Size in meters (y-direction)
    roughness: float  # Surface roughness (meters)
    permittivity: float  # Relative permittivity
    conductivity: float  # Conductivity (S/m)
    height_variation: float = 0.0  # Height variation within patch (meters)
    id: str = field(default_factory=lambda: f"terrain_{np.random.randint(0, 1000000)}")
    
    def get_reflection_coefficient(self, frequency: float, incident_angle: float) -> complex:
        """
        Calculate complex reflection coefficient using Fresnel equations.
        
        Args:
            frequency: Signal frequency in Hz
            incident_angle: Angle of incidence in radians
            
        Returns:
            Complex reflection coefficient
        """
        # Calculate wavelength
        wavelength = 3e8 / frequency
        
        # Simplified Fresnel reflection coefficient for horizontal polarization
        # For vertical polarization, the equation would be different
        sin_theta = np.sin(incident_angle)
        cos_theta = np.cos(incident_angle)
        
        # Complex permittivity
        epsilon_c = complex(self.permittivity, -60 * self.conductivity * wavelength)
        
        # Reflection coefficient (horizontal polarization)
        numerator = cos_theta - np.sqrt(epsilon_c - sin_theta**2)
        denominator = cos_theta + np.sqrt(epsilon_c - sin_theta**2)
        
        reflection_coef = numerator / denominator
        
        # Apply roughness factor (Rayleigh criterion)
        roughness_factor = np.exp(-(4 * np.pi * self.roughness * cos_theta / wavelength)**2 / 2)
        
        return reflection_coef * roughness_factor


class ContinuousClutterModel(DiscreteClutterModel):
    """
    Enhanced clutter model with continuous terrain patches.
    Extends the discrete clutter model with more realistic terrain modeling.
    """
    
    def __init__(self):
        """Initialize the continuous clutter model."""
        super().__init__()
        self.terrain_patches: List[TerrainPatch] = []
        
    def add_terrain_patch(self, patch: TerrainPatch) -> None:
        """Add a terrain patch to the model."""
        self.terrain_patches.append(patch)
        logger.debug(f"Added terrain patch at {patch.center}, size={patch.size_x}x{patch.size_y}m")
        
    def generate_reflections(self, signal: Signal, rx_platform: Platform) -> List[Signal]:
        """
        Generate reflected signals from clutter and terrain.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            
        Returns:
            List of reflected signals
        """
        # Get reflections from discrete clutter points and patches
        reflected_signals = super().generate_reflections(signal, rx_platform)
        
        # Add reflections from terrain patches
        for patch in self.terrain_patches:
            # Calculate incident angle (simplified)
            tx_to_patch_vector = [
                patch.center.x - signal.origin.x,
                patch.center.y - signal.origin.y,
                patch.center.z - signal.origin.z
            ]
            
            # Normalize vector
            distance = np.sqrt(sum(v**2 for v in tx_to_patch_vector))
            if distance < 1e-6:
                continue
                
            tx_to_patch_unit = [v/distance for v in tx_to_patch_vector]
            
            # Simplified normal vector (assuming flat patch)
            normal_vector = [0, 0, 1]  # Pointing up
            
            # Calculate incident angle
            cos_incident = abs(sum(a*b for a, b in zip(tx_to_patch_unit, normal_vector)))
            incident_angle = np.arccos(cos_incident)
            
            # Calculate reflection coefficient
            refl_coef = patch.get_reflection_coefficient(
                signal.waveform.center_frequency, 
                incident_angle
            )
            
            # Calculate distances
            tx_to_patch = distance
            patch_to_rx = patch.center.distance_to(rx_platform.position)
            
            # Skip if either distance is too small
            if tx_to_patch < 1e-6 or patch_to_rx < 1e-6:
                continue
                
            # Calculate total path length and delay
            total_path = tx_to_patch + patch_to_rx
            delay = total_path / 3e8  # Speed of light
            
            # Calculate power at patch (free space path loss)
            freq = signal.waveform.center_frequency
            wavelength = 3e8 / freq
            
            # Power at patch
            power_at_patch = signal.power - 20 * np.log10(4 * np.pi * tx_to_patch / wavelength)
            
            # Reflected power
            reflected_power = power_at_patch + 20 * np.log10(abs(refl_coef))
            
            # Power at receiver
            received_power = reflected_power - 20 * np.log10(4 * np.pi * patch_to_rx / wavelength)
            
            # Create reflected signal - without the phase_shift parameter
            reflected = Signal(
                source_id=signal.source_id,
                waveform=signal.waveform,
                origin=patch.center,  # Reflection comes from the patch center (simplified)
                source_velocity=signal.source_velocity,
                emission_time=signal.emission_time + delay,
                power=received_power,
                propagation_delay=delay
                # Removed phase_shift parameter
            )
            
            # If you need to track phase information, you could add it to metadata if Signal supports it
            # For example:
            # if hasattr(reflected, 'metadata'):
            #     reflected.metadata['phase_shift'] = np.angle(refl_coef)
            
            reflected_signals.append(reflected)
            
        return reflected_signals