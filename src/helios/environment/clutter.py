"""Clutter models for RF environment simulation."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ClutterPoint:
    """Represents a discrete reflection point in the environment."""
    position: Position
    rcs: float  # Radar Cross Section in m²
    reflection_coefficient: float = 0.8  # Default reflection coefficient
    phase_shift: float = np.pi  # Default phase shift (radians)
    id: str = field(default_factory=lambda: f"clutter_{np.random.randint(0, 1000000)}")

@dataclass
class ClutterPatch:
    """Represents a larger area of clutter with similar properties."""
    center: Position
    size_x: float  # Size in meters (x-direction)
    size_y: float  # Size in meters (y-direction)
    rcs_density: float  # RCS per square meter
    reflection_coefficient: float = 0.7
    phase_shift: float = np.pi
    id: str = field(default_factory=lambda: f"patch_{np.random.randint(0, 1000000)}")
    
    def get_rcs(self) -> float:
        """Get total RCS of the patch."""
        return self.rcs_density * self.size_x * self.size_y
    
    def is_point_inside(self, position: Position) -> bool:
        """Check if a position is inside this patch."""
        half_x = self.size_x / 2
        half_y = self.size_y / 2
        
        return (abs(position.x - self.center.x) <= half_x and 
                abs(position.y - self.center.y) <= half_y)

class DiscreteClutterModel:
    """
    Models discrete clutter points and patches in the environment.
    Used to simulate reflections from objects and terrain.
    """
    
    def __init__(self):
        """Initialize the clutter model."""
        self.clutter_points: List[ClutterPoint] = []
        self.clutter_patches: List[ClutterPatch] = []
    
    def add_clutter_point(self, point: ClutterPoint) -> None:
        """Add a clutter point to the model."""
        self.clutter_points.append(point)
        logger.debug(f"Added clutter point at {point.position}, RCS={point.rcs} m²")
    
    def add_clutter_patch(self, patch: ClutterPatch) -> None:
        """Add a clutter patch to the model."""
        self.clutter_patches.append(patch)
        logger.debug(f"Added clutter patch at {patch.center}, size={patch.size_x}x{patch.size_y}m, RCS={patch.get_rcs()} m²")
    
    def generate_reflections(self, signal: Signal, rx_platform: Platform) -> List[Signal]:
        """
        Generate reflected signals from clutter for a given transmitted signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            
        Returns:
            List of reflected signals
        """
        reflected_signals = []
        
        # Process reflections from discrete points
        for point in self.clutter_points:
            # Calculate distances
            tx_to_point = signal.origin.distance_to(point.position)
            point_to_rx = point.position.distance_to(rx_platform.position)
            
            # Skip if either distance is too small
            if tx_to_point < 1e-6 or point_to_rx < 1e-6:
                continue
            
            # Calculate total path length
            total_path = tx_to_point + point_to_rx
            
            # Calculate delay
            delay = total_path / 3e8  # Speed of light
            
            # Calculate power at clutter point (free space path loss)
            freq = signal.waveform.center_frequency
            wavelength = 3e8 / freq
            
            # Power at clutter point (using radar equation)
            power_at_point = signal.power - 20 * np.log10(4 * np.pi * tx_to_point / wavelength)
            
            # Reflected power (using RCS)
            reflected_power = power_at_point + 10 * np.log10(point.rcs / (4 * np.pi))
            
            # Power at receiver (free space path loss from point to receiver)
            received_power = reflected_power - 20 * np.log10(4 * np.pi * point_to_rx / wavelength)
            
            # Create reflected signal
            reflected = Signal(
                source_id=signal.source_id,
                waveform=signal.waveform,
                origin=point.position,  # Reflection comes from the clutter point
                source_velocity=signal.source_velocity,  # Keep original velocity
                emission_time=signal.emission_time + delay,  # Delayed by propagation
                power=received_power,  # Attenuated power
                propagation_delay=delay
            )
            
            reflected_signals.append(reflected)
        
        # Process reflections from patches (simplified)
        for patch in self.clutter_patches:
            # Use patch center for simplified calculation
            tx_to_patch = signal.origin.distance_to(patch.center)
            patch_to_rx = patch.center.distance_to(rx_platform.position)
            
            # Skip if either distance is too small
            if tx_to_patch < 1e-6 or patch_to_rx < 1e-6:
                continue
            
            # Calculate total path length
            total_path = tx_to_patch + patch_to_rx
            
            # Calculate delay
            delay = total_path / 3e8  # Speed of light
            
            # Calculate power using patch RCS
            freq = signal.waveform.center_frequency
            wavelength = 3e8 / freq
            
            # Power at patch (using radar equation)
            power_at_patch = signal.power - 20 * np.log10(4 * np.pi * tx_to_patch / wavelength)
            
            # Reflected power (using patch RCS)
            patch_rcs = patch.get_rcs()
            reflected_power = power_at_patch + 10 * np.log10(patch_rcs / (4 * np.pi))
            
            # Power at receiver
            received_power = reflected_power - 20 * np.log10(4 * np.pi * patch_to_rx / wavelength)
            
            # Create reflected signal
            reflected = Signal(
                source_id=signal.source_id,
                waveform=signal.waveform,
                origin=patch.center,  # Reflection comes from the patch center
                source_velocity=signal.source_velocity,
                emission_time=signal.emission_time + delay,
                power=received_power,
                propagation_delay=delay
            )
            
            reflected_signals.append(reflected)
        
        return reflected_signals