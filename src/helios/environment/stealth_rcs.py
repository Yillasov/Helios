"""Specialized RCS models for stealth targets."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from helios.core.data_structures import Orientation
from helios.environment.rcs import RCSModel, RCSModelType
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class StealthAircraftRCSModel(RCSModel):
    """
    RCS model for stealth aircraft with angular surfaces.
    
    This model simulates the radar-deflecting properties of stealth aircraft
    by modeling their characteristic angular surfaces and radar-absorbing materials.
    """
    
    def __init__(self, 
                 length: float, 
                 wingspan: float, 
                 ram_effectiveness: float = 0.8):
        """
        Initialize stealth aircraft RCS model.
        
        Args:
            length: Aircraft length in meters
            wingspan: Aircraft wingspan in meters
            ram_effectiveness: Radar Absorbing Material effectiveness (0-1)
        """
        super().__init__(RCSModelType.COMPLEX)
        self.length = length
        self.wingspan = wingspan
        self.ram_effectiveness = min(1.0, max(0.0, ram_effectiveness))
        
        # Baseline RCS for a conventional aircraft of similar size
        self.baseline_rcs = 0.01 * (length * wingspan)  # Simplified approximation
        
        # Angular sectors with different RCS characteristics
        self.sector_rcs_factors = {
            'front': 0.01,    # Very low RCS from front aspect
            'side': 0.1,      # Higher from side aspect
            'rear': 0.3,      # Highest from rear aspect (engine exhaust)
            'top': 0.05,      # Low from top
            'bottom': 0.2     # Higher from bottom (less stealth treatment)
        }
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate stealth aircraft RCS based on aspect angle and frequency.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the aircraft
            incident_direction: Direction of incident wave (unit vector)
            
        Returns:
            RCS value in m²
        """
        # Normalize incident direction
        incident = np.array(incident_direction)
        if np.linalg.norm(incident) > 0:
            incident = incident / np.linalg.norm(incident)
        
        # Aircraft body reference frame (simplified)
        # Forward vector in aircraft body frame
        forward = np.array([
            np.cos(target_orientation.pitch) * np.cos(target_orientation.yaw),
            np.cos(target_orientation.pitch) * np.sin(target_orientation.yaw),
            -np.sin(target_orientation.pitch)
        ])
        
        # Right vector in aircraft body frame
        right = np.array([
            np.sin(target_orientation.roll) * np.sin(target_orientation.pitch) * np.cos(target_orientation.yaw) - 
            np.cos(target_orientation.roll) * np.sin(target_orientation.yaw),
            np.sin(target_orientation.roll) * np.sin(target_orientation.pitch) * np.sin(target_orientation.yaw) + 
            np.cos(target_orientation.roll) * np.cos(target_orientation.yaw),
            np.sin(target_orientation.roll) * np.cos(target_orientation.pitch)
        ])
        
        # Up vector in aircraft body frame
        up = np.array([
            np.cos(target_orientation.roll) * np.sin(target_orientation.pitch) * np.cos(target_orientation.yaw) + 
            np.sin(target_orientation.roll) * np.sin(target_orientation.yaw),
            np.cos(target_orientation.roll) * np.sin(target_orientation.pitch) * np.sin(target_orientation.yaw) - 
            np.sin(target_orientation.roll) * np.cos(target_orientation.yaw),
            np.cos(target_orientation.roll) * np.cos(target_orientation.pitch)
        ])
        
        # Calculate dot products to determine aspect angles
        front_aspect = -np.dot(incident, forward)  # Negative because incident is toward aircraft
        side_aspect = np.abs(np.dot(incident, right))
        top_aspect = np.dot(incident, up)
        
        # Determine dominant aspect
        if front_aspect > 0.7:  # Mostly front aspect
            dominant_sector = 'front'
        elif front_aspect < -0.7:  # Mostly rear aspect
            dominant_sector = 'rear'
        elif side_aspect > 0.7:  # Mostly side aspect
            dominant_sector = 'side'
        elif top_aspect > 0.7:  # Mostly top aspect
            dominant_sector = 'top'
        elif top_aspect < -0.7:  # Mostly bottom aspect
            dominant_sector = 'bottom'
        else:  # Mixed aspect
            # Weighted average of different aspects
            weights = {
                'front': max(0, front_aspect),
                'rear': max(0, -front_aspect),
                'side': side_aspect,
                'top': max(0, top_aspect),
                'bottom': max(0, -top_aspect)
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
                
            # Calculate weighted RCS factor
            rcs_factor = sum(weights[sector] * self.sector_rcs_factors[sector] 
                            for sector in weights)
            
            # Apply frequency effects and RAM
            return self._apply_frequency_and_ram_effects(
                self.baseline_rcs * rcs_factor, 
                frequency
            )
        
        # Apply RCS factor for dominant sector
        rcs = self.baseline_rcs * self.sector_rcs_factors[dominant_sector]
        
        # Apply frequency effects and RAM
        return self._apply_frequency_and_ram_effects(rcs, frequency)
    
    def _apply_frequency_and_ram_effects(self, base_rcs: float, frequency: float) -> float:
        """Apply frequency-dependent effects and radar absorbing materials."""
        # RAM effectiveness varies with frequency
        # Most effective in X-band (8-12 GHz)
        x_band_center = 10e9  # 10 GHz
        
        # Calculate frequency-dependent RAM factor
        freq_factor = 1.0
        if 8e9 <= frequency <= 12e9:  # X-band
            # Maximum effectiveness in X-band
            freq_factor = 1.0
        else:
            # Reduced effectiveness outside X-band
            distance_from_xband = min(abs(frequency - 8e9), abs(frequency - 12e9))
            freq_factor = max(0.5, 1.0 - (distance_from_xband / 4e9))
        
        # Apply RAM reduction
        ram_factor = 1.0 - (self.ram_effectiveness * freq_factor)
        
        return base_rcs * ram_factor