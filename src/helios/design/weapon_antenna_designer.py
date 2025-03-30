"""Specialized antenna pattern designer for directional weapon systems."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

from helios.design.rf_components import Antenna
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class AntennaPatternType(Enum):
    """Types of specialized antenna patterns for weapon systems."""
    PENCIL_BEAM = auto()       # Narrow, high-gain beam
    SECTOR_SCAN = auto()       # Wide azimuth, narrow elevation
    TRACK_WHILE_SCAN = auto()  # Multiple beams for tracking and scanning
    MONOPULSE = auto()         # Precision tracking with sum/difference patterns
    PHASED_ARRAY = auto()      # Electronically steerable array


class WeaponAntennaDesigner:
    """Designer for specialized weapon system antenna patterns."""
    
    def __init__(self):
        """Initialize the weapon antenna designer."""
        self.patterns: Dict[str, Dict[str, Any]] = {}
    
    def create_pencil_beam(self, 
                          name: str,
                          center_frequency: float,
                          beamwidth: float,
                          gain: float) -> Antenna:
        """Create a pencil beam antenna for precision targeting.
        
        Args:
            name: Antenna name
            center_frequency: Center frequency in Hz
            beamwidth: 3dB beamwidth in degrees
            gain: Antenna gain in dBi
            
        Returns:
            Configured antenna component
        """
        antenna = Antenna(name=name)
        antenna.center_frequency = center_frequency
        antenna.gain = gain
        antenna.radiation_pattern = "directional"
        antenna.polarization = "linear"
        
        # Store pattern parameters for later use
        self.patterns[antenna.id] = {
            "type": AntennaPatternType.PENCIL_BEAM,
            "beamwidth": beamwidth,
            "gain": gain,
            "pattern_function": self._generate_pencil_beam_pattern(beamwidth)
        }
        
        logger.info(f"Created pencil beam antenna: {name}, gain: {gain}dBi, beamwidth: {beamwidth}°")
        return antenna
    
    def create_sector_scan(self,
                          name: str,
                          center_frequency: float,
                          azimuth_width: float,
                          elevation_width: float,
                          gain: float) -> Antenna:
        """Create a sector scan antenna for wide area coverage.
        
        Args:
            name: Antenna name
            center_frequency: Center frequency in Hz
            azimuth_width: Azimuth beamwidth in degrees
            elevation_width: Elevation beamwidth in degrees
            gain: Antenna gain in dBi
            
        Returns:
            Configured antenna component
        """
        antenna = Antenna(name=name)
        antenna.center_frequency = center_frequency
        antenna.gain = gain
        antenna.radiation_pattern = "directional"
        
        # Store pattern parameters
        self.patterns[antenna.id] = {
            "type": AntennaPatternType.SECTOR_SCAN,
            "azimuth_width": azimuth_width,
            "elevation_width": elevation_width,
            "gain": gain,
            "pattern_function": self._generate_sector_pattern(azimuth_width, elevation_width)
        }
        
        logger.info(f"Created sector scan antenna: {name}, gain: {gain}dBi, " 
                   f"azimuth: {azimuth_width}°, elevation: {elevation_width}°")
        return antenna
    
    def calculate_gain(self, antenna_id: str, azimuth: float, elevation: float) -> float:
        """Calculate antenna gain at specific direction.
        
        Args:
            antenna_id: Antenna ID
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            
        Returns:
            Gain in dBi at specified direction
        """
        if antenna_id not in self.patterns:
            logger.warning(f"Antenna ID {antenna_id} not found in patterns")
            return 0.0
            
        pattern = self.patterns[antenna_id]
        pattern_func = pattern["pattern_function"]
        
        # Calculate relative gain (0-1) from pattern function
        relative_gain = pattern_func(azimuth, elevation)
        
        # Convert to dBi
        max_gain = pattern["gain"]
        if relative_gain <= 0:
            return -30.0  # Minimum gain (effectively zero)
        else:
            return max_gain + 10 * np.log10(relative_gain)
    
    def _generate_pencil_beam_pattern(self, beamwidth: float):
        """Generate a pencil beam pattern function."""
        def pattern_func(azimuth: float, elevation: float) -> float:
            # Calculate angular distance from boresight
            angular_distance = np.sqrt(azimuth**2 + elevation**2)
            
            # Gaussian beam pattern
            return np.exp(-2.77 * (angular_distance / beamwidth)**2)
            
        return pattern_func
    
    def _generate_sector_pattern(self, azimuth_width: float, elevation_width: float):
        """Generate a sector scan pattern function."""
        def pattern_func(azimuth: float, elevation: float) -> float:
            # Normalized distances from boresight
            az_norm = azimuth / (azimuth_width/2)
            el_norm = elevation / (elevation_width/2)
            
            # Separate patterns for azimuth and elevation
            if abs(az_norm) <= 1 and abs(el_norm) <= 1:
                # Within main beam
                az_pattern = np.cos(np.pi/2 * az_norm)**2
                el_pattern = np.cos(np.pi/2 * el_norm)**2
                return az_pattern * el_pattern
            else:
                # Side lobes (simplified)
                return 0.05 * np.exp(-0.5 * (abs(az_norm) + abs(el_norm)))
                
        return pattern_func