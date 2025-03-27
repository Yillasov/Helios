"""Data structures for cognitive radio environmental feedback."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class SpectrumBand:
    """Represents a frequency band in the spectrum."""
    start_freq: float  # Hz
    end_freq: float    # Hz
    center_freq: float = field(init=False)
    bandwidth: float = field(init=False)
    
    def __post_init__(self):
        self.center_freq = (self.start_freq + self.end_freq) / 2
        self.bandwidth = self.end_freq - self.start_freq
    
    def contains_frequency(self, freq: float) -> bool:
        """Check if a frequency is within this band."""
        return self.start_freq <= freq <= self.end_freq

@dataclass
class SpectrumOccupancy:
    """Represents the occupancy of a frequency band."""
    band: SpectrumBand
    power_level: float  # dBm
    noise_floor: float  # dBm
    snr: float = field(init=False)
    is_occupied: bool = field(init=False)
    occupation_threshold: float = 6.0  # dB above noise floor to consider occupied
    
    def __post_init__(self):
        self.snr = self.power_level - self.noise_floor
        self.is_occupied = self.snr > self.occupation_threshold

@dataclass
class EnvironmentState:
    """Represents the current RF environment state."""
    timestamp: float  # Simulation time
    spectrum_occupancy: Dict[str, SpectrumOccupancy] = field(default_factory=dict)
    interference_sources: List[Dict[str, Any]] = field(default_factory=list)
    overall_noise_level: float = -100.0  # dBm
    
    # Additional metrics
    average_snr: float = 0.0
    detected_signals: int = 0
    
    def get_band_occupancy(self, freq: float) -> Optional[SpectrumOccupancy]:
        """Get occupancy information for the band containing the specified frequency."""
        for occupancy in self.spectrum_occupancy.values():
            if occupancy.band.contains_frequency(freq):
                return occupancy
        return None
    
    def find_clear_bands(self, min_bandwidth: float = 0.0) -> List[SpectrumBand]:
        """Find unoccupied frequency bands with sufficient bandwidth."""
        clear_bands = []
        for occupancy in self.spectrum_occupancy.values():
            if not occupancy.is_occupied and occupancy.band.bandwidth >= min_bandwidth:
                clear_bands.append(occupancy.band)
        return clear_bands